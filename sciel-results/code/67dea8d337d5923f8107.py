"""
Pack 50 equal circles in the unit disk.

Strategy:
  1. Start from a dense hex lattice and keep the 50 points closest to the origin.
  2. Run a projected subgradient ascent on the log-sum-exp surrogate of the
     min of all pairwise half-distances and container margins.  Each step is
     clipped to the unit disk and to a fixed per-point move budget to keep the
     iterates well-conditioned.
  3. Periodically perform a "bottleneck repair": identify the pair of points (or
     a point and the boundary) that currently has the smallest slack and push
     those two points apart in the direction that increases that slack.
  4. Use several random restarts and keep the best configuration found.
"""
import numpy as np
import time

N = 50
DEADLINE = 145.0


def _compute_score(points):
    """Score = min(min pair distance/2, min container margin)."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists_origin = np.linalg.norm(points, axis=1)
    container_margin = 1.0 - np.max(dists_origin)
    if container_margin < 0:
        return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_half = np.min(dists) / 2.0
    return min(min_pair_half, container_margin)


def _project_to_disk(points):
    """Clip points so that |p_i| <= 1 - eps."""
    eps = 1e-9
    r = np.linalg.norm(points, axis=1)
    scale = np.minimum(1.0, (1.0 - eps) / np.maximum(r, eps))
    return points * scale[:, np.newaxis]


def _hex_init(radius_bound=0.85, spacing=0.22):
    """Return 50 points from a hex lattice inside the disk."""
    pts = []
    dx = spacing
    dy = spacing * np.sqrt(3) / 2
    max_coord = radius_bound / dx + 2
    for i in range(-int(max_coord), int(max_coord) + 1):
        for j in range(-int(max_coord), int(max_coord) + 1):
            x = i * dx + (j % 2) * dx / 2
            y = j * dy
            if x * x + y * y <= radius_bound * radius_bound:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) < N:
        ang = np.random.uniform(0, 2 * np.pi, N - len(pts))
        rad = np.sqrt(np.random.uniform(0, radius_bound ** 2, N - len(pts)))
        extra = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
        pts = np.vstack([pts, extra])
    else:
        d = np.linalg.norm(pts, axis=1)
        idx = np.argsort(d)[:N]
        pts = pts[idx]
    return pts


def _soft_min_value_and_subgrad(points, beta):
    """
    Value and subgradient of the soft-min of all slacks.
    slacks = [ d_ij/2 for i<j, 1-|p_i| for i ].
    We return (soft_min_value, subgrad of negative soft-min for maximization).
    Actually we want to maximize soft_min(slacks), so we minimize -soft_min(slacks).
    grad of -soft_min wrt p is -sum_k w_k * (d slack_k / dp),
    where w_k = exp(-beta*slack_k) / sum_j exp(-beta*slack_j).
    """
    pts = points
    # container slacks
    r = np.linalg.norm(pts, axis=1)
    c_slacks = 1.0 - r  # shape (N,)

    # pairwise slacks
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    pair_slacks = d[mask] / 2.0  # shape (M,)

    all_slacks = np.concatenate([c_slacks, pair_slacks])
    # For numerical stability
    m = np.min(all_slacks)
    exps = np.exp(-beta * (all_slacks - m))
    w = exps / np.sum(exps)

    val = -m - np.log(np.sum(exps)) / beta

    # Build subgradient of -soft_min
    # d soft_min / dp_i = sum_k w_k * d slack_k / dp_i
    # d(-soft_min)/dp_i = -sum_k w_k * d slack_k / dp_i
    subgrad = np.zeros_like(pts)

    # Container part: slack_k = 1 - |p_k|, d/dp_k = -p_k / |p_k|
    wc = w[:N]
    for i in range(N):
        if r[i] > 1e-12:
            subgrad[i] -= wc[i] * (-pts[i] / r[i])  # minus of negative => plus

    # Pair part: slack for pair (i,j) = d_ij/2
    # d/dp_i = (p_i - p_j) / (2 d_ij)  ... wait:
    # d_ij = |p_i - p_j|, d slack / dp_i = (p_i - p_j) / (2 d_ij)
    # d(-soft_min)/dp_i = -sum_k w_k * d slack_k / dp_i
    # So for pair (i,j): contribution to subgrad[i] is -w_pair * (p_i - p_j)/(2 d)
    # and to subgrad[j] is -w_pair * (p_j - p_i)/(2 d) = +w_pair * (p_i - p_j)/(2 d)
    wp = w[N:]
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            d_ij = d[i, j]
            if d_ij > 1e-12:
                vec = (pts[i] - pts[j]) / (2.0 * d_ij)
                subgrad[i] -= wp[idx] * vec
                subgrad[j] += wp[idx] * vec
            idx += 1

    return val, subgrad


def _bottleneck_repair(pts, target_r, max_iter=80, step=0.015):
    """
    Try to increase the minimum slack toward target_r by moving the
    currently bottleneck pair (or point-boundary) apart.
    """
    for _ in range(max_iter):
        # Compute all slacks
        r = np.linalg.norm(pts, axis=1)
        c_slacks = 1.0 - r
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        pair_slacks = d[mask] / 2.0

        all_slacks = np.concatenate([c_slacks, pair_slacks])
        min_slack = np.min(all_slacks)
        if min_slack >= target_r:
            break

        # Find which slack is minimal
        min_idx = np.argmin(all_slacks)
        if min_idx < N:
            # container violation for point i
            i = min_idx
            if r[i] < 1e-12:
                # move outward? Actually if r[i]==0 slack=1 which is large.
                # If r[i] near 1, push inward.
                pts[i] *= (target_r / 1.0) if target_r > 0 else 0.9
            else:
                # Move point toward origin to increase 1-|p|
                pts[i] -= step * (pts[i] / r[i])
        else:
            # pair violation
            k = min_idx - N
            # find i,j from k
            # k-th upper triangle entry: i = k - j(j+1)/2 - 1 ... simpler loop
            i = j = None
            count = 0
            for ii in range(N):
                for jj in range(ii + 1, N):
                    if count == k:
                        i, j = ii, jj
                        break
                    count += 1
                if i is not None:
                    break
            if i is None:
                break
            d_ij = np.linalg.norm(pts[i] - pts[j])
            if d_ij < 1e-12:
                pts[i] += np.array([step, 0.0])
                pts[j] -= np.array([step, 0.0])
            else:
                vec = (pts[i] - pts[j]) / d_ij
                pts[i] += step * 0.5 * vec
                pts[j] -= step * 0.5 * vec

        pts = _project_to_disk(pts)

    return pts


def _run_subgradient(pts, beta_start=20.0, beta_end=80.0,
                     step_start=0.04, step_end=0.005,
                     max_iter=3000, time_budget=30.0):
    t0 = time.time()
    best = _compute_score(pts)
    best_pts = pts.copy()
    n_iters = max_iter

    for it in range(n_iters):
        if time.time() - t0 > time_budget:
            break
        frac = it / max(1, n_iters - 1)
        beta = beta_start + (beta_end - beta_start) * frac
        step = step_start + (step_end - step_start) * frac

        val, subgrad = _soft_min_value_and_subgrad(pts, beta)
        pts = pts + step * subgrad
        pts = _project_to_disk(pts)

        # occasional repair
        if it % 50 == 0:
            cur = _compute_score(pts)
            pts = _bottleneck_repair(pts, cur + 0.005, max_iter=40, step=0.012)
            pts = _project_to_disk(pts)

        score = _compute_score(pts)
        if score > best:
            best = score
            best_pts = pts.copy()

    return best_pts, best


def construct():
    start = time.time()
    deadline = start + DEADLINE

    rng = np.random.default_rng(42)

    best_score = 0.0
    best_pts = None

    # Multiple restarts
    n_restarts = 6
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break

        if restart == 0:
            pts = _hex_init(radius_bound=0.85, spacing=0.21)
        elif restart == 1:
            pts = _hex_init(radius_bound=0.82, spacing=0.24)
        else:
            # random-ish: hex + jitter
            base = _hex_init(radius_bound=0.80, spacing=0.23)
            pts = base + rng.normal(0, 0.03, base.shape)
            pts = _project_to_disk(pts)

        time_left = deadline - 5 - (time.time() - start)
        budget = max(8.0, min(35.0, time_left / max(1, n_restarts - restart)))

        # Phase 1: subgradient ascent
        pts, score = _run_subgradient(pts, time_budget=budget * 0.6)

        # Phase 2: bottleneck repair loop with decreasing target
        cur = _compute_score(pts)
        target = cur + 0.01
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            pts = _bottleneck_repair(pts, target, max_iter=60, step=0.01)
            new_score = _compute_score(pts)
            if new_score > cur:
                cur = new_score
                target = cur + 0.008
            else:
                target = cur + 0.004

        # Phase 3: short subgradient polish
        if time.time() < deadline - 5:
            pts, score2 = _run_subgradient(pts, beta_start=60.0, beta_end=120.0,
                                           step_start=0.015, step_end=0.003,
                                           max_iter=800,
                                           time_budget=min(10.0, deadline - 5 - (time.time() - start)))

        s = _compute_score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    if best_pts is None:
        best_pts = _hex_init()

    # Final safety: ensure validity
    best_pts = _project_to_disk(best_pts)
    return best_pts
