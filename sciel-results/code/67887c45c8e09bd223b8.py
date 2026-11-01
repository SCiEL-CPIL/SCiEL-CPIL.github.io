"""Improved packer: dense distance matrix + fast simulated annealing."""
import numpy as np
import time

N = 300
SEED = 12345
T_START = time.time()
DEADLINE = 110.0  # safety margin before the 120 s wall-clock limit


def _init_dist(points):
    """Compute initial pairwise distance matrix."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    return dists


def _update_dist_row(dist, i, new_pt, points):
    """Update row and column i of the distance matrix."""
    # dist[i, :] = norm(points[i] - points)
    # dist[:, i] = norm(points - points[i])
    diff = points - new_pt
    dists = np.sqrt(np.sum(diff * diff, axis=1))
    dist[i, :] = dists
    dist[:, i] = dists


def _score_fast(points, dist):
    """Return the feasible common radius (0.0 if invalid)."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # distance to container boundary
    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - np.max(norm)
    if r_bound < 0.0:
        return 0.0
    # minimum pairwise half-distance
    # Note: dist has 0 on diagonal
    r_pair = 0.5 * np.min(dist)
    r = min(r_bound, r_pair)
    return float(r) if r > 0.0 else 0.0


def _lattice_seed(spacing):
    """Generate a hexagonal lattice and pick N points inside the disk."""
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        keep = rad <= 1.0 - spacing / 2.0
        pts = pts[keep]
        if len(pts) >= N:
            # prefer inner points (smaller radius) to maximise margin
            order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
            return pts[order[:N]].copy()
        spacing *= 0.97


def _anneal(points, dist, t_budget, rng):
    """Simulated annealing with dense distance matrix updates."""
    start = time.time()
    best = points.copy()
    best_dist = dist.copy()
    best_r = _score_fast(best, best_dist)
    cur = points.copy()
    cur_dist = dist.copy()
    cur_r = best_r

    # initial temperature from typical pair distance
    d0 = np.min(cur_dist)
    T = 0.02 * d0
    T_min = 1e-5 * d0
    step = 0.05 * d0

    while time.time() - start < t_budget and T > T_min:
        # propose a random move on a random point
        idx = rng.integers(N)
        ang = rng.random() * 2.0 * np.pi
        mag = rng.random() * step
        dx = mag * np.cos(ang)
        dy = mag * np.sin(ang)

        new_pt = cur[idx].copy()
        new_pt[0] += dx
        new_pt[1] += dy

        # reject if it leaves the container (with margin)
        if np.hypot(new_pt[0], new_pt[1]) > 1.0 - 1e-9:
            T *= 0.999
            continue

        old_pt = cur[idx].copy()
        old_row = cur_dist[idx].copy()
        cur[idx] = new_pt
        _update_dist_row(cur_dist, idx, new_pt, cur)
        new_r = _score_fast(cur, cur_dist)

        if new_r >= cur_r or rng.random() < np.exp((new_r - cur_r) / max(T, 1e-12)):
            cur_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = cur.copy()
                best_dist = cur_dist.copy()
        else:
            cur[idx] = old_pt
            cur_dist[idx, :] = old_row
            cur_dist[:, idx] = old_row

        T *= 0.999
        step *= 0.9999

    return best, best_dist, best_r


def _polish(points, dist, t_budget, rng):
    """High-precision polish pass: tighter tolerance, more iterations, local solve."""
    start = time.time()
    best = points.copy()
    best_dist = dist.copy()
    best_r = _score_fast(best, best_dist)
    cur = points.copy()
    cur_dist = dist.copy()
    cur_r = best_r

    # Smaller step size for polishing
    d0 = np.min(cur_dist)
    step = 0.01 * d0
    min_step = 1e-6 * d0

    while time.time() - start < t_budget:
        # Focus on points that are "tight" (close to boundary or neighbors)
        norms = np.linalg.norm(cur, axis=1)
        boundary_margin = 1.0 - norms
        min_neighbor = np.min(cur_dist, axis=1) # min distance to any other point
        
        # Points with small margin are candidates for improvement
        tight_mask = (boundary_margin < 0.01 * d0) | (min_neighbor < 0.01 * d0)
        tight_idx = np.where(tight_mask)[0]

        if len(tight_idx) == 0:
            # No tight points, do a random move
            idx = rng.integers(N)
        else:
            idx = tight_idx[rng.integers(len(tight_idx))]

        ang = rng.random() * 2.0 * np.pi
        mag = rng.random() * step
        dx = mag * np.cos(ang)
        dy = mag * np.sin(ang)

        new_pt = cur[idx].copy()
        new_pt[0] += dx
        new_pt[1] += dy

        if np.hypot(new_pt[0], new_pt[1]) > 1.0 - 1e-9:
            continue

        old_pt = cur[idx].copy()
        old_row = cur_dist[idx].copy()
        cur[idx] = new_pt
        _update_dist_row(cur_dist, idx, new_pt, cur)
        new_r = _score_fast(cur, cur_dist)

        if new_r >= cur_r:
            cur_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = cur.copy()
                best_dist = cur_dist.copy()
            step *= 0.999
        else:
            cur[idx] = old_pt
            cur_dist[idx, :] = old_row
            cur_dist[:, idx] = old_row
            step *= 1.01

        if step < min_step:
            step = min_step
        if step > 0.1 * d0:
            step = 0.1 * d0

    return best, best_dist, best_r


def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # initial hex-lattice seed
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    pts = _lattice_seed(spacing)
    dist = _init_dist(pts)
    best = pts.copy()
    best_dist = dist.copy()
    best_r = _score_fast(best, best_dist)

    # Phase 1: Multi-start annealing within the time budget
    while time.time() - t0 < DEADLINE - 20.0:  # reserve 20s for polishing
        # small perturbation of the seed for diversity
        jitter = 0.01 * rng.standard_normal((N, 2))
        cand = best + jitter
        # clip inside container
        rad = np.linalg.norm(cand, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            cand[mask] = cand[mask] / rad[mask, None] * (1.0 - 1e-6)
        
        cand_dist = _init_dist(cand)

        t_left = DEADLINE - 20.0 - (time.time() - t0)
        if t_left < 5.0:
            break
        t_alloc = min(t_left - 2.0, 15.0)  # cap per-anneal budget
        cand, cand_dist, r = _anneal(cand, cand_dist, t_alloc, rng)
        if r > best_r:
            best_r = r
            best = cand
            best_dist = cand_dist

    # Phase 2: High-precision polish pass
    t_left = DEADLINE - (time.time() - t0)
    if t_left > 5.0:
        best, best_dist, best_r = _polish(best, best_dist, t_left - 2.0, rng)

    # final safety: ensure valid output
    final_r = _score_fast(best, best_dist)
    if final_r <= 0.0:
        # fall back to a guaranteed-valid lattice
        best = _lattice_seed(2.0 / (np.sqrt(N) + 1.0))

    return best
