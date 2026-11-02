"""Rewritten program: ring-structured init + fast pairwise-repulsion local search."""
import numpy as np
import time

N = 300


def _score(pts):
    """Return the common radius (score) for a point set, or -inf if invalid."""
    if pts is None:
        return -np.inf
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return -np.inf
    r = np.hypot(pts[:, 0], pts[:, 1])
    if np.max(r) > 1.0:
        return -np.inf
    d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    dmin = np.min(d)
    if not np.isfinite(dmin) or dmin <= 0.0:
        return -np.inf
    return min(dmin * 0.5, 1.0 - np.max(r))


def _ring_init(rng):
    """Concentric-ring initial layout: 1 center + 12 rings (25,26,25,26,25,26,25,26,25,26,25,25)."""
    counts = [25, 26, 25, 26, 25, 26, 25, 26, 25, 26, 25, 25]  # sum = 300
    pts = np.zeros((N, 2))
    idx = 0
    # center
    pts[idx] = [0.0, 0.0]
    idx += 1
    # radii: outer ring near the boundary
    radii = np.linspace(0.12, 0.94, len(counts))
    for count, rad in zip(counts, radii):
        base = rng.uniform(0, 2 * np.pi)
        thetas = base + np.linspace(0, 2 * np.pi, count, endpoint=False)
        for th in thetas:
            pts[idx] = [rad * np.cos(th), rad * np.sin(th)]
            idx += 1
    return pts


def _local_search(pts, max_time, rng):
    """Local search maximizing min(dmin/2, 1 - max_r) via gradient ascent on a smoothed score."""
    best = pts.copy()
    best_score = _score(best)
    t0 = time.time()

    # Adaptive step: start small
    step = 0.004
    patience = 0

    while time.time() - t0 < max_time:
        p = best
        # compute distances
        d = np.sqrt(np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=-1))
        np.fill_diagonal(d, np.inf)
        dmin = np.min(d)
        rmax = np.max(np.hypot(p[:, 0], p[:, 1]))
        score = min(dmin * 0.5, 1.0 - rmax)

        # Build gradient: for the bottleneck constraint, push points apart / inward
        # Use a soft version: weight pairs by proximity to dmin, and points by proximity to boundary
        # Soft-min over pair distances with temperature
        temp = 0.02
        # log-sum-exp style soft min of pair distances
        d_safe = np.where(d > 1e-9, d, 1e-9)
        w = np.exp(-(d_safe - dmin) / temp)  # weight peaks near the min
        np.fill_diagonal(w, 0.0)
        # gradient contribution from pair repulsion
        diff = p[:, None, :] - p[None, :, :]
        grad = np.einsum('ij,ijk->k', w, diff)  # sum over j: push i away from close j

        # boundary term: pull inward if near boundary
        r = np.hypot(p[:, 0], p[:, 1])
        btemp = 0.02
        bw = np.exp(-(1.0 - r) / btemp)  # weight peaks near boundary
        # inward gradient: -direction of position (unit)
        pos_unit = np.where(r > 1e-9, p / np.where(r > 1e-9, r[:, None], 1.0), 0.0)
        grad -= bw[:, None] * pos_unit

        # normalize gradient to avoid huge steps
        gn = np.linalg.norm(grad)
        if gn < 1e-12:
            break
        grad = grad / gn

        # Try a step in gradient direction (plus small random jitter)
        jitter = rng.normal(0, 0.3, size=grad.shape)
        cand = p + step * (grad + jitter)

        # Project back: keep inside unit disk (soft clamp)
        cr = np.hypot(cand[:, 0], cand[:, 1])
        mask = cr > 0.999
        if np.any(mask):
            cand[mask] = 0.999 * cand[mask] / cr[mask, None]

        cand_score = _score(cand)
        if cand_score > best_score + 1e-12:
            best = cand
            best_score = cand_score
            patience = 0
            # increase step slightly
            step = min(step * 1.05, 0.02)
        else:
            patience += 1
            step = max(step * 0.85, 0.0005)
            if patience > 40:
                # restart step size and jitter harder
                step = 0.003
                patience = 0

    return best, best_score


def construct():
    t_start = time.time()
    # Reserve 20s safety margin; use up to 120s for search
    max_search = 100.0
    deadline = t_start + max_search

    rng = np.random.default_rng(12345)

    best_pts = None
    best_score = -np.inf

    # Generate several ring-based inits with different random rotations/seeds
    n_inits = 6
    for k in range(n_inits):
        if time.time() > deadline - 5:
            break
        rng_k = np.random.default_rng(1000 + k)
        pts = _ring_init(rng_k)
        # Small random perturbation
        pts = pts + rng_k.normal(0, 0.002, size=pts.shape)
        # Quick polish with limited time per init
        remaining = deadline - time.time()
        if remaining < 2:
            break
        # Allocate search time across inits
        alloc = min(remaining / max(1, n_inits - k), 20.0)
        polished, sc = _local_search(pts, alloc, rng_k)
        if sc > best_score:
            best_score = sc
            best_pts = polished.copy()

    # Final: if we have time, do one more focused polish on the best
    remaining = deadline - time.time()
    if best_pts is not None and remaining > 3:
        polished, sc = _local_search(best_pts, remaining - 2.0, rng)
        if sc > best_score:
            best_score = sc
            best_pts = polished.copy()

    if best_pts is None:
        # Fallback: simple hex grid
        pts = _ring_init(np.random.default_rng(0))
        best_pts = pts

    # Final validity check
    final_score = _score(best_pts)
    if final_score < 0:
        # Shouldn't happen, but guard
        best_pts = _ring_init(np.random.default_rng(0))

    return best_pts
