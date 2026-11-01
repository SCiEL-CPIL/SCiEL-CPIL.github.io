"""Packing 50 equal circles in the unit square [-0.5, 0.5]^2."""
N = 50
import time
import numpy as np
from scipy.spatial import cKDTree


def _score(pts):
    """Return the achievable common radius for a centre array (N,2)."""
    # Boundary term: distance from each centre to the nearest side.
    d_bound = np.min(0.5 - np.abs(pts), axis=1)
    r = float(np.min(d_bound))
    # Pairwise term: half the minimum centre distance.
    if N > 1:
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=2)
        dmin = float(np.min(d[:, 1]))
        r = min(r, dmin / 2.0)
    return r


def _valid(pts):
    return (pts.shape == (N, 2)
            and np.all(np.isfinite(pts))
            and np.all(pts >= -0.5 - 1e-9)
            and np.all(pts <= 0.5 + 1e-9))


def _hex_initial():
    """Hexagonal (triangular) lattice, rows staggered, clipped to square."""
    # Choose a lattice spacing that yields roughly 50 points inside the square.
    # Area per lattice point ~ (sqrt(3)/2)*s^2 for spacing s.
    # Want 50 points in area 1 -> s ~ sqrt(2/(50*sqrt(3))) ~ 0.153.
    s = 0.152
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    xs = np.arange(-0.5, 0.5 + 1e-9, dx)
    pts = []
    row = 0
    y = -0.5
    while y <= 0.5 + 1e-9:
        offset = (row % 2) * (dx / 2.0)
        for x in xs:
            xx = x + offset
            if -0.5 <= xx <= 0.5 and -0.5 <= y <= 0.5:
                pts.append([xx, y])
        y += dy
        row += 1
    pts = np.array(pts, dtype=float)
    if len(pts) < N:
        # Pad with a fine grid to reach N (shouldn't happen with s above).
        grid = _grid_initial()
        pts = np.vstack([pts, grid[len(pts):]])
    else:
        # Keep N points closest to the centre to maximise boundary margin.
        d2 = np.sum(pts * pts, axis=1)
        order = np.argsort(d2)
        pts = pts[order[:N]]
    return pts.copy()


def _grid_initial():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _perturb_and_refine(pts, rng, step0, steps, cool, time_budget):
    """Local search: random perturbations accepted if score improves, with cooling."""
    best = pts.copy()
    best_s = _score(best)
    step = step0
    t0 = time.time()
    for it in range(steps):
        if time.time() - t0 > time_budget:
            break
        cand = best.copy()
        # Perturb a random subset of points.
        n_move = rng.integers(1, max(2, N // 4) + 1)
        idx = rng.choice(N, size=n_move, replace=False)
        cand[idx] += rng.normal(0.0, step, size=(n_move, 2))
        # Clamp to square (with small margin to avoid boundary violations).
        cand = np.clip(cand, -0.5, 0.5)
        s = _score(cand)
        if s > best_s + 1e-15:
            best = cand
            best_s = s
            step = max(step0 * 0.5, step * 1.0)  # keep step, small boost
        else:
            step *= cool
        step = min(step, step0)
    return best, best_s


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    rng = np.random.default_rng(12345)

    # Initial candidates: hexagonal and grid.
    candidates = [_hex_initial(), _grid_initial()]
    best_pts = candidates[0].copy()
    best_s = _score(best_pts)
    for c in candidates[1:]:
        s = _score(c)
        if s > best_s:
            best_s = s
            best_pts = c.copy()

    # Multi-restart local optimization with random perturbations.
    step0 = 0.05
    steps_per_restart = 4000
    cool = 0.995
    restarts = 12

    for r in range(restarts):
        if time.time() > deadline:
            break
        # Fresh start: either best with jitter, or a fresh random-ish layout.
        if r == 0:
            cur = best_pts.copy()
        elif r % 3 == 1:
            cur = _hex_initial()
            # Add small random jitter to break symmetry.
            cur += rng.normal(0.0, 0.02, size=(N, 2))
            cur = np.clip(cur, -0.5, 0.5)
        elif r % 3 == 2:
            cur = _grid_initial()
            cur += rng.normal(0.0, 0.02, size=(N, 2))
            cur = np.clip(cur, -0.5, 0.5)
        else:
            cur = best_pts.copy()
            cur += rng.normal(0.0, 0.03, size=(N, 2))
            cur = np.clip(cur, -0.5, 0.5)

        # Ensure validity (finite, in square).
        if not _valid(cur):
            cur = best_pts.copy()

        cur, s = _perturb_and_refine(cur, rng, step0, steps_per_restart, cool,
                                     time_budget=max(0.1, deadline - time.time()))
        if _valid(cur) and s > best_s:
            best_s = s
            best_pts = cur.copy()

    # Final safety: clamp and validate.
    best_pts = np.clip(best_pts, -0.5, 0.5)
    if not _valid(best_pts):
        best_pts = _grid_initial()

    return best_pts
