import numpy as np
from scipy.spatial.distance import cdist
import time

N = 200
HALF = 0.5
_TIME_LIMIT = 145  # Leave 5s margin for 150s kill


def _score(pts):
    """Compute the maximum radius for a set of N points in [-0.5, 0.5]^2."""
    if pts.shape[0] != N or not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -HALF) or np.any(pts > HALF):
        return 0.0
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    wall = np.min([
        np.min(pts[:, 0]) + HALF,
        HALF - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + HALF,
        HALF - np.max(pts[:, 1]),
    ])
    r = min(min_pair / 2.0, wall)
    return r if r > 0 else 0.0


def _hex_points(s, ox, oy):
    """
    Generate hexagonal lattice points with spacing s and offset (ox, oy).
    Returns points inside [-0.5, 0.5]^2.
    """
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    cols = int(np.ceil(1.0 / dx)) + 4
    rows = int(np.ceil(1.0 / dy)) + 4
    
    pts = []
    for j in range(cols):
        x = -0.5 + j * dx + ox
        if x < -HALF - 1e-9 or x > HALF + 1e-9:
            continue
        yoff = (dy / 2.0) if (j % 2 == 1) else 0.0
        for i in range(rows):
            y = -0.5 + i * dy + oy + yoff
            if y < -HALF - 1e-9 or y > HALF + 1e-9:
                continue
            pts.append((x, y))
    
    arr = np.array(pts)
    if arr.shape[0] == 0:
        return arr
    keep = np.all(np.abs(arr) <= HALF + 1e-9, axis=1)
    return arr[keep]


def _greedy_select_fast(pts, N_target):
    """
    Faster greedy farthest-point selection using vectorized operations.
    """
    if len(pts) <= N_target:
        return pts.copy()
    
    n = len(pts)
    selected = []
    remaining_mask = np.ones(n, dtype=bool)
    
    # Start with point closest to origin
    d2_origin = np.sum(pts ** 2, axis=1)
    first = int(np.argmin(d2_origin))
    selected.append(first)
    remaining_mask[first] = False
    
    # min_dist for each point to the set of selected points
    min_dists = np.linalg.norm(pts - pts[first], axis=1)
    
    for _ in range(N_target - 1):
        # Among remaining, find max min_dist
        min_dists_sel = min_dists.copy()
        min_dists_sel[~remaining_mask] = -np.inf
        best_idx = int(np.argmax(min_dists_sel))
        
        if not remaining_mask[best_idx]:
            break
        
        selected.append(best_idx)
        remaining_mask[best_idx] = False
        
        # Update min_dists
        d = np.linalg.norm(pts - pts[best_idx], axis=1)
        min_dists = np.minimum(min_dists, d)
    
    return pts[selected]


def _coordinate_ascent_polish(c, iters=3000, step0=5e-3, seed=42):
    """
    Coordinate ascent polish: try small perturbations in each coordinate
    and keep if they improve the score.
    """
    c = c.copy()
    best = _score(c)
    rng = np.random.default_rng(seed)
    step = step0
    
    for it in range(iters):
        if time.time() > _TIME_LIMIT - 5:
            break
        
        # Pick a random point and coordinate
        i = int(rng.integers(N))
        d = int(rng.integers(2))
        
        improved = False
        for sgn in (1.0, -1.0):
            for sc in (step, step * 0.5, step * 2.0):
                c2 = c.copy()
                c2[i, d] += sgn * sc
                if not np.all(np.abs(c2) <= HALF + 1e-12):
                    continue
                v = _score(c2)
                if v > best + 1e-12:
                    c = c2
                    best = v
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            step *= 0.85
        else:
            step = min(step * 1.05, 8e-3)
    
    return c, best


def _local_refine(pts, n_iters=2000):
    """
    Local refinement: iteratively find the bottleneck (closest pair or wall)
    and adjust points to increase the minimum radius.
    """
    pts = pts.copy()
    
    for it in range(n_iters):
        if time.time() > _TIME_LIMIT - 5:
            break
        
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        
        # Find min pair
        min_pair_idx = np.unravel_index(np.argmin(D), D.shape)
        i, j = min_pair_idx
        d_pair = D[i, j]
        
        # Find min wall distance
        wall_dists = np.array([
            np.min(pts[:, 0]) + HALF,
            HALF - np.max(pts[:, 0]),
            np.min(pts[:, 1]) + HALF,
            HALF - np.max(pts[:, 1]),
        ])
        min_wall_val = np.min(wall_dists)
        min_wall_idx = np.argmin(wall_dists)
        
        # Learning rate
        lr = 0.005 / (1 + it * 0.005)
        
        # Determine which constraint is limiting
        if d_pair / 2.0 <= min_wall_val:
            # Push pair apart
            diff = pts[j] - pts[i]
            norm = np.linalg.norm(diff)
            if norm < 1e-10:
                pts[i, 0] += 0.001
                pts[j, 0] -= 0.001
            else:
                unit = diff / norm
                step = lr * min(1.0, 1.0 / norm)
                pts[i] -= step * unit
                pts[j] += step * unit
        else:
            # Push wall particle inward
            if min_wall_idx == 0:  # left wall
                idx = np.argmin(pts[:, 0])
                pts[idx, 0] += lr
            elif min_wall_idx == 1:  # right wall
                idx = np.argmax(pts[:, 0])
                pts[idx, 0] -= lr
            elif min_wall_idx == 2:  # bottom wall
                idx = np.argmin(pts[:, 1])
                pts[idx, 1] += lr
            else:  # top wall
                idx = np.argmax(pts[:, 1])
                pts[idx, 1] -= lr
        
        # Clamp to bounds
        pts[:, 0] = np.clip(pts[:, 0], -HALF + 1e-9, HALF - 1e-9)
        pts[:, 1] = np.clip(pts[:, 1], -HALF + 1e-9, HALF - 1e-9)
    
    return pts


def construct():
    t0 = time.time()
    best_r = 0.0
    best_pts = None
    
    rng = np.random.default_rng(2024)
    
    # Target spacing estimate for hex packing of 200 circles in unit square
    # Area per circle ~ 1/200, hex spacing s ~ sqrt(1/(200 * sqrt(3)/2)) ~ 0.076
    s0 = 1.0 / np.sqrt(N * np.sqrt(3.0) / 2.0)
    
    # Multiple hex-grid trials with random offsets and varied spacing
    n_trials = 50
    for trial in range(n_trials):
        if time.time() > _TIME_LIMIT - 30:
            break
        
        # Vary spacing slightly and use random offsets
        s = s0 * (0.97 + 0.06 * rng.random())
        ox = rng.uniform(-0.5, 0.5)
        oy = rng.uniform(-0.5, 0.5)
        
        cands = _hex_points(s, ox, oy)
        if cands.shape[0] < N:
            continue
        
        # Greedy select N points
        selected = _greedy_select_fast(cands, N)
        if selected.shape[0] < N:
            continue
        
        r = _score(selected)
        
        # Local refine
        refined = _local_refine(selected, n_iters=300)
        r_ref = _score(refined)
        
        if r_ref > best_r:
            best_r = r_ref
            best_pts = refined.copy()
        
        # Coordinate ascent polish
        if time.time() < _TIME_LIMIT - 15:
            polished, r_pol = _coordinate_ascent_polish(refined, iters=1000, seed=trial)
            if r_pol > best_r:
                best_r = r_pol
                best_pts = polished.copy()
    
    # If we have time, do more polish on the best solution
    if best_pts is not None and time.time() - t0 < _TIME_LIMIT - 10:
        polished, r_pol = _coordinate_ascent_polish(best_pts, iters=2000, seed=999)
        if r_pol > best_r:
            best_r = r_pol
            best_pts = polished.copy()
    
    # Final local refinement
    if best_pts is not None and time.time() - t0 < _TIME_LIMIT - 5:
        final_pts = _local_refine(best_pts, n_iters=500)
        r_final = _score(final_pts)
        if r_final > best_r:
            best_r = r_final
            best_pts = final_pts
    
    if best_pts is None:
        # Fallback: simple grid
        rows, cols = 14, 15
        xs = np.linspace(-0.45, 0.45, cols)
        ys = np.linspace(-0.45, 0.45, rows)
        grid = np.array([(x, y) for y in ys for x in xs])
        best_pts = grid[:N]
    
    # Final validity check and clamp
    best_pts = np.clip(best_pts, -HALF, HALF)
    
    return best_pts
