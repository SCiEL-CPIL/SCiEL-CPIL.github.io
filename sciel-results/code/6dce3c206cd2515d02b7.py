"""Simulated Annealing packing of 300 equal circles in a unit circle."""
import numpy as np
import time

N = 300
TIME_LIMIT = 118.0  # safety margin before 150s kill


def _hex_init(n=N):
    """Dense hexagonal lattice clipped to the unit disk, keep n closest to center."""
    spacing = 0.095
    k = int(1.15 / spacing) + 2
    xs, ys = [], []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xs.append(x)
            ys.append(y)
    pts = np.column_stack([xs, ys])
    r = np.hypot(pts[:, 0], pts[:, 1])
    inside = r <= 1.0
    pts = pts[inside]
    r = r[inside]
    if len(pts) < n:
        return _hex_init_fallback(n)
    order = np.argsort(r, kind="stable")
    return pts[order[:n]].copy()


def _hex_init_fallback(n=N):
    spacing = 0.08
    k = int(1.2 / spacing) + 2
    xs, ys = [], []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xs.append(x)
            ys.append(y)
    pts = np.column_stack([xs, ys])
    r = np.hypot(pts[:, 0], pts[:, 1])
    inside = r <= 1.0
    pts = pts[inside]
    r = r[inside]
    order = np.argsort(r, kind="stable")
    return pts[order[:n]].copy()


def _score(pts):
    """Achieved common radius (min over pairs and boundary)."""
    n = len(pts)
    d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2)) / 2.0
    min_bound = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair, min_bound)


def _anneal(pts, start_time, time_limit):
    """Simulated Annealing to maximize the minimum separation."""
    n = len(pts)
    pts = pts.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Annealing parameters
    T = 0.05  # initial temperature (scale of perturbations)
    T_min = 1e-7
    cooling_rate = 0.9995  # per iteration
    max_iter = 50000
    
    it = 0
    rng = np.random.default_rng(42)
    
    while it < max_iter:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Compute current score and identify the "stressed" circle
        # The circle involved in the minimum-distance pair is the one to perturb
        d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
        np.fill_diagonal(d2, np.inf)
        min_d2 = np.min(d2)
        min_idx = np.argmin(d2)
        i, j = np.unravel_index(min_idx, (n, n))
        
        rad = np.hypot(pts[:, 0], pts[:, 1])
        min_bound = 1.0 - np.max(rad)
        current_score = min(np.sqrt(min_d2) / 2.0, min_bound)
        
        # Decide which circle to perturb: the one in the min pair, or a boundary-violating one
        # Perturb circle i (or j, randomly chosen from the pair)
        if rng.random() < 0.5:
            idx = i
        else:
            idx = j
        
        # Generate a random perturbation
        # Scale perturbation by temperature
        scale = T * 0.1
        dx = rng.normal(0, scale)
        dy = rng.normal(0, scale)
        
        new_pts = pts.copy()
        new_pts[idx, 0] += dx
        new_pts[idx, 1] += dy
        
        # Enforce boundary constraint
        new_rad = np.hypot(new_pts[idx, 0], new_pts[idx, 1])
        if new_rad > 1.0:
            new_pts[idx] = new_pts[idx] / new_rad
        
        # Compute new score
        new_d2 = np.sum(new_pts * new_pts, axis=1)[:, None] - 2.0 * (new_pts @ new_pts.T) + np.sum(new_pts * new_pts, axis=1)[None, :]
        np.fill_diagonal(new_d2, np.inf)
        new_min_d2 = np.min(new_d2)
        new_rad = np.hypot(new_pts[:, 0], new_pts[:, 1])
        new_min_bound = 1.0 - np.max(new_rad)
        new_score = min(np.sqrt(new_min_d2) / 2.0, new_min_bound)
        
        # Acceptance criterion (Metropolis)
        delta = new_score - current_score
        if delta > 0 or rng.random() < np.exp(delta / T):
            pts = new_pts
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        
        # Cool down
        T = max(T * cooling_rate, T_min)
        
        it += 1
        if it % 1000 == 0:
            # Periodic full re-evaluation to catch any drift
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    pts = _hex_init(N)
    # ensure valid start: clip inside
    rad = np.hypot(pts[:, 0], pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        pts[out] = pts[out] / rad[out, None]

    base_score = _score(pts)
    best_pts = pts.copy()
    best_score = base_score

    try:
        opt_pts, opt_score = _anneal(pts, start_time, TIME_LIMIT)
        if opt_score > best_score:
            best_pts = opt_pts
            best_score = opt_score
    except Exception:
        pass

    # final safety: ensure finite, inside, shape (N,2)
    best_pts = np.asarray(best_pts, dtype=float)
    if best_pts.shape != (N, 2):
        best_pts = _hex_init(N)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N)
    rad = np.hypot(best_pts[:, 0], best_pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        best_pts[out] = best_pts[out] / rad[out, None]

    return best_pts
