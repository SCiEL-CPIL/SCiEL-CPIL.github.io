"""Optimized 50-sphere packing in unit cube."""
import time
import numpy as np
from scipy.spatial.distance import pdist

N = 50
_T0 = time.time()
_DEADLINE = 145.0  # safety margin below the 150 s wall-clock limit


def _score(pts):
    """Exact radius: min(pair_dist/2, wall_dist)."""
    d = pdist(pts)
    min_pair = d.min() if d.size > 0 else np.inf
    min_wall = np.min(0.5 - np.abs(pts))
    return min(min_pair * 0.5, min_wall)


def _soft_min_and_grad(pts, beta=200.0):
    """Vectorized soft-min of all constraints (pair dists/2 and wall dists).
    Returns (soft_min_value, gradient w.r.t. pts)."""
    # Pairwise distances
    d = pdist(pts)
    n_pairs = len(d)
    
    # Wall distances
    wall = 0.5 - np.abs(pts)  # (N, 3)
    n_walls = N * 3
    n_constr = n_pairs + n_walls
    
    # All constraint values
    vals = np.empty(n_constr)
    vals[:n_pairs] = d * 0.5
    vals[n_pairs:] = wall.ravel()
    
    # Soft-min: -logsumexp(-beta * vals) / beta
    neg_beta_x = -beta * vals
    m = neg_beta_x.max()
    exp_shifted = np.exp(neg_beta_x - m)
    logsumexp_val = m + np.log(exp_shifted.sum())
    soft_min_val = -logsumexp_val / beta
    
    # Weights (softmax of -beta*vals)
    weights = exp_shifted / exp_shifted.sum()
    
    # Build full distance matrix for vectorized pair gradient
    dist_matrix = np.zeros((N, N))
    idx = np.triu_indices(N, k=1)
    dist_matrix[idx] = d
    dist_matrix = dist_matrix + dist_matrix.T
    
    # Pair gradient contribution: for each pair (a,b) with weight w,
    # d(soft_min)/d(pts[a]) += w * (pts[a]-pts[b]) / dist(a,b)
    # d(soft_min)/d(pts[b]) -= w * (pts[a]-pts[b]) / dist(a,b)
    # Vectorize: for each point a, sum over b of w_ab * (pts[a]-pts[b]) / dist_ab
    pair_w = weights[:n_pairs]
    
    # Build weight matrix (symmetric)
    wmat = np.zeros((N, N))
    wmat[idx] = pair_w
    wmat = wmat + wmat.T
    
    # Avoid division by zero on diagonal
    dist_safe = dist_matrix.copy()
    np.fill_diagonal(dist_safe, 1.0)
    
    # diff matrix: diff[a,b] = pts[a] - pts[b]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    
    # Contribution per point: sum_b w[a,b] * diff[a,b] / dist[a,b]
    # weight per element: wmat / dist_safe
    w_over_d = wmat / dist_safe  # (N, N)
    # Multiply into diff: (N, N, 3) * (N, N, 1)
    contrib = diff * w_over_d[:, :, np.newaxis]
    grad_pair = contrib.sum(axis=1)  # (N, 3)
    
    # Wall gradient: for point j, axis k: weight * (-sign(pts[j,k]))
    wall_w = weights[n_pairs:]  # (N*3,)
    wall_w_mat = wall_w.reshape(N, 3)
    signs = np.sign(pts)
    # Where pts is ~0, sign is 0, which is fine (subgradient)
    grad_wall = wall_w_mat * (-signs)
    
    grad = grad_pair + grad_wall
    return soft_min_val, grad


def _line_search(pts, grad, r_cur, max_step=0.05, min_step=1e-6, max_iter=20):
    """Backtracking line search: find step that improves r."""
    # Normalize gradient
    gn = np.linalg.norm(grad)
    if gn < 1e-15:
        return pts, r_cur
    grad = grad / gn
    
    step = max_step
    for _ in range(max_iter):
        cand = pts + step * grad
        cand_r = _score(cand)
        if cand_r >= r_cur - 1e-12:
            return cand, cand_r
        step *= 0.5
        if step < min_step:
            break
    return pts, r_cur


def _gradient_ascent(pts, iters, rng, beta_start=100.0, beta_end=2000.0):
    """Gradient ascent with line search and increasing beta."""
    cur = pts.copy()
    cur_r = _score(cur)
    best = cur.copy()
    best_r = cur_r
    
    for i in range(iters):
        if time.time() > _DEADLINE:
            break
        frac = i / max(iters, 1)
        beta = beta_start * (beta_end / beta_start) ** frac
        
        soft_val, g = _soft_min_and_grad(cur, beta=beta)
        
        # Small noise to escape flat regions
        g = g + rng.normal(0.0, 0.001, size=cur.shape)
        
        # Line search
        cand, cand_r = _line_search(cur, g, cur_r, max_step=0.03 * (1.0 - 0.5 * frac))
        
        if cand_r > cur_r:
            cur = cand
            cur_r = cand_r
            if cur_r > best_r:
                best_r = cur_r
                best = cur.copy()
    
    return best, best_r


def _anneal(pts, T0, T1, steps, rng):
    """Simulated annealing."""
    cur = pts.copy()
    cur_r = _score(cur)
    best = cur.copy()
    best_r = cur_r
    
    for i in range(steps):
        if time.time() > _DEADLINE:
            break
        T = T0 * (T1 / T0) ** (i / steps)
        kick = 0.015 * (T / T0 + 0.1)
        n_move = max(1, N // 4)
        idx = rng.choice(N, size=n_move, replace=False)
        cand = cur.copy()
        cand[idx] += rng.normal(0.0, kick, size=(n_move, 3))
        cand_r = _score(cand)
        if cand_r > cur_r or rng.random() < np.exp((cand_r - cur_r) / max(T, 1e-10)):
            cur = cand
            cur_r = cand_r
            if cur_r > best_r:
                best_r = cur_r
                best = cur.copy()
    
    return best, best_r


def _polish(pts, max_iters=2000, rng=None):
    """High-precision polish with very high beta and tiny steps."""
    if rng is None:
        rng = np.random.default_rng(99)
    cur = pts.copy()
    cur_r = _score(cur)
    best = cur.copy()
    best_r = cur_r
    
    for i in range(max_iters):
        if time.time() > _DEADLINE:
            break
        frac = i / max(max_iters, 1)
        beta = 5000.0 * (1.0 + frac)  # 5000 to 10000
        step_max = 0.002 * (1.0 - 0.8 * frac)
        
        soft_val, g = _soft_min_and_grad(cur, beta=beta)
        g = g + rng.normal(0.0, 0.0003, size=cur.shape)
        
        cand, cand_r = _line_search(cur, g, cur_r, max_step=step_max, min_step=1e-8)
        
        if cand_r > cur_r:
            cur = cand
            cur_r = cand_r
            if cur_r > best_r:
                best_r = cur_r
                best = cur.copy()
    
    return best, best_r


def _make_starts(rng):
    """Generate diverse starting configurations."""
    starts = []
    
    # 5x5x2 slab: 50 points exactly
    m2 = 5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    slab = np.array([(x, y, z) for z in ticks2[:2] for y in ticks2 for x in ticks2])
    starts.append(slab.copy())
    
    # 4x4x4 grid, take 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    starts.append(grid[:N].copy())
    
    # Jittered 5x5x2 slab
    s = slab + rng.normal(0.0, 0.01, size=slab.shape)
    s = np.clip(s, -0.5 + 1e-6, 0.5 - 1e-6)
    starts.append(s)
    
    # Random valid start: place points in a 5x5x2 grid with random offsets
    # Use a body-centered arrangement
    for _ in range(2):
        s = slab + rng.normal(0.0, 0.02, size=slab.shape)
        s = np.clip(s, -0.5 + 1e-6, 0.5 - 1e-6)
        starts.append(s)
    
    return starts


def construct():
    rng = np.random.default_rng(42)
    
    starts = _make_starts(rng)
    
    best = starts[0].copy()
    best_r = _score(best)
    
    for s in starts:
        if time.time() > _DEADLINE:
            break
        
        # Phase 1: Annealing to escape local minima
        a, ar = _anneal(s, T0=0.02, T1=0.0002, steps=2000, rng=rng)
        if ar > best_r:
            best_r, best = ar, a
        
        if time.time() > _DEADLINE:
            break
        
        # Phase 2: Gradient ascent with line search
        g, gr = _gradient_ascent(a, iters=1500, rng=rng, beta_start=100.0, beta_end=2000.0)
        if gr > best_r:
            best_r, best = gr, g
    
    # Phase 3: Final high-precision polish
    if time.time() < _DEADLINE - 5:
        p, pr = _polish(best, max_iters=3000, rng=rng)
        if pr > best_r:
            best_r, best = pr, p
    
    # Safety: ensure all centres are strictly inside the cube
    eps = 1e-12
    best = np.clip(best, -0.5 + eps, 0.5 - eps)
    
    return best
