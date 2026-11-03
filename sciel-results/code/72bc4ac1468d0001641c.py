"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial.distance import pdist

_T0 = time.time()
_DEADLINE = 115.0  # safety margin below the 120 s wall-clock limit


def _score(pts):
    """Exact radius for a centre array: min(pair_dist/2, wall_dist)."""
    d = pdist(pts)
    min_pair = d.min() if d.size > 0 else np.inf
    min_wall = np.min(0.5 - np.abs(pts))
    return min(min_pair * 0.5, min_wall)


def _soft_score(pts, beta=200.0):
    """Smooth surrogate using log-sum-exp for soft-min.
    Returns (score, gradient) where score approximates the min radius
    and gradient is the subgradient of the soft-min."""
    n_pairs = N * (N - 1) // 2
    n_walls = N * 3
    
    # Pairwise distances (vectorized)
    d = pdist(pts)
    diff = pts[:N-1] - pts[1:]  # not directly usable for all pairs, use pdist mapping
    
    # Build pair index arrays (vectorized)
    pair_idx_a = np.repeat(np.arange(N), np.arange(N-1, 0, -1))
    pair_idx_b = np.concatenate([np.arange(a+1, N) for a in range(N-1)])
    
    # Compute pairwise diffs and distances vectorized
    diffs = pts[pair_idx_a] - pts[pair_idx_b]  # (n_pairs, 3)
    dists = np.linalg.norm(diffs, axis=1)  # (n_pairs,)
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    
    # Wall distances: 0.5 - |pts|
    wall = 0.5 - np.abs(pts)  # shape (N, 3)
    
    # Collect all constraint values
    vals = np.empty(n_pairs + n_walls)
    vals[:n_pairs] = d * 0.5
    vals[n_pairs:] = wall.ravel()
    
    # Soft-min via log-sum-exp
    neg_beta_x = -beta * vals
    m = neg_beta_x.max()
    logsumexp_val = m + np.log(np.exp(neg_beta_x - m).sum())
    soft_min_val = -logsumexp_val / beta
    
    # Weights for each constraint
    exp_vals = np.exp(neg_beta_x - m)
    weights = exp_vals / exp_vals.sum()
    
    # Vectorized gradient computation
    grad = np.zeros_like(pts)
    
    # Pair contributions: grad[a] += w * diff/dist, grad[b] -= w * diff/dist
    pair_w = weights[:n_pairs] / safe_dists  # (n_pairs,)
    # Accumulate into grad using np.add.at for repeated indices
    contrib = pair_w[:, np.newaxis] * diffs  # (n_pairs, 3)
    np.add.at(grad, pair_idx_a, contrib)
    np.add.at(grad, pair_idx_b, -contrib)
    
    # Wall contributions: grad[j,k] += w * (-sign(pts[j,k]))
    wall_w = weights[n_pairs:]  # (N*3,)
    wall_sign = np.sign(pts).ravel()  # (N*3,)
    wall_grad_flat = wall_w * (-wall_sign)
    # Scatter back to grad
    j_idx = np.repeat(np.arange(N), 3)
    k_idx = np.tile(np.arange(3), N)
    np.add.at(grad, (j_idx, k_idx), wall_grad_flat)
    
    return soft_min_val, grad


def _anneal(pts, T0, T1, steps, rng):
    """Simulated annealing with smaller kicks."""
    cur = pts.copy()
    cur_r = _score(cur)
    best = cur.copy()
    best_r = cur_r
    for i in range(steps):
        T = T0 * (T1 / T0) ** (i / steps)
        kick = 0.01 * (T / T0 + 0.05)
        idx = rng.choice(N, size=max(1, N // 5), replace=False)
        cand = cur.copy()
        cand[idx] += rng.normal(0.0, kick, size=(len(idx), 3))
        cand_r = _score(cand)
        if cand_r > cur_r or rng.random() < np.exp((cand_r - cur_r) / max(T, 1e-9)):
            cur = cand
            cur_r = cand_r
            if cur_r > best_r:
                best_r = cur_r
                best = cur.copy()
        if i % 500 == 0 and time.time() > _DEADLINE:
            break
    return best, best_r


def _gradient_ascent(pts, iters, step0, step1, rng, beta=200.0):
    """Smooth gradient ascent using soft-min surrogate."""
    cur = pts.copy()
    cur_r = _score(cur)
    best = cur.copy()
    best_r = cur_r
    for i in range(iters):
        step = step0 * (step1 / step0) ** (i / iters)
        soft_val, g = _soft_score(cur, beta=beta)
        
        # Add small noise to escape flat regions
        g = g + rng.normal(0.0, 0.002, size=cur.shape)
        
        # Normalize gradient to avoid overshooting
        gn = np.linalg.norm(g)
        if gn > 1e-12:
            g = g / gn
        
        cand = cur + step * g
        cand_r = _score(cand)
        if cand_r >= cur_r - 1e-9:
            cur = cand
            cur_r = cand_r
            if cur_r > best_r:
                best_r = cur_r
                best = cur.copy()
        if i % 100 == 0 and time.time() > _DEADLINE:
            break
    return best, best_r


def _polish(pts, max_iters=5000, beta_start=500.0, beta_end=5000.0, rng=None):
    """High-precision polish pass with increasing beta (sharper soft-min)."""
    if rng is None:
        rng = np.random.default_rng(42)
    cur = pts.copy()
    cur_r = _score(cur)
    best = cur.copy()
    best_r = cur_r
    
    for i in range(max_iters):
        # Increase beta over time for sharper approximation
        frac = i / max_iters
        beta = beta_start * (beta_end / beta_start) ** frac
        step = 0.001 * (1.0 - 0.9 * frac)  # Decrease step size
        
        soft_val, g = _soft_score(cur, beta=beta)
        
        # Very small noise
        g = g + rng.normal(0.0, 0.0005, size=cur.shape)
        
        gn = np.linalg.norm(g)
        if gn > 1e-12:
            g = g / gn
        
        cand = cur + step * g
        cand_r = _score(cand)
        if cand_r >= cur_r - 1e-10:
            cur = cand
            cur_r = cand_r
            if cur_r > best_r:
                best_r = cur_r
                best = cur.copy()
        
        if i % 200 == 0 and time.time() > _DEADLINE:
            break
    
    return best, best_r


def construct():
    global _T0
    rng = np.random.default_rng(12345)

    # Start from a 4x4x4 grid (the seed's arrangement)
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    base = grid[:N].copy()
    base_r = _score(base)
    best = base.copy()
    best_r = base_r

    starts = [base]
    
    # 5x5x2 slab arrangement
    m2 = 5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    slab = np.array([(x, y, z) for z in ticks2[:2] for y in ticks2 for x in ticks2])
    if len(slab) >= N:
        starts.append(slab[:N].copy())
    
    # A few random valid starts (jittered grid) to escape the local optimum.
    for _ in range(3):
        s = base + rng.normal(0.0, 0.02, size=base.shape)
        s = np.clip(s, -0.5 + 1e-6, 0.5 - 1e-6)
        starts.append(s)

    for s in starts:
        if time.time() > _DEADLINE:
            break
        a, ar = _anneal(s, T0=0.03, T1=0.0005, steps=3000, rng=rng)
        if ar > best_r:
            best_r, best = ar, a
        if time.time() > _DEADLINE:
            break
        g, gr = _gradient_ascent(a, iters=1500, step0=0.008, step1=0.0003, rng=rng, beta=200.0)
        if gr > best_r:
            best_r, best = gr, g

    # Final high-precision polish pass
    if time.time() < _DEADLINE - 15:
        p, pr = _polish(best, max_iters=2000, beta_start=500.0, beta_end=5000.0, rng=rng)
        if pr > best_r:
            best_r, best = pr, p
    
    # Critical constraint refinement pass
    if time.time() < _DEADLINE - 5:
        c, cr = _critical_refine(best, max_iters=2000, rng=rng)
        if cr > best_r:
            best_r, best = cr, c
    
    # One more polish after critical refinement
    if time.time() < _DEADLINE - 2:
        p, pr = _polish(best, max_iters=1000, beta_start=1000.0, beta_end=10000.0, rng=rng)
        if pr > best_r:
            best_r, best = pr, p
    
    # Final safety: ensure all centres are strictly inside the cube
    eps = 1e-12
    best = np.clip(best, -0.5 + eps, 0.5 - eps)
    
    return best
# EVOLVE-BLOCK-END
