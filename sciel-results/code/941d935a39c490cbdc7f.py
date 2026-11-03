"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial.distance import pdist

_T0 = time.time()
_DEADLINE = 115.0  # safety margin below the 120 s wall-clock limit


def _score(pts):
    """Radius for a centre array: min(pair_dist/2, wall_dist)."""
    d = pdist(pts)
    min_pair = d.min() if d.size > 0 else np.inf
    min_wall = np.min(0.5 - np.abs(pts))
    return min(min_pair * 0.5, min_wall)


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


def _gradient_ascent(pts, iters, step0, step1, rng):
    """Vectorized gradient ascent on the minimum radius."""
    cur = pts.copy()
    cur_r = _score(cur)
    best = cur.copy()
    best_r = cur_r
    for i in range(iters):
        step = step0 * (step1 / step0) ** (i / iters)
        # Compute all pairwise distances
        d = pdist(cur)
        dist_matrix = np.zeros((N, N))
        idx = np.triu_indices(N, k=1)
        dist_matrix[idx] = d
        dist_matrix = dist_matrix + dist_matrix.T
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Find minimum pair distance
        min_pair_d = d.min() if d.size > 0 else np.inf
        # Find minimum wall distance
        wall = 0.5 - np.abs(cur)
        min_wall_d = wall.min()
        
        # Determine which constraint is limiting
        g = np.zeros_like(cur)
        
        # Pair subgradient: push apart the closest pair(s)
        threshold = min_pair_d + 1e-6
        for a in range(N):
            for b in range(a + 1, N):
                if dist_matrix[a, b] < threshold:
                    diff = cur[a] - cur[b]
                    n = np.linalg.norm(diff)
                    if n > 1e-12:
                        g[a] += diff / n
                        g[b] -= diff / n
        
        # Wall subgradient: push inward the closest wall(s)
        wall_threshold = min_wall_d + 1e-6
        for k in range(3):
            for j in range(N):
                if wall[j, k] < wall_threshold:
                    g[j, k] += np.sign(cur[j, k])
        
        # Add small noise to escape flat regions
        g = g + rng.normal(0.0, 0.003, size=cur.shape)
        
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
        g, gr = _gradient_ascent(a, iters=1500, step0=0.008, step1=0.0003, rng=rng)
        if gr > best_r:
            best_r, best = gr, g

    # Final safety: ensure all centres are strictly inside the cube
    eps = 1e-12
    best = np.clip(best, -0.5 + eps, 0.5 - eps)
    
    return best
# EVOLVE-BLOCK-END
