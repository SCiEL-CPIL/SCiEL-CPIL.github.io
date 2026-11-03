import time
import numpy as np
from scipy.spatial import cKDTree

N = 50
TIME_LIMIT = 130.0
RNG_SEED = 123456789


def _score(c):
    c = np.asarray(c, dtype=float)
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    if np.any(c < -0.5) or np.any(c > 0.5):
        return -1.0
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r = float(np.min(dist) * 0.5)
    m = np.min(np.abs(c), axis=1)
    r = min(r, float(0.5 - np.max(m)))
    return r


def _score_and_grad(c):
    """Return (score, gradient) where gradient[i,k] is d(score)/d(c[i,k])."""
    c = np.asarray(c, dtype=float)
    n = c.shape[0]
    # Compute pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-30)
    np.fill_diagonal(dist, np.inf)
    
    # Find minimum pair distance
    min_idx = np.unravel_index(np.argmin(dist), dist.shape)
    i, j = min_idx
    if i == j:
        return 0.0, np.zeros_like(c)
    
    d = dist[i, j]
    # Also check wall distances
    wall_dists = np.empty((n, 3))
    for k in range(3):
        wall_dists[:, k] = 0.5 - np.abs(c[:, k])
    min_wall = np.min(wall_dists)
    
    if d / 2.0 <= min_wall:
        # Spheres are the constraint
        r = d / 2.0
        # Gradient: only i and j contribute
        grad = np.zeros_like(c)
        if d > 1e-15:
            direction = (c[i] - c[j]) / d
            grad[i] += direction * 0.5
            grad[j] -= direction * 0.5
        return r, grad
    else:
        # Walls are the constraint
        r = float(min_wall)
        grad = np.zeros_like(c)
        # Find which wall is closest
        min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        wi, wk = min_wall_idx
        if c[wi, wk] > 0:
            grad[wi, wk] = -1.0
        else:
            grad[wi, wk] = 1.0
        return r, grad


def _local_search(c0, deadline, rng, iters_cap=500000):
    c = np.array(c0, dtype=float)
    best = _score(c)
    step = 0.01
    n = c.shape[0]
    it = 0
    no_improve = 0
    
    while it < iters_cap:
        it += 1
        if time.time() > deadline:
            break
        
        # Get gradient
        sc, grad = _score_and_grad(c)
        if sc > best + 1e-12:
            best = sc
        
        # Normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-15:
            break
        
        # Try moving in gradient direction
        moved = False
        for trial_step in (step, step * 0.5, step * 0.25):
            cand = c + trial_step * grad / gnorm
            # Clip to box
            cand = np.clip(cand, -0.5 + 1e-9, 0.5 - 1e-9)
            sc_cand = _score(cand)
            if sc_cand > best + 1e-12:
                c = cand
                best = sc_cand
                step = min(step * 1.1, 0.05)
                no_improve = 0
                moved = True
                break
        
        if not moved:
            no_improve += 1
            step *= 0.85
            if step < 1e-7:
                break
            if no_improve > 200:
                break
        
        # Occasionally try a random perturbation to escape local minima
        if it % 500 == 0:
            i = int(rng.integers(0, n))
            k = int(rng.integers(0, 3))
            sgn = 1 if rng.random() > 0.5 else -1
            cand = c.copy()
            cand[i, k] += sgn * step * 0.5
            if -0.5 <= cand[i, k] <= 0.5:
                sc_cand = _score(cand)
                if sc_cand > best + 1e-12:
                    c = cand
                    best = sc_cand
                    no_improve = 0
    
    return c, best


def _generate_good_init(rng):
    """Generate a better initial configuration using a structured approach."""
    # Start with a 4x4x4 grid (64 points), pick 50
    ticks = (np.arange(4) + 0.5) / 4.0 - 0.5
    pts = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Randomly select 50 from 64
    idx = rng.permutation(64)[:50]
    c = pts[idx].copy()
    
    # Add small random jitter to break symmetry
    c += (rng.random((N, 3)) - 0.5) * 0.01
    c = np.clip(c, -0.5 + 1e-6, 0.5 - 1e-6)
    
    return c


def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT
    rng = np.random.default_rng(RNG_SEED)
    
    best_c = None
    best_r = -1.0
    
    # Multiple restarts with different initializations
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # Structured initialization
            c0 = _generate_good_init(rng)
        elif restart == 1:
            # Slightly perturbed structured init
            c0 = _generate_good_init(rng)
            c0 += (rng.random((N, 3)) - 0.5) * 0.05
            c0 = np.clip(c0, -0.5 + 1e-6, 0.5 - 1e-6)
        else:
            # Random initialization
            c0 = (rng.random((N, 3)) - 0.5) * 0.9
            c0 = np.clip(c0, -0.5 + 1e-6, 0.5 - 1e-6)
        
        # Local search
        c, r = _local_search(c0, deadline, rng, iters_cap=300000)
        
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    # Final polish with smaller steps
    if best_c is not None and time.time() < deadline:
        c, r = _local_search(best_c, deadline, rng, iters_cap=200000)
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    if best_c is None:
        # Fallback
        ticks = (np.arange(4) + 0.5) / 4.0 - 0.5
        pts = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = pts[:N].copy()
    
    return best_c
