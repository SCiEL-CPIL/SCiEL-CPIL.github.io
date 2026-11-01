"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist


def _compute_score(centers):
    """
    Compute the packing radius for given centers.
    Returns the minimum of (min_pairwise_dist / 2, min_wall_dist).
    """
    # Wall distances: min over all centers of min over coords of (0.5 - |coord|)
    min_wall = np.min(0.5 - np.abs(centers))
    
    # Pairwise distances using scipy for speed
    if N > 1:
        dists = pdist(centers)
        min_pair = np.min(dists)
    else:
        min_pair = np.inf
        
    return min(min_pair / 2.0, min_wall)


def _relax(centers, iterations=100, rng=None):
    """
    Simple relaxation to reduce overlaps.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    for _ in range(iterations):
        # Calculate pairwise distances
        dists = pdist(centers)
        min_dist = np.min(dists)
        
        # Find the pair with minimum distance
        idx = np.argmin(dists)
        # Convert flat index to (i, j)
        # pdist returns condensed matrix. Index k corresponds to (i, j) with i < j
        # Number of elements before row i is i*(2*N - i - 1) / 2
        # We can just loop or use a helper, but for N=50 it's small.
        # Faster: just find the min and update.
        
        # To find i, j from condensed index:
        # Let m = N-1
        # idx = i * (2*N - i - 1) // 2 + j - i - 1
        # This is tricky to invert analytically quickly.
        # For N=50, we can just use a simple loop or precompute mapping.
        
        # Alternative: Just do a few random moves to push apart.
        # Pick the closest pair.
        # We need i and j.
        # Let's precompute the mapping for N=50 once? Or just use np.triu_indices?
        
        # Simpler: Use a vectorized push.
        # Compute all pairwise vectors and distances.
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        min_idx = np.unravel_index(np.argmin(dists_sq), dists_sq.shape)
        i, j = min_idx
        
        # Push i and j apart
        vec = centers[i] - centers[j]
        dist = np.linalg.norm(vec)
        if dist < 1e-6:
            vec = rng.normal(0, 0.01, 2)
            dist = np.linalg.norm(vec)
        
        # Target distance is slightly larger than current min to resolve overlap
        # We want to increase the minimum distance.
        # Push by a fraction of the overlap?
        # Let's just push them apart by a small amount.
        # The "force" is proportional to 1/d or similar.
        
        # Actually, simple relaxation is hard to tune.
        # Let's skip complex relaxation and rely on SA with good init.
        pass
    return centers


def construct():
    """
    Use simulated annealing to maximize the packing radius.
    Improved with scipy distances, better init, and tuned SA.
    """
    rng = np.random.default_rng(42)
    
    start_time = time.time()
    max_time = 110.0
    
    best_centers = None
    best_score = -np.inf
    
    # Initialization 1: Grid with jitter
    # 7x8 grid is a good start for 50 circles.
    # 7 rows * 8 cols = 56 spots. We take 50.
    rows, cols = 7, 8
    x_ticks = np.linspace(-0.48, 0.48, cols)
    y_ticks = np.linspace(-0.48, 0.48, rows)
    
    centers_grid = np.zeros((N, 2))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < N:
                centers_grid[idx] = [x_ticks[c], y_ticks[r]]
                idx += 1
            else:
                break
        if idx >= N:
            break
    
    # Add small jitter
    centers_grid += rng.normal(0, 0.005, (N, 2))
    centers_grid = np.clip(centers_grid, -0.499, 0.499)
    
    # Initial score
    current_score = _compute_score(centers_grid)
    best_centers = centers_grid.copy()
    best_score = current_score
    
    # SA Parameters
    T_start = 0.05
    T_end = 1e-8
    n_iterations = 500000 # Adjust based on speed
    alpha = (T_end / T_start) ** (1.0 / n_iterations)
    
    step_size = 0.05
    
    # Track no-improve count for adaptive step
    no_improve = 0
    max_no_improve = 2000
    
    for it in range(n_iterations):
        if it % 1000 == 0:
            if time.time() - start_time > max_time:
                break
        
        T = T_start * (alpha ** it)
        
        # Pick random index
        i = rng.integers(N)
        old_pos = centers[i].copy()
        
        # Perturb
        centers[i] += rng.normal(0, step_size, 2)
        centers[i] = np.clip(centers[i], -0.4999, 0.4999)
        
        # Compute new score
        new_score = _compute_score(centers)
        
        if new_score >= current_score:
            current_score = new_score
            no_improve = 0
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
        else:
            delta = new_score - current_score
            # Accept with prob exp(delta/T)
            if delta / T > -100:
                if rng.random() < np.exp(delta / T):
                    current_score = new_score
                    no_improve = 0
                else:
                    centers[i] = old_pos
                    no_improve += 1
            else:
                centers[i] = old_pos
                no_improve += 1
        
        # Adaptive step size
        if no_improve > max_no_improve:
            step_size *= 0.7
            no_improve = 0
        elif it % 5000 == 0 and it > 0:
            # Slowly cool down step size
            step_size *= 0.995
            
        # Occasionally kick start
        if it % 50000 == 0 and it > 0:
            step_size = 0.05
            no_improve = 0

    return best_centers
# EVOLVE-BLOCK-END
