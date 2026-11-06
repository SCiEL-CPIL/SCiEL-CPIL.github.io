"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(centers):
    """Compute the radius for a given set of centers."""
    # Distance to boundary
    dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary = np.min(dist_to_boundary)
    
    # Minimum distance between pairs
    n = len(centers)
    min_pair_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair_dist:
                min_pair_dist = d
    
    return min(min_pair_dist / 2.0, min_boundary)

def _score_fast(centers):
    """Fast score computation using vectorized operations."""
    dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary = np.min(dist_to_boundary)
    
    # Vectorized pair distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)

def _optimize_from_start(centers, start_time, deadline, max_iter=3000):
    """Optimize from a starting configuration using gradient-free local search."""
    best_centers = centers.copy()
    best_score = _score_fast(centers)
    
    # Adaptive step size
    step_size = 0.02
    patience = 0
    max_patience = 50
    
    for it in range(max_iter):
        if time.time() - start_time > deadline:
            break
        
        # Generate perturbation
        if patience > max_patience:
            # Reset to best and increase step size
            centers = best_centers.copy()
            step_size *= 1.5
            patience = 0
        
        # Random perturbation
        noise = np.random.randn(N, 2) * step_size
        new_centers = centers + noise
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_score = _score_fast(new_centers)
        
        if new_score > best_score:
            best_score = new_score
            best_centers = new_centers.copy()
            centers = new_centers.copy()
            patience = 0
            step_size *= 0.95  # Slightly decrease step size on improvement
        else:
            patience += 1
            step_size *= 0.9  # Decrease step size on failure
    
    return best_centers, best_score

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    best_centers = None
    best_score = -np.inf
    
    # Generate multiple initial configurations and optimize each
    
    # Method 1: Hexagonal grid variations
    for n_cols in range(5, 12):
        for n_rows in range(5, 12):
            if n_cols * n_rows < N:
                continue
            
            dx = 2.0 / n_cols
            dy = dx * np.sqrt(3) / 2.0
            
            total_w = (n_cols - 1) * dx + dx/2
            total_h = (n_rows - 1) * dy
            
            offset_x = -total_w / 2.0
            offset_y = -total_h / 2.0
            
            centers = []
            for i in range(n_rows):
                for j in range(n_cols):
                    x = offset_x + j * dx + (i % 2) * dx / 2.0
                    y = offset_y + i * dy
                    centers.append([x, y])
            
            centers = np.array(centers[:N])
            
            if len(centers) < N:
                continue
            
            if np.any(np.abs(centers) > 0.5):
                continue
            
            s = _score_fast(centers)
            if s > best_score:
                best_score = s
                best_centers = centers.copy()
            
            if time.time() - start_time > deadline:
                break
    
    # Method 2: Square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N]
    s = _score_fast(grid)
    if s > best_score:
        best_score = s
        best_centers = grid.copy()
    
    # Method 3: Random initialization with multiple restarts
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() - start_time > deadline:
            break
        
        # Random points
        centers = np.random.uniform(-0.45, 0.45, (N, 2))
        s = _score_fast(centers)
        if s > best_score:
            best_score = s
            best_centers = centers.copy()
    
    if best_centers is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Optimize from the best initial configuration
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 5:
        optimized_centers, opt_score = _optimize_from_start(
            best_centers, start_time, deadline, max_iter=2000
        )
        if opt_score > best_score:
            best_score = opt_score
            best_centers = optimized_centers
    
    return best_centers
# EVOLVE-BLOCK-END
