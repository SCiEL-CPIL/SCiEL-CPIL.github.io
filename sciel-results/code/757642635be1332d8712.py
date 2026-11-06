"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def _score_fast(centers):
    """Fast score computation using cKDTree for O(N log N) nearest-neighbor queries."""
    # Boundary distances
    dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary = np.min(dist_to_boundary)
    
    # Nearest neighbor distances using cKDTree
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)  # k=2 to get self + nearest neighbor
    min_pair_dist = np.min(dists[:, 1])  # distance to nearest other point
    
    return min(min_pair_dist / 2.0, min_boundary)

def _soft_min_score(centers, beta=20.0):
    """Smooth surrogate using soft-min of all constraint distances."""
    # Boundary distances
    boundary_dists = np.min(0.5 - np.abs(centers), axis=1)
    
    # Pair distances via cKDTree (only need nearest neighbors for the min)
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    pair_dists = dists[:, 1] / 2.0
    
    # Combine all constraints
    all_dists = np.concatenate([pair_dists, boundary_dists])
    
    # Soft min via log-sum-exp
    neg_beta_x = -beta * all_dists
    m = np.max(neg_beta_x)
    log_sum_exp = m + np.log(np.sum(np.exp(neg_beta_x - m)))
    soft_min_val = -log_sum_exp / beta
    
    return soft_min_val

def _polish_coordinate_descent(centers, start_time, deadline, max_sweeps=300):
    """High-precision polish: coordinate descent optimizing the min-score directly."""
    best_centers = centers.copy()
    best_score = _score_fast(centers)
    coord_step = 1e-3
    
    for sweep in range(max_sweeps):
        if time.time() - start_time > deadline:
            break
        
        improved = False
        for i in range(N):
            for dim in range(2):
                best_val = best_centers[i, dim]
                best_s = best_score
                
                # Binary search for optimal coordinate value
                lo = max(-0.5, best_val - coord_step)
                hi = min(0.5, best_val + coord_step)
                
                # Try several candidate values
                candidates = np.linspace(lo, hi, 11)
                for trial_val in candidates:
                    trial = best_centers.copy()
                    trial[i, dim] = trial_val
                    s = _score_fast(trial)
                    if s > best_s + 1e-14:
                        best_s = s
                        best_val = trial_val
                
                if abs(best_val - best_centers[i, dim]) > 1e-12:
                    best_centers[i, dim] = best_val
                    improved = True
        
        if best_score < best_s - 1e-14:
            best_score = best_s
        
        # Shrink step as we converge
        if sweep % 10 == 9:
            coord_step *= 0.75
        
        if not improved and coord_step < 1e-8:
            break
    
    return best_centers, best_score

def _optimize_from_start(centers, start_time, deadline, max_iter=5000):
    """Optimize using smooth surrogate-guided search with adaptive step size."""
    best_centers = centers.copy()
    best_score = _score_fast(centers)
    
    step_size = 0.01
    patience = 0
    max_patience = 100
    best_soft = _soft_min_score(centers, beta=15.0)
    
    for it in range(max_iter):
        if time.time() - start_time > deadline:
            break
        
        if patience > max_patience:
            # Reset to best and increase step size
            centers = best_centers.copy()
            step_size *= 1.4
            patience = 0
        
        # Generate perturbation with occasional large jumps
        if np.random.random() < 0.08:
            noise = np.random.randn(N, 2) * step_size * 2.5
        else:
            noise = np.random.randn(N, 2) * step_size
        
        new_centers = centers + noise
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_soft = _soft_min_score(new_centers, beta=15.0)
        new_score = _score_fast(new_centers)
        
        # Accept if exact score improves, or soft score improves significantly
        if new_score > best_score + 1e-12:
            best_score = new_score
            best_centers = new_centers.copy()
            centers = new_centers.copy()
            best_soft = new_soft
            patience = 0
            step_size *= 0.96
        elif new_soft > best_soft + 0.001 and new_score > best_score - 1e-5:
            # Accept if soft score improves and we're not too far from best
            centers = new_centers.copy()
            best_soft = new_soft
            patience = 0
            step_size *= 0.97
        else:
            patience += 1
            step_size *= 0.92
    
    return best_centers, best_score

def _generate_hex_grid(n_cols, n_rows, offset_x=0.0, offset_y=0.0):
    """Generate a hexagonal grid of points, centered in the square."""
    if n_cols * n_rows < N:
        return None
    
    # For hex packing, optimal spacing
    # Width = (n_cols - 1) * dx + dx/2 (for staggered rows)
    # We want this to fit in [-0.5, 0.5]
    dx = 1.0 / (n_cols - 0.5)  # spacing
    dy = dx * np.sqrt(3) / 2.0
    
    # Total dimensions
    total_w = (n_cols - 1) * dx + dx / 2.0
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
    
    # Check bounds
    if np.any(np.abs(centers) > 0.5):
        return None
    
    return centers

def construct():
    start_time = time.time()
    deadline = 118.0  # Safety margin before 150s kill
    
    best_centers = None
    best_score = -np.inf
    
    # Phase 0: Generate diverse initial configurations
    initial_configs = []
    
    # Method 1: Hexagonal grids with various dimensions
    for n_cols in range(5, 13):
        for n_rows in range(5, 13):
            if n_cols * n_rows < N:
                continue
            centers = _generate_hex_grid(n_cols, n_rows)
            if centers is not None:
                s = _score_fast(centers)
                initial_configs.append((s, centers.copy()))
    
    # Method 2: Square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= N:
        grid = grid[:N]
        s = _score_fast(grid)
        initial_configs.append((s, grid.copy()))
    
    # Method 3: Random initializations
    for r in range(10):
        centers = np.random.uniform(-0.48, 0.48, (N, 2))
        s = _score_fast(centers)
        initial_configs.append((s, centers.copy()))
    
    # Sort by score and keep top configurations
    initial_configs.sort(key=lambda x: x[0], reverse=True)
    top_configs = [c for _, c in initial_configs[:5]]
    
    if not top_configs:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Phase 1: Optimize from each top configuration
    for config in top_configs:
        if time.time() - start_time > deadline - 10:
            break
        
        elapsed = time.time() - start_time
        remaining = deadline - 10 - elapsed
        if remaining < 5:
            break
        
        phase_deadline = start_time + min(remaining, 25.0)
        optimized_centers, opt_score = _optimize_from_start(
            config, start_time, phase_deadline, max_iter=3000
        )
        
        if opt_score > best_score:
            best_score = opt_score
            best_centers = optimized_centers.copy()
    
    # Phase 2: High-precision coordinate descent polish
    if best_centers is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 5:
            polish_deadline = deadline - 3.0
            polished_centers, polish_score = _polish_coordinate_descent(
                best_centers, start_time, polish_deadline, max_sweeps=200
            )
            if polish_score > best_score:
                best_score = polish_score
                best_centers = polished_centers
    
    if best_centers is None:
        # Final fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
