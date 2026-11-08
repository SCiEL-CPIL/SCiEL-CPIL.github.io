"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_sum_r = 0.0
    best_circles = None
    
    # Vectorized objective function
    def objective(circles):
        """
        circles: (N, 3) array of (x, y, r)
        Returns: -sum(r) + penalty (minimize this)
        """
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Wall constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
        # Violations: max(0, r - x), etc.
        v1 = np.maximum(0, r - x)
        v2 = np.maximum(0, r - (1 - x))
        v3 = np.maximum(0, r - y)
        v4 = np.maximum(0, r - (1 - y))
        
        wall_penalty = np.sum(v1**2) + np.sum(v2**2) + np.sum(v3**2) + np.sum(v4**2)
        
        # Overlap constraints: dist(i,j) >= r_i + r_j
        # dist^2 = (x_i - x_j)^2 + (y_i - y_j)^2
        # We need dist >= r_i + r_j => dist^2 >= (r_i + r_j)^2
        # Violation: max(0, (r_i + r_j)^2 - dist^2)
        
        # Compute pairwise differences
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        r_sum_sq = r_sum**2
        
        # Only consider i < j to avoid double counting and diagonal
        # Create a mask for i < j
        i_idx, j_idx = np.triu_indices(N, k=1)
        dist_sq_pairs = dist_sq[i_idx, j_idx]
        r_sum_sq_pairs = r_sum_sq[i_idx, j_idx]
        
        overlap_violations = np.maximum(0, r_sum_sq_pairs - dist_sq_pairs)
        overlap_penalty = np.sum(overlap_violations**2)
        
        penalty = 100.0 * (wall_penalty + overlap_penalty)
        
        return -np.sum(r) + penalty
    
    # Vectorized gradient (numerical or analytical)
    # For now, we'll use a simple coordinate descent with local search
    
    def local_search(circles, max_iter=5000, lr=0.01):
        """
        Coordinate descent with gradient approximation.
        """
        x = circles[:, 0].copy()
        y = circles[:, 1].copy()
        r = circles[:, 2].copy()
        
        current_obj = objective(np.column_stack([x, y, r]))
        best_obj = current_obj
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        rng = np.random.RandomState(42)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random circle to update
            i = rng.randint(N)
            
            # Perturb position and radius
            dx = rng.randn() * lr
            dy = rng.randn() * lr
            dr = rng.randn() * lr * 0.5
            
            new_x = np.clip(x[i] + dx, 0, 1)
            new_y = np.clip(y[i] + dy, 0, 1)
            new_r = np.clip(r[i] + dr, 0.001, 0.5)
            
            # Create new circles array
            new_circles = np.column_stack([x, y, r]).copy()
            new_circles[i] = [new_x, new_y, new_r]
            
            new_obj = objective(new_circles)
            
            # Accept if better or with probability
            if new_obj < current_obj:
                x[i] = new_x
                y[i] = new_y
                r[i] = new_r
                current_obj = new_obj
                
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
            else:
                # Metropolis acceptance
                delta = new_obj - current_obj
                t = 0.01 * (0.001 / 0.01) ** (it / max_iter)
                if delta < 0 or rng.rand() < np.exp(-delta / t):
                    x[i] = new_x
                    y[i] = new_y
                    r[i] = new_r
                    current_obj = new_obj
            
            # Decay learning rate
            lr = 0.01 * (1.0 - it / max_iter) + 0.001
        
        return np.column_stack([best_x, best_y, best_r]), best_obj
    
    # Multiple initializations
    initializations = []
    
    # 1. Grid-based initialization
    side = 5
    for jitter_seed in range(3):
        rng_init = np.random.RandomState(jitter_seed)
        x = (np.arange(N) % side + 0.5 + rng_init.randn(N) * 0.1) / side
        y = (np.arange(N) // side + 0.5 + rng_init.randn(N) * 0.1) / side
        r = np.full(N, 0.5 / side * 0.8)
        x = np.clip(x, 0.02, 0.98)
        y = np.clip(y, 0.02, 0.98)
        initializations.append(np.column_stack([x, y, r]))
    
    # 2. Random initialization
    for rand_seed in range(5):
        rng_init = np.random.RandomState(100 + rand_seed)
        x = rng_init.rand(N) * 0.8 + 0.1
        y = rng_init.rand(N) * 0.8 + 0.1
        r = rng_init.rand(N) * 0.05 + 0.03
        initializations.append(np.column_stack([x, y, r]))
    
    # 3. Spiral initialization
    for spiral_seed in range(2):
        angles = 2 * np.pi * np.arange(N) / N * (1.618 + 0.1 * spiral_seed)
        x = 0.5 + 0.4 * np.cos(angles)
        y = 0.5 + 0.4 * np.sin(angles)
        r = np.full(N, 0.08)
        x = np.clip(x, 0.02, 0.98)
        y = np.clip(y, 0.02, 0.98)
        initializations.append(np.column_stack([x, y, r]))
    
    # Run local search on each initialization
    for init_circles in initializations:
        if time.time() > deadline:
            break
        
        result_circles, result_obj = local_search(init_circles, max_iter=10000)
        
        # Extract sum of radii
        sum_r = np.sum(result_circles[:, 2])
        
        if sum_r > best_sum_r:
            best_sum_r = sum_r
            best_circles = result_circles
    
    # Final validation and cleanup
    if best_circles is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count < N:
                    cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01])
                    count += 1
        best_circles = np.array(cells)
    
    # Ensure validity: fix any remaining violations
    x, y, r = best_circles[:, 0].copy(), best_circles[:, 1].copy(), best_circles[:, 2].copy()
    
    # Fix wall violations
    for i in range(N):
        r[i] = min(r[i], x[i], 1 - x[i], y[i], 1 - y[i])
        r[i] = max(r[i], 1e-6)
    
    # Fix overlaps by shrinking radii
    for iteration in range(100):
        fixed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    shrink = overlap / 2 + 0.001
                    r[i] = max(1e-6, r[i] - shrink)
                    r[j] = max(1e-6, r[j] - shrink)
                    fixed = True
        if not fixed:
            break
    
    best_circles = np.column_stack([x, y, r])
    
    return best_circles
# EVOLVE-BLOCK-END
