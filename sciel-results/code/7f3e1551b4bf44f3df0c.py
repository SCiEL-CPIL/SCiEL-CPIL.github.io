"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def construct():
    """Construct a good packing of 200 equal circles in the unit square [-1/2, 1/2]^2."""
    
    def compute_radius(centers):
        """Compute the maximum common radius for a given set of centers."""
        n = len(centers)
        if n < 2:
            return 0.5
        # Wall distances
        wall = np.min(np.stack([
            0.5 - centers[:, 0],
            centers[:, 0] + 0.5,
            0.5 - centers[:, 1],
            centers[:, 1] + 0.5
        ]), axis=0)
        wall_r = np.min(wall)
        
        # Pairwise distances (only compute upper triangle)
        dx = centers[:, 0][:, None] - centers[:, 0][None, :]
        dy = centers[:, 1][:, None] - centers[:, 1][None, :]
        dists = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(wall_r, min_pair)
    
    def hexagonal_packing(n_rows, n_cols):
        """Generate hexagonal packing centers."""
        centers = []
        dx = 0.2  # placeholder, will be adjusted
        dy = 0.2 * np.sqrt(3) / 2
        
        for i in range(n_rows):
            for j in range(n_cols):
                x = (j - (n_cols - 1) / 2) * dx
                if i % 2 == 1:
                    x += dx / 2
                y = (i - (n_rows - 1) / 2) * dy
                centers.append([x, y])
        return np.array(centers)
    
    def generate_hex_initial(n):
        """Generate a good hexagonal initial configuration for n circles."""
        # For 200 circles, try different row/column configurations
        best_centers = None
        best_r = 0
        
        for rows in range(10, 25):
            # Distribute n circles into rows
            base_cols = n // rows
            extra = n % rows
            
            centers = []
            for i in range(rows):
                cols = base_cols + (1 if i < extra else 0)
                if cols == 0:
                    continue
                for j in range(cols):
                    x = (j - (cols - 1) / 2) * 0.1
                    if i % 2 == 1:
                        x += 0.05
                    y = (i - (rows - 1) / 2) * 0.15
                    centers.append([x, y])
            
            centers = np.array(centers)
            if len(centers) != n:
                continue
            
            r = compute_radius(centers)
            if r > best_r:
                best_r = r
                best_centers = centers.copy()
        
        return best_centers if best_centers is not None else None
    
    def local_search(centers, max_iter=5000, step_size=0.01, deadline=None):
        """Local search to improve packing."""
        best_centers = centers.copy()
        best_r = compute_radius(best_centers)
        
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            
            # Random perturbation
            idx = np.random.randint(len(best_centers))
            dx = np.random.uniform(-step_size, step_size)
            dy = np.random.uniform(-step_size, step_size)
            
            new_centers = best_centers.copy()
            new_centers[idx, 0] += dx
            new_centers[idx, 1] += dy
            
            # Check bounds
            if (np.any(new_centers[:, 0] < -0.5) or np.any(new_centers[:, 0] > 0.5) or
                np.any(new_centers[:, 1] < -0.5) or np.any(new_centers[:, 1] > 0.5)):
                continue
            
            r = compute_radius(new_centers)
            if r > best_r:
                best_r = r
                best_centers = new_centers.copy()
                step_size = max(0.001, step_size * 0.99)
            else:
                step_size = min(0.05, step_size * 1.01)
        
        return best_centers, best_r
    
    def repulsion_search(centers, max_iter=2000, deadline=None):
        """Repulsion-based optimization."""
        best_centers = centers.copy()
        best_r = compute_radius(best_centers)
        
        step = 0.01
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            
            new_centers = best_centers.copy()
            for i in range(len(new_centers)):
                # Compute forces from other circles
                dx = new_centers[i, 0] - new_centers[:, 0]
                dy = new_centers[i, 1] - new_centers[:, 1]
                dists = np.sqrt(dx**2 + dy**2)
                dists[i] = np.inf
                
                # Repulsion force (inversely proportional to distance)
                force_x = np.sum(dx / (dists + 1e-8)**2)
                force_y = np.sum(dy / (dists + 1e-8)**2)
                
                # Wall repulsion
                wall_force_x = 0
                wall_force_y = 0
                if new_centers[i, 0] < 0:
                    wall_force_x += (0.5 - new_centers[i, 0]) * 10
                elif new_centers[i, 0] > 0:
                    wall_force_x -= (new_centers[i, 0] + 0.5) * 10
                if new_centers[i, 1] < 0:
                    wall_force_y += (0.5 - new_centers[i, 1]) * 10
                elif new_centers[i, 1] > 0:
                    wall_force_y -= (new_centers[i, 1] + 0.5) * 10
                
                # Update position
                new_centers[i, 0] += step * (force_x + wall_force_x) * 0.01
                new_centers[i, 1] += step * (force_y + wall_force_y) * 0.01
                
                # Keep within bounds
                new_centers[i, 0] = np.clip(new_centers[i, 0], -0.49, 0.49)
                new_centers[i, 1] = np.clip(new_centers[i, 1], -0.49, 0.49)
            
            r = compute_radius(new_centers)
            if r > best_r:
                best_r = r
                best_centers = new_centers.copy()
            
            step *= 0.995
        
        return best_centers, best_r
    
    # Main algorithm
    start_time = time.time()
    deadline = start_time + 120  # Leave margin for evaluation
    
    # Generate initial hexagonal configuration
    initial = generate_hex_initial(N)
    if initial is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        initial = grid[:N]
    
    best_centers = initial.copy()
    best_r = compute_radius(best_centers)
    
    # Local search phase
    centers, r = local_search(initial, max_iter=3000, step_size=0.02, deadline=deadline)
    if r > best_r:
        best_r = r
        best_centers = centers.copy()
    
    # Repulsion search phase
    if time.time() < deadline - 10:
        centers, r = repulsion_search(best_centers, max_iter=1000, deadline=deadline)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    # Final local refinement
    if time.time() < deadline - 5:
        centers, r = local_search(best_centers, max_iter=2000, step_size=0.005, deadline=deadline)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    return best_centers
# EVOLVE-BLOCK-END
