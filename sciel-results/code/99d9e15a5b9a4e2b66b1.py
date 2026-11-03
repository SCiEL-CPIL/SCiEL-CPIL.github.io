import numpy as np
import time
import math
from scipy.spatial import cKDTree

N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.spatial import cKDTree

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def compute_radius(centers):
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        # Use KDTree for efficient nearest neighbor search
        tree = cKDTree(centers)
        # Query for the 2 nearest neighbors (self + 1 other)
        dists, _ = tree.query(centers, k=2)
        min_pair = np.min(dists[:, 1])
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def hexagonal_init(n_rows, n_cols, d_scale=1.0):
        d = d_scale / (n_cols + 0.5)
        row_spacing = d * math.sqrt(3) / 2
        total_height = (n_rows - 1) * row_spacing + d
        if total_height > 1.0 + 1e-6:
            return None
        pts = []
        for row in range(n_rows):
            y = -0.5 + d / 2 + row * row_spacing
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = -0.5 + d / 2
            else:
                n_in_row = n_cols - 1
                x_start = -0.5 + d
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            pts = pts[:N]
        elif len(pts) < N:
            return None
        return pts

    def square_grid_init(m):
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    def get_bottleneck(centers):
        """
        Returns (bottleneck_type, indices, value)
        bottleneck_type: 'wall' or 'pair'
        indices: tuple of indices (i,) for wall, (i,j) for pair
        value: the current radius (min distance)
        """
        wall_dists = 0.5 - np.abs(centers)
        min_wall_idx = np.argmin(wall_dists)
        min_wall = wall_dists[min_wall_idx]
        
        tree = cKDTree(centers)
        dists, indices = tree.query(centers, k=2)
        min_pair_dist = np.min(dists[:, 1])
        min_pair_idx = np.argmin(dists[:, 1])
        i = min_pair_idx
        j = indices[min_pair_idx, 1]
        
        if min_wall < min_pair_dist / 2.0:
            return 'wall', (min_wall_idx,), min_wall
        else:
            return 'pair', (i, j), min_pair_dist / 2.0

    def local_search_bottleneck(centers, time_limit=5.0, step_size=0.002):
        """
        Iteratively resolves the current bottleneck by moving the involved particles.
        Uses a simple heuristic: move away from the constraint.
        """
        centers = centers.copy()
        start = time.time()
        
        while time.time() < start + time_limit:
            b_type, idx, val = get_bottleneck(centers)
            
            if b_type == 'wall':
                i = idx[0]
                axis = min_wall_idx % 2 # This is wrong, need to recover axis
                # Re-calculate which wall is closest
                # wall_dists is shape (N, 2). min_wall_idx is flat index.
                i = min_wall_idx // 2
                axis = min_wall_idx % 2
                
                # Move inwards
                if centers[i, axis] > 0:
                    centers[i, axis] -= step_size
                else:
                    centers[i, axis] += step_size
            else:
                i, j = idx
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random kick
                    centers[i] += np.random.randn(2) * step_size
                    centers[j] -= np.random.randn(2) * step_size
                else:
                    direction = diff / dist
                    # Move i away from j, j away from i
                    # But be careful not to push into other particles or walls
                    # Simple step
                    centers[i] += direction * step_size
                    centers[j] -= direction * step_size
            
            # Clamp to bounds
            centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
            centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
            
            # Occasionally randomize step size or direction to escape local minima
            if np.random.rand() < 0.05:
                step_size = np.random.uniform(0.001, 0.005)

        return centers

    def simulated_annealing(centers, time_limit=10.0):
        """
        Simulated annealing on the exact objective (maximize radius).
        Moves: random displacement of one or two particles.
        Acceptance: Metropolis criterion based on radius change.
        """
        centers = centers.copy()
        start = time.time()
        
        current_r = compute_radius(centers)
        best_r = current_r
        best_centers = centers.copy()
        
        T = 0.001 # Initial temperature (scale of radius change)
        T_min = 1e-6
        alpha = 0.999 # Cooling rate
        
        while time.time() < start + time_limit and T > T_min:
            # Generate a move
            # Move 1 or 2 random particles by a random small amount
            n_move = np.random.randint(1, 3)
            idxs = np.random.choice(N, n_move, replace=False)
            delta = np.random.randn(n_move, 2) * 0.01
            
            new_centers = centers.copy()
            new_centers[idxs] += delta
            new_centers[:, 0] = np.clip(new_centers[:, 0], -0.5, 0.5)
            new_centers[:, 1] = np.clip(new_centers[:, 1], -0.5, 0.5)
            
            new_r = compute_radius(new_centers)
            delta_r = new_r - current_r
            
            if delta_r > 0 or np.random.rand() < np.exp(delta_r / T):
                centers = new_centers
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            
            T *= alpha

        return best_centers

    def global_optimization(centers, time_limit=10.0):
        """
        A combination of local search and simulated annealing, with restarts.
        """
        best_centers = centers.copy()
        best_r = compute_radius(centers)
        start = time.time()
        
        # Phase 1: Local search to improve initial config
        ls_centers = local_search_bottleneck(centers, time_limit=min(5.0, (deadline - time.time()) / 2))
        r_ls = compute_radius(ls_centers)
        if r_ls > best_r:
            best_r = r_ls
            best_centers = ls_centers.copy()
        
        # Phase 2: Simulated Annealing from best config
        if time.time() < start + time_limit - 2:
            sa_centers = simulated_annealing(best_centers, time_limit=min(5.0, (deadline - time.time()) - 2))
            r_sa = compute_radius(sa_centers)
            if r_sa > best_r:
                best_r = r_sa
                best_centers = sa_centers.copy()
        
        return best_centers

    # Main logic
    best_centers = None
    best_radius = 0.0
    
    # 1. Generate initial configurations
    inits = []
    
    # Hexagonal
    for n_rows in range(13, 17):
        if time.time() > deadline - 60:
            break
        n_cols = math.ceil(N / n_rows)
        for d_scale in [0.95, 1.0, 1.05]:
            pts = hexagonal_init(n_rows, n_cols, d_scale)
            if pts is not None:
                inits.append(pts)
    
    # Square Grid
    m = int(np.ceil(np.sqrt(N)))
    grid = square_grid_init(m)
    inits.append(grid)
    
    # Random
    for _ in range(3):
        if time.time() > deadline - 60:
            break
        rand_pts = np.random.uniform(-0.49, 0.49, (N, 2))
        inits.append(rand_pts)
    
    # 2. For each init, run the new strategy: Global Optimization (LS + SA)
    for init in inits:
        if time.time() > deadline - 20:
            break
            
        # Quick eval
        r_init = compute_radius(init)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = init.copy()
            
        # Run global optimization
        if time.time() < deadline - 5:
            opt_centers = global_optimization(init, time_limit=10.0)
            r_opt = compute_radius(opt_centers)
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = opt_centers.copy()
                
    # Final global polish if time remains
    if time.time() < deadline - 2 and best_centers is not None:
        final_polished = local_search_bottleneck(best_centers, time_limit=2.0, step_size=0.0005)
        r_final = compute_radius(final_polished)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = final_polished.copy()
            
    if best_centers is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END
