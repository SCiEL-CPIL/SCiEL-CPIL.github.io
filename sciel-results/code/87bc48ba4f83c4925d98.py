"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def compute_radius(centers):
        """Compute the maximum radius for given centers."""
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        n = len(centers)
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def energy_and_grad(x, n=N, beta=10.0, wall_beta=20.0):
        """
        Smooth energy: sum over pairs of 1/(d_ij + eps) + wall penalty.
        Lower energy = better packing.
        """
        centers = x.reshape(n, 2)
        # Wall penalty: sum of (max(0, 0.5 - |c_i| - margin))^2 style, but smooth
        # Use softplus-like: for each coordinate, penalty if too close to wall
        # d_i = min over coords of (0.5 - |c_ik|)
        # We'll use: for each point, dist_to_wall = min(0.5-|x|, 0.5-|y|)
        # penalty = sum( 1/(dist_to_wall + eps) ) if dist_to_wall < threshold
        # Actually simpler: use (0.5 - |x|) and (0.5 - |y|) directly with inverse
        
        # Pairwise distances
        # Compute all pairwise distances efficiently
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Inverse distance sum (smooth repulsion)
        eps = 1e-6
        pair_energy = np.sum(1.0 / (dists + eps))
        
        # Wall penalty: for each point, distance to nearest wall
        wx = 0.5 - np.abs(centers[:, 0])
        wy = 0.5 - np.abs(centers[:, 1])
        wall_dist = np.minimum(wx, wy)
        # Penalize points that are too close to wall
        wall_eps = 1e-4
        wall_energy = np.sum(1.0 / (wall_dist + wall_eps)) * wall_beta
        
        total_energy = pair_energy + wall_energy
        
        # Gradient
        grad = np.zeros_like(x)
        # Pairwise gradient: d/dx_i sum_j 1/(d_ij+eps) = -sum_j (x_i - x_j)/(d_ij+eps)^2 * (1/d_ij) * ... 
        # Actually d/dx_i [1/(d_ij + eps)] = -(x_i - x_j) / (d_ij + eps)^2 / d_ij * ... 
        # Let's use: grad_i = -sum_{j != i} (x_i - x_j) / (d_ij + eps)^2 / d_ij
        # But d_ij = |x_i - x_j|, so (x_i - x_j)/d_ij is unit vector
        # So grad_i = -sum_{j != i} unit_ij / (d_ij + eps)^2
        # Where unit_ij = (x_i - x_j) / d_ij
        
        # This is O(n^2) but n=200, so 40000 pairs, manageable
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = dists[i, j]
                if d < 1e-10:
                    continue
                unit = (centers[i] - centers[j]) / d
                grad[2*i:2*i+2] -= unit / (d + eps) ** 2
        
        # Wall gradient
        for i in range(n):
            # d/dx of 1/(0.5 - |x| + eps)
            # If x > 0: d/dx [1/(0.5-x+eps)] = 1/(0.5-x+eps)^2
            # If x < 0: d/dx [1/(0.5+x+eps)] = -1/(0.5+x+eps)^2
            wx_i = 0.5 - abs(centers[i, 0])
            wy_i = 0.5 - abs(centers[i, 1])
            
            # For x coordinate
            if centers[i, 0] >= 0:
                grad[2*i] += wall_beta / (wx_i + wall_eps) ** 2
            else:
                grad[2*i] -= wall_beta / (wx_i + wall_eps) ** 2
            
            # For y coordinate
            if centers[i, 1] >= 0:
                grad[2*i+1] += wall_beta / (wy_i + wall_eps) ** 2
            else:
                grad[2*i+1] -= wall_beta / (wy_i + wall_eps) ** 2
        
        return total_energy, grad

    def hexagonal_initialization(n_rows, n_cols):
        """Create hexagonal packing layout."""
        d_from_width = 1.0 / n_cols
        d_from_height = 1.0 / ((n_rows - 1) * math.sqrt(3) / 2 + 1)
        d = min(d_from_width, d_from_height)
        
        total_width = (n_cols - 1) * d + d
        total_height = (n_rows - 1) * d * math.sqrt(3) / 2 + d
        
        x_offset = -0.5 + total_width / 2 - d / 2
        y_offset = -0.5 + total_height / 2 - d / 2
        
        pts = []
        for row in range(n_rows):
            y = y_offset + row * d * math.sqrt(3) / 2
            if row % 2 == 1:
                n_in_row = n_cols - 1
                x_start = x_offset + d / 2
            else:
                n_in_row = n_cols
                x_start = x_offset
            
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) >= N:
            pts = pts[:N]
        elif len(pts) < N:
            # Pad with random points
            while len(pts) < N:
                pts.append([np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4)])
            pts = np.array(pts)
        
        return pts

    def square_grid_init():
        """Create square grid layout."""
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        if len(grid) >= N:
            grid = grid[:N]
        elif len(grid) < N:
            while len(grid) < N:
                grid.append([np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4)])
            grid = np.array(grid)
        return grid

    def force_directed_optimize(centers, max_iter=300, time_limit=60.0):
        """
        Force-directed optimization using L-BFGS-B on smooth energy.
        """
        x0 = centers.flatten()
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        best_x = x0.copy()
        best_energy = energy_and_grad(x0)[0]
        
        result = minimize(
            energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfun': max_iter * 10, 'gtol': 1e-8}
        )
        
        if result.x is not None and np.all(np.isfinite(result.x)):
            e = result.fun
            if e < best_energy:
                best_energy = e
                best_x = result.x.copy()
        
        return best_x.reshape(N, 2)

    def bottleneck_polish(centers, max_iter=200):
        """
        Final non-smooth bottleneck polishing: move points to increase min radius.
        """
        centers = centers.copy()
        current_r = compute_radius(centers)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Find bottleneck
            wall_dists = 0.5 - np.abs(centers)
            min_wall_idx = np.argmin(wall_dists)
            min_wall = wall_dists.flat[min_wall_idx]
            
            n = len(centers)
            dist_matrix = cdist(centers, centers)
            np.fill_diagonal(dist_matrix, np.inf)
            min_pair_idx = np.argmin(dist_matrix)
            min_pair = dist_matrix.flat[min_pair_idx]
            
            if min_wall < min_pair / 2.0:
                # Wall bottleneck
                i = min_wall_idx // 2
                orig = centers[i].copy()
                x, y = orig
                
                # Determine direction away from nearest wall
                if abs(x) > abs(y):
                    direction = np.array([np.sign(x), 0]) if x != 0 else np.array([1, 0])
                    direction = -direction  # Move away from wall
                else:
                    direction = np.array([0, np.sign(y)]) if y != 0 else np.array([0, 1])
                    direction = -direction
                
                best_pos = orig
                best_r = current_r
                for step in [0.0005, 0.001, 0.002, 0.005]:
                    new_center = orig + direction * step
                    if -0.5 <= new_center[0] <= 0.5 and -0.5 <= new_center[1] <= 0.5:
                        centers[i] = new_center
                        new_r = compute_radius(centers)
                        if new_r > best_r + 1e-12:
                            best_r = new_r
                            best_pos = new_center.copy()
                centers[i] = best_pos
                
            else:
                # Pair bottleneck
                i, j = np.unravel_index(min_pair_idx, (n, n))
                orig_i = centers[i].copy()
                orig_j = centers[j].copy()
                
                direction = orig_i - orig_j
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction = direction / norm
                else:
                    direction = np.array([1, 0])
                
                best_pos_i = orig_i
                best_pos_j = orig_j
                best_r = current_r
                
                for step in [0.0005, 0.001, 0.002, 0.005]:
                    # Move i away from j
                    new_i = orig_i + direction * step
                    if -0.5 <= new_i[0] <= 0.5 and -0.5 <= new_i[1] <= 0.5:
                        centers[i] = new_i
                        new_r = compute_radius(centers)
                        if new_r > best_r + 1e-12:
                            best_r = new_r
                            best_pos_i = new_i.copy()
                    centers[i] = orig_i
                    
                    # Move j away from i
                    new_j = orig_j - direction * step
                    if -0.5 <= new_j[0] <= 0.5 and -0.5 <= new_j[1] <= 0.5:
                        centers[j] = new_j
                        new_r = compute_radius(centers)
                        if new_r > best_r + 1e-12:
                            best_r = new_r
                            best_pos_j = new_j.copy()
                    centers[j] = orig_j
                
                centers[i] = best_pos_i
                centers[j] = best_pos_j
            
            new_r = compute_radius(centers)
            if abs(new_r - current_r) < 1e-14:
                break
            current_r = new_r
        
        return centers

    # Generate initial configurations
    best_centers = None
    best_radius = 0.0

    # Try different hexagonal layouts
    for n_rows in range(12, 20):
        if time.time() > deadline - 80:
            break
        n_cols = math.ceil(N / n_rows)
        pts = hexagonal_initialization(n_rows, n_cols)
        if pts is not None and len(pts) == N:
            r = compute_radius(pts)
            if r > best_radius:
                best_radius = r
                best_centers = pts.copy()

    # Try square grid
    grid = square_grid_init()
    r = compute_radius(grid)
    if r > best_radius:
        best_radius = r
        best_centers = grid.copy()

    # Force-directed optimization on the best initial configuration
    if best_centers is not None and time.time() < deadline - 30:
        optimized = force_directed_optimize(best_centers, max_iter=200, time_limit=50.0)
        r = compute_radius(optimized)
        if r > best_radius:
            best_radius = r
            best_centers = optimized.copy()

    # Bottleneck polish
    if best_centers is not None and time.time() < deadline - 10:
        polished = bottleneck_polish(best_centers, max_iter=150)
        r = compute_radius(polished)
        if r > best_radius:
            best_radius = r
            best_centers = polished.copy()

    if best_centers is None:
        # Fallback: square grid
        best_centers = square_grid_init()

    return best_centers
# EVOLVE-BLOCK-END
