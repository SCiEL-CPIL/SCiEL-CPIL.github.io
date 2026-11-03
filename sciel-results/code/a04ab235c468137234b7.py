import numpy as np
import time
import math
from scipy.spatial.distance import cdist
from scipy.optimize import linprog

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def compute_radius(centers):
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        # Check if centers are inside the square
        if np.any(np.abs(centers) > 0.5 + 1e-9):
            return 0.0
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
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
            y = -0.5 + d/2 + row * row_spacing
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = -0.5 + d/2
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

    def newton_step(centers, r, active_pairs, active_walls, max_iter=50, tol=1e-8):
        """
        Solves for a new configuration where all active constraints are exactly tight at radius r.
        This is a system of linear equations (Newton step for the non-linear constraints approximated locally).
        Actually, for fixed r, we want to find dC such that:
        |C_i + dC_i - C_j - dC_j|^2 = (2r)^2 for active pairs
        C_i + dC_i = boundary for active walls
        
        Linearizing around C:
        2 (C_i - C_j) . (dC_i - dC_j) = (2r)^2 - |C_i - C_j|^2
        
        This is a linear system A dC = b.
        """
        n = len(centers)
        # Number of variables is 2N
        # Number of equations is len(active_pairs) + len(active_walls)
        
        if len(active_pairs) == 0 and len(active_walls) == 0:
            return centers
            
        # If we have more constraints than variables, we need to solve a least squares problem
        # A is (M, 2N)
        
        M = len(active_pairs) + len(active_walls)
        A = np.zeros((M, 2 * n))
        b = np.zeros(M)
        
        row = 0
        # Pair constraints
        for i, j in active_pairs:
            diff = centers[i] - centers[j]
            dist_sq = np.dot(diff, diff)
            target_sq = (2 * r) ** 2
            # Equation: 2 * diff . (dC_i - dC_j) = target_sq - dist_sq
            # dC_i is at index 2i, 2i+1
            # dC_j is at index 2j, 2j+1
            A[row, 2*i] = 2 * diff[0]
            A[row, 2*i+1] = 2 * diff[1]
            A[row, 2*j] = -2 * diff[0]
            A[row, 2*j+1] = -2 * diff[1]
            b[row] = target_sq - dist_sq
            row += 1
            
        # Wall constraints
        # Wall constraints are linear: C_i,k + dC_i,k = +/- 0.5
        # So dC_i,k = target - C_i,k
        # This can be represented as a row in A with 1 at the appropriate spot.
        for i, axis, sign in active_walls:
            # C_i, axis + dC_i, axis = sign * 0.5
            # dC_i, axis = sign * 0.5 - C_i, axis
            idx = 2 * i + axis
            A[row, idx] = 1.0
            b[row] = sign * 0.5 - centers[i, axis]
            row += 1
            
        # Solve A dC = b
        # If M > 2N, use lstsq. If M <= 2N, use lstsq as well for robustness (underdetermined -> min norm).
        try:
            dC, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return centers
            
        dC = dC.reshape(n, 2)
        
        # Check if this step improves the minimum distance
        new_centers = centers + dC
        
        # Clamp to square (hard constraint)
        new_centers[:, 0] = np.clip(new_centers[:, 0], -0.5, 0.5)
        new_centers[:, 1] = np.clip(new_centers[:, 1], -0.5, 0.5)
        
        return new_centers

    def inflate(centers, time_limit=10.0):
        """
        Main inflation loop.
        """
        centers = centers.copy()
        start = time.time()
        
        # Initial radius
        r = compute_radius(centers)
        if r <= 0:
            return centers
            
        # To ensure valid start for inflation, we might need to shrink slightly if on boundary
        # But compute_radius gives the max valid r.
        # We want to increase r.
        
        # We will try to increase r by a small amount delta_r.
        # Then solve for centers that satisfy the new r.
        
        current_r = r
        stall_count = 0
        
        while time.time() < start + time_limit:
            # Identify active constraints for current_r
            dist_matrix = cdist(centers, centers)
            np.fill_diagonal(dist_matrix, np.inf)
            min_dist = np.min(dist_matrix)
            
            wall_dists = 0.5 - np.abs(centers)
            min_wall = np.min(wall_dists)
            
            # Current r is min(min_wall, min_dist/2)
            # If min_dist/2 < min_wall, pairs are active.
            # If min_wall < min_dist/2, walls are active.
            
            # Find all pairs with distance <= 2*current_r + epsilon
            eps = 1e-5
            active_pairs = []
            for i in range(N):
                for j in range(i+1, N):
                    if dist_matrix[i, j] <= 2 * current_r + eps:
                        active_pairs.append((i, j))
                        
            active_walls = []
            for i in range(N):
                for axis in range(2):
                    # Check if close to wall
                    # Wall dist is 0.5 - |c|
                    # If 0.5 - |c_i, axis| <= eps, it's active
                    if wall_dists[i, axis] <= eps:
                        # Determine sign
                        sign = 1 if centers[i, axis] > 0 else -1
                        active_walls.append((i, axis, sign))
                        
            if len(active_pairs) == 0 and len(active_walls) == 0:
                # No constraints? Should not happen if r is correct.
                # Just try to expand slightly
                current_r *= 1.01
                continue

            # Try to increase r
            # Heuristic: try a small step
            delta_r = 1e-4 
            # Adaptive step: if we stalled, reduce step?
            if stall_count > 0:
                delta_r *= 0.5
            if time.time() > start + time_limit - 2.0:
                break
                
            new_r = current_r + delta_r
            
            # Solve for new centers
            new_centers = newton_step(centers, new_r, active_pairs, active_walls)
            
            # Verify new centers
            new_r_actual = compute_radius(new_centers)
            
            if new_r_actual > current_r + 1e-7:
                # Successful expansion
                centers = new_centers
                current_r = new_r_actual
                stall_count = 0
            else:
                # Failed expansion
                stall_count += 1
                if stall_count > 20:
                    # Try a different strategy: random perturbation
                    # Or break
                    break
                # Reduce step size for next try
                # (already handled by stall_count logic above)
                
            # Check time
            if time.time() > start + time_limit - 1.0:
                break
                
        return centers

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
    for _ in range(5):
        if time.time() > deadline - 60:
            break
        rand_pts = np.random.uniform(-0.49, 0.49, (N, 2))
        inits.append(rand_pts)
        
    # 2. For each init, run inflation
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        r_init = compute_radius(init)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = init.copy()
            
        # Run inflation
        # Allocate time based on remaining time
        remaining = deadline - time.time()
        t_limit = min(remaining - 5.0, 10.0)
        if t_limit > 1.0:
            inflated = inflate(init, time_limit=t_limit)
            r_inf = compute_radius(inflated)
            if r_inf > best_radius:
                best_radius = r_inf
                best_centers = inflated.copy()
                
    # Final polish if time remains
    if time.time() < deadline - 5 and best_centers is not None:
        # Run inflation on the best result found
        remaining = deadline - time.time()
        t_limit = min(remaining - 2.0, 5.0)
        if t_limit > 1.0:
            final_polished = inflate(best_centers, time_limit=t_limit)
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
