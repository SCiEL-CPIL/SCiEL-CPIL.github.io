"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def calculate_score(points):
        """Calculate the maximum common radius for a given set of points."""
        if len(points) < 2:
            return 1.0
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_dist = np.min(np.sqrt(dist_sq))
        
        r_i = np.linalg.norm(points, axis=1)
        min_bound_dist = np.min(1.0 - r_i)
        
        return min(min_pair_dist / 2.0, min_bound_dist)

    def gen_hex_lattice_superlattice():
        """Generate a dense hexagonal lattice (superlattice) with many points."""
        # Generate points with a fine spacing to have plenty to choose from
        spacing = 0.05  # Fine enough to have many points
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.5:  # Generate a larger region
                    pts.append((x, y))
        return np.array(pts) if pts else np.empty((0, 2))

    def apply_affine_transform(points, theta, scale, shear):
        """Apply rotation, scaling, and shear transformation."""
        # Rotation matrix
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        # Shear matrix
        S = np.array([[1.0, shear],
                      [0.0, 1.0]])
        # Combined transformation
        M = R @ S @ np.diag([scale, scale])
        return points @ M.T

    def select_best_points(transformed_pts, n_select):
        """Select n_select points that fit best inside the unit circle."""
        # Compute distance from origin
        dists = np.linalg.norm(transformed_pts, axis=1)
        
        # Only consider points that can fit (distance <= 1 - estimated_radius)
        # We'll estimate radius later, for now just take closest to center
        # that are within reasonable bounds
        valid_mask = dists < 0.95  # Conservative bound
        if np.sum(valid_mask) < n_select:
            # If not enough, take all valid and add some random
            valid_pts = transformed_pts[valid_mask]
            remaining = n_select - len(valid_pts)
            if remaining > 0:
                # Add points from outside but still reasonable
                all_sorted = np.argsort(dists)
                valid_pts = np.vstack([valid_pts, transformed_pts[all_sorted[:remaining]]])
            return valid_pts[:n_select]
        
        # Sort by distance from center and take the closest
        sorted_idx = np.argsort(dists[valid_mask])
        selected_indices = np.where(valid_mask)[0][sorted_idx[:n_select]]
        return transformed_pts[selected_indices]

    # 1. Initialization with superlattice approach
    superlattice = gen_hex_lattice_superlattice()
    
    # Try several different affine transformations
    best_init = None
    best_init_score = -np.inf
    
    # Deterministic set of transformations to try
    theta_values = [0, np.pi/6, np.pi/12, np.pi/3, np.pi/4]
    scale_values = [0.95, 1.0, 1.05, 1.1]
    shear_values = [0.0, 0.1, -0.1, 0.2, -0.2]
    
    for theta in theta_values:
        for scale in scale_values:
            for shear in shear_values:
                if time.time() > deadline - 100:
                    break
                transformed = apply_affine_transform(superlattice, theta, scale, shear)
                selected = select_best_points(transformed, N)
                
                if len(selected) == N:
                    score = calculate_score(selected)
                    if score > best_init_score:
                        best_init_score = score
                        best_init = selected.copy()
    
    if best_init is None:
        # Fallback to simple hex lattice
        spacing = np.sqrt(2.0 * np.pi / (np.sqrt(3.0) * N))
        points = gen_hex_lattice(spacing)
        while len(points) < N and spacing > 0.01:
            spacing *= 0.95
            points = gen_hex_lattice(spacing)
        if len(points) >= N:
            dists = np.linalg.norm(points, axis=1)
            sorted_idx = np.argsort(dists)
            best_init = points[sorted_idx[:N]]
        else:
            best_init = points
            while len(best_init) < N:
                best_init = np.vstack([best_init, np.random.uniform(-0.9, 0.9, size=(1, 2))])
            best_init = best_init[:N]
            best_init_score = calculate_score(best_init)
    
    init_points = best_init

    best_points = init_points.copy()
    best_score = calculate_score(init_points)

    # 2. Define energy function for L-BFGS-B
    # We use a smooth surrogate for the max-min problem.
    # Maximize r such that |x_i - x_j| >= 2r and |x_i| + r <= 1.
    # Equivalent to minimizing -r subject to constraints.
    # Alternatively, minimize sum of violations.
    # Let's stick to the repulsion energy which drives points apart and to boundary.
    # To improve upon the simple 1/d^2, we can use a softer potential or 
    # optimize the 'temperature' of the soft-min.
    
    # A better approach: use the 'soft-min' of distances.
    # max_r = min( min_{i<j} d_ij/2, min_i (1 - |x_i|) )
    # log-sum-exp trick for min: min(x) = -log(sum(exp(-x/T))) * T
    # But we want to maximize min, so we minimize -min = -(-log(sum(exp(-x/T))) * T) = log(sum(exp(-x/T))) * T
    # This is smooth.
    
    def energy_and_grad(flat_x):
        x = flat_x.reshape((N, 2))
        
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1e-9)
        dist = np.sqrt(dist_sq)
        
        # Boundary distances
        r_i = np.linalg.norm(x, axis=1)
        d_b = 1.0 - r_i
        
        # We want to maximize the minimum of {d_ij/2, d_b_i}
        # Let's define the "constraints" as values we want to be large.
        # Values: V = [d_ij/2 for i<j, d_b_i for i]
        # We want to maximize min(V).
        # Surrogate: -T * log( sum(exp(-V/T)) )
        
        # To keep gradient stable and computation reasonable, we only consider 
        # the closest pairs and boundary points? 
        # For N=300, N^2 is 90k. That's fine.
        
        # However, computing exp for all 90k pairs every iteration might be slow.
        # The previous 1/d^2 repulsion was faster.
        # Let's try to improve the 1/d^2 approach by adding a term that 
        # specifically penalizes the *minimum* distance more strongly.
        
        # Actually, the standard repulsion 1/d^2 works well but can get stuck 
        # in local minima where the packing is not uniform.
        
        # Let's try a different energy:
        # E = sum_{i<j} max(0, (2r_target - d_ij))^2 + sum_i max(0, (r_target - (1-|x_i|)))^2
        # This is non-smooth.
        
        # Let's stick to L-BFGS-B with the repulsion energy but refine the 
        # optimization parameters and potentially add a second pass.
        
        # Gradient of E_pair = sum 1/d^2
        dist_sq_safe = dist_sq + 1e-9
        inv_dist_sq = 1.0 / dist_sq_safe
        np.fill_diagonal(inv_dist_sq, 0.0)
        E_pair = np.sum(inv_dist_sq)
        
        grad_factor = -2.0 / (dist_sq_safe**2)
        np.fill_diagonal(grad_factor, 0.0)
        grad_pair = np.sum(grad_factor[:, :, np.newaxis] * diff, axis=1)
        
        # Gradient of E_bound = sum 1/(1-r)^2
        r_safe = np.where(r_i < 1e-9, 1e-9, r_i)
        d_b_safe = np.where(d_b < 1e-6, 1e-6, d_b)
        E_bound = np.sum(1.0 / d_b_safe**2)
        grad_bound = 2.0 * (x / r_safe[:, np.newaxis]) / d_b_safe[:, np.newaxis]**3
        
        total_energy = E_pair + E_bound
        total_grad = grad_pair + grad_bound
        
        return total_energy, total_grad

    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # 3. Optimization using L-BFGS-B
    try:
        res = minimize(
            energy_and_grad, 
            best_points.flatten(), 
            jac=True, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 5000, 'disp': False, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        optimized_points = res.x.reshape((N, 2))
        
        if np.all(np.isfinite(optimized_points)):
            r_i = np.linalg.norm(optimized_points, axis=1)
            if np.all(r_i <= 1.0):
                score = calculate_score(optimized_points)
                if score > best_score:
                    best_points = optimized_points
                    best_score = score
    except Exception:
        pass

    # 4. Local Refinement / Polish
    # Perform a more targeted local search to squeeze out more radius.
    # We can try to move pairs of points that are closest together further apart,
    # while respecting the boundary.
    
    if time.time() < deadline - 10.0:
        current_points = best_points.copy()
        current_score = best_score
        
        # Use a small step size and iterate
        step_size = 0.0005
        no_improve_count = 0
        max_no_improve = 50
        
        for _ in range(2000):
            if time.time() > deadline - 5.0:
                break
            if no_improve_count > max_no_improve:
                break
            
            # Find the bottleneck
            diff = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            min_pair_dist = np.min(np.sqrt(dist_sq))
            
            r_i = np.linalg.norm(current_points, axis=1)
            min_bound_dist = np.min(1.0 - r_i)
            
            current_r = min(min_pair_dist / 2.0, min_bound_dist)
            
            if min_pair_dist / 2.0 < min_bound_dist:
                # Pair is limiting
                idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
                i, j = idx
                if i == j:
                    continue
                direction = current_points[i] - current_points[j]
                norm = np.linalg.norm(direction)
                if norm > 1e-9:
                    direction /= norm
                    # Push apart
                    old_i = current_points[i].copy()
                    old_j = current_points[j].copy()
                    current_points[i] += direction * step_size
                    current_points[j] -= direction * step_size
                    
                    new_score = calculate_score(current_points)
                    if new_score > current_score:
                        current_score = new_score
                        best_points = current_points.copy()
                        best_score = current_score
                        no_improve_count = 0
                    else:
                        # Revert
                        current_points[i] = old_i
                        current_points[j] = old_j
                        no_improve_count += 1
            else:
                # Boundary is limiting
                idx = np.argmin(1.0 - r_i)
                point = current_points[idx]
                norm = np.linalg.norm(point)
                if norm > 1e-9:
                    direction = -point / norm
                    old_p = current_points[idx].copy()
                    current_points[idx] += direction * step_size
                    
                    new_score = calculate_score(current_points)
                    if new_score > current_score:
                        current_score = new_score
                        best_points = current_points.copy()
                        best_score = current_score
                        no_improve_count = 0
                    else:
                        current_points[idx] = old_p
                        no_improve_count += 1
                        # Try a smaller step size if stuck
                        if no_improve_count % 10 == 0:
                            step_size *= 0.9

    return best_points
# EVOLVE-BLOCK-END
