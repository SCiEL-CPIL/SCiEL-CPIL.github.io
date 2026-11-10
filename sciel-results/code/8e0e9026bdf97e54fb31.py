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
        # Pairwise min distance
        # Using broadcasting for N=300 is O(N^2) memory, 300x300x2 is small.
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_dist = np.min(np.sqrt(dist_sq))
        
        # Boundary min distance
        r_i = np.linalg.norm(points, axis=1)
        min_bound_dist = np.min(1.0 - r_i)
        
        return min(min_pair_dist / 2.0, min_bound_dist)

    def gen_hex_lattice(spacing):
        """Generate hexagonal lattice points inside the unit circle."""
        # Estimate range
        k = int(1.0 / spacing) + 2
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                # We need r + circle_radius <= 1.
                # We don't know circle_radius yet, but it's approx spacing/2.
                # Using spacing/2 as a conservative estimate for the margin.
                if r <= 1.0 - spacing * 0.5:
                    pts.append((x, y))
        return np.array(pts) if pts else np.empty((0, 2))

    # 1. Initialization
    # Estimate spacing to fill the circle with N points
    # Area of unit circle = pi. Area of hex cell = sqrt(3)/2 * s^2.
    # N * sqrt(3)/2 * s^2 ~ pi  => s ~ sqrt(2*pi / (sqrt(3)*N))
    spacing = np.sqrt(2.0 * np.pi / (np.sqrt(3.0) * N))
    
    # Adjust spacing to ensure we have at least N points
    # If we have too few, reduce spacing. If too many, we'll trim.
    points = gen_hex_lattice(spacing)
    
    # Loop to ensure enough points if initial estimate was too large
    while len(points) < N and spacing > 0.01:
        spacing *= 0.95
        points = gen_hex_lattice(spacing)
        
    if len(points) >= N:
        # Select the N points closest to the center
        # This biases towards the center, but since spacing is tuned to N,
        # it mostly just trims the outer boundary irregularities.
        dists = np.linalg.norm(points, axis=1)
        sorted_idx = np.argsort(dists)
        init_points = points[sorted_idx[:N]]
    else:
        # Fallback: pad with random points if we still don't have enough
        # (Should not happen with the loop above)
        init_points = points
        while len(init_points) < N:
            init_points = np.vstack([init_points, np.random.uniform(-0.9, 0.9, size=(1, 2))])
        init_points = init_points[:N]

    best_points = init_points.copy()
    best_score = calculate_score(init_points)

    # 2. Define energy function for L-BFGS-B
    def energy_and_grad(flat_x):
        x = flat_x.reshape((N, 2))
        
        # Calculate pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero
        dist_sq_safe = dist_sq + 1e-9
        np.fill_diagonal(dist_sq_safe, 1.0)
        
        # Energy term 1: Pairwise repulsion
        # E_pair = sum_{i!=j} 1/d_ij^2
        inv_dist_sq = 1.0 / dist_sq_safe
        np.fill_diagonal(inv_dist_sq, 0.0)
        E_pair = np.sum(inv_dist_sq)
        
        # Gradient of E_pair
        # d/dx_i (1/|x_i - x_j|^2) = -2 (x_i - x_j) / |x_i - x_j|^4
        grad_factor = -2.0 / (dist_sq_safe**2)
        np.fill_diagonal(grad_factor, 0.0)
        grad_pair = np.sum(grad_factor[:, :, np.newaxis] * diff, axis=1)
        
        # Energy term 2: Boundary repulsion
        r_i = np.linalg.norm(x, axis=1)
        r_safe = np.where(r_i < 1e-9, 1e-9, r_i)
        d_b = 1.0 - r_i
        # If a point is outside, d_b is negative.
        # We want to push it back.
        # Using 1/d_b^2 is problematic if d_b < 0 (it's still positive, but gradient direction?)
        # If d_b < 0, 1/d_b^2 is positive.
        # Grad = 2 * (x/r) / d_b^3.
        # If d_b is negative, d_b^3 is negative.
        # x/r is outward.
        # So grad is inward. Correct.
        # However, if d_b is close to 0, it blows up.
        # We clamp d_b to a small positive value for stability?
        # No, if we clamp, the gradient might point wrong way.
        # Better to just let it be, but add a large penalty if outside.
        # Actually, L-BFGS-B with bounds (-1, 1) helps keep them inside.
        # But we want them strictly inside.
        # Let's use a soft barrier.
        # If d_b < 0, we are in a penalty region.
        # To keep it simple and robust, we assume the optimizer stays mostly inside
        # due to the repulsion from other points and the boundary term.
        
        d_b_safe = np.where(d_b < 1e-6, 1e-6, d_b)
        
        E_bound = np.sum(1.0 / d_b_safe**2)
        
        # Gradient of E_bound
        # d/dx_i (1/d_b^2) = 2 * (x_i / r_i) / d_b^3
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
            options={'maxiter': 3000, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        optimized_points = res.x.reshape((N, 2))
        
        # Check validity
        if np.all(np.isfinite(optimized_points)):
            # Ensure points are inside the circle
            r_i = np.linalg.norm(optimized_points, axis=1)
            if np.all(r_i <= 1.0):
                score = calculate_score(optimized_points)
                if score > best_score:
                    best_points = optimized_points
                    best_score = score
    except Exception:
        pass

    # 4. Local Refinement
    # If we have time, perform a few steps of local search on the bottleneck.
    if time.time() < deadline - 5.0:
        current_points = best_points.copy()
        current_score = best_score
        # Use a small step size
        step_size = 0.001
        
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            
            # Find bottleneck
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
                if i == j: continue
                direction = current_points[i] - current_points[j]
                norm = np.linalg.norm(direction)
                if norm > 1e-9:
                    direction /= norm
                    # Push apart
                    current_points[i] += direction * step_size
                    current_points[j] -= direction * step_size
            else:
                # Boundary is limiting
                idx = np.argmin(1.0 - r_i)
                point = current_points[idx]
                norm = np.linalg.norm(point)
                if norm > 1e-9:
                    direction = -point / norm
                    current_points[idx] += direction * step_size
            
            new_score = calculate_score(current_points)
            if new_score > current_score:
                current_score = new_score
                best_points = current_points.copy()
                best_score = current_score
            else:
                # Revert if not improved? 
                # Since we modified in place, we need to revert.
                # This is inefficient. 
                # Alternative: just accept small degradations? No.
                # Let's just stop if no improvement for a while?
                # Or simply don't do this local search if it's complex to revert.
                # Given L-BFGS-B is already good, maybe skip this to save code complexity?
                # Inspiration 2 had this but it was messy.
                # Let's stick to L-BFGS-B. It's robust.
                break

    return best_points
# EVOLVE-BLOCK-END
