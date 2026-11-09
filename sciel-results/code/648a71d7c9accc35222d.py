import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def get_score(points):
        """Calculate the exact minimum radius for a given set of points."""
        points = np.asarray(points).reshape(-1, 3)
        if points.shape != (N, 3):
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        if np.any(np.abs(points) > 0.5 + 1e-9):
            return 0.0
        
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Wall distances
        wall_dists = 0.5 - np.abs(points)
        min_wall = np.min(np.min(wall_dists, axis=1))
        
        return min(min_pair, min_wall)

    def generate_initial_grid():
        """Generate a 5x5x4 grid."""
        x5 = np.linspace(-0.4, 0.4, 5)
        y5 = np.linspace(-0.4, 0.4, 5)
        z4 = np.linspace(-0.3, 0.3, 4)
        grid = np.array([(x, y, z) for z in z4 for y in y5 for x in x5])
        return grid

    def penalty_objective(x_flat, r):
        """
        Penalty function for a given radius r.
        Minimizing this finds a valid packing if possible.
        """
        points = x_flat.reshape(-1, 3)
        penalty = 0.0
        
        # Pairwise distance penalty
        # We want |c_i - c_j| >= 2r
        # Penalty: max(0, 2r - |c_i - c_j|)^2
        diff = points[:, np.newaxis, :] - points[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1e6)
        
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        violations = np.maximum(0, 2 * r - pair_dists)
        penalty += np.sum(violations ** 2)
        
        # Wall distance penalty
        # We want 0.5 - |c_ik| >= r  => |c_ik| <= 0.5 - r
        # Penalty: max(0, r - (0.5 - |c_ik|))^2 = max(0, |c_ik| - (0.5 - r))^2
        wall_violations = np.maximum(0, np.abs(points) - (0.5 - r))
        penalty += np.sum(wall_violations ** 2)
        
        return penalty

    def penalty_gradient(x_flat, r):
        """
        Gradient of the penalty function.
        """
        points = x_flat.reshape(-1, 3)
        grad = np.zeros_like(points)
        
        # Pairwise distance gradient
        diff = points[:, np.newaxis, :] - points[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1e6)
        
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        violations = np.maximum(0, 2 * r - pair_dists)
        
        # Gradient of (2r - d)^2 w.r.t d is -2(2r - d)
        # Gradient of d w.r.t x_i is (x_i - x_j) / d
        # So grad_i += -2 * violations * (x_i - x_j) / d
        # We need to accumulate this for all pairs.
        
        # Vectorized approach for gradient accumulation is tricky with dense matrices.
        # For N=100, N^2 = 10000, which is small enough to loop in Python if needed, 
        # but let's try to vectorize.
        
        # Actually, L-BFGS-B can compute numerical gradients, which is slow.
        # Let's provide an analytical gradient.
        
        # For each pair (i, j):
        # g_i += -2 * v_ij * (p_i - p_j) / d_ij
        # g_j += -2 * v_ij * (p_j - p_i) / d_ij
        
        # We can use np.einsum or similar, but looping over violating pairs is faster if few.
        # However, initially many pairs might violate.
        
        # Let's stick to numerical gradient for simplicity and robustness, 
        # but that might be too slow. 
        # Let's try a simple loop for the gradient, it's only 100 points.
        
        # Reset grad
        grad = np.zeros((N, 3))
        
        for i in range(N):
            for j in range(i + 1, N):
                d_vec = points[i] - points[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-12:
                    d_vec = np.array([1e-12, 0, 0])
                    d = 1e-12
                v = 2 * r - d
                if v > 0:
                    # grad_i += -2 * v * d_vec / d
                    # grad_j += -2 * v * (-d_vec) / d = 2 * v * d_vec / d
                    factor = -2 * v / d
                    grad[i] += factor * d_vec
                    grad[j] -= factor * d_vec
        
        # Wall gradient
        # Penalty: max(0, |x| - (0.5 - r))^2
        # Let u = |x| - (0.5 - r)
        # If u > 0, penalty = u^2, grad = 2u * sign(x)
        wall_limit = 0.5 - r
        wall_violations = np.maximum(0, np.abs(points) - wall_limit)
        # grad = 2 * wall_violations * sign(points)
        # Note: sign(0) is 0, which is fine.
        grad += 2 * wall_violations * np.sign(points)
        
        return grad.flatten()

    # Initial configuration
    init_points = generate_initial_grid()
    
    # Binary search for radius
    r_low = 0.100
    r_high = 0.115 # Best known is ~0.1114
    
    best_points = init_points
    best_score = get_score(init_points)
    
    # We will iterate a few times, increasing r and checking feasibility.
    # Instead of strict binary search, we can do a linear sweep or adaptive step.
    # Let's try to jump to a high r and see if we can fix it.
    
    # Start with a slightly higher r than the grid's natural radius
    # Grid 5x5x4 has min distance 0.2, so r=0.1.
    # Let's try to push r up.
    
    current_r = best_score
    max_iter = 20
    step_size = 0.001
    
    for _ in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        # Try to increase radius slightly
        target_r = current_r + step_size
        
        # Run optimization to find a packing with radius target_r
        x0 = best_points.flatten()
        
        try:
            res = minimize(
                lambda x: penalty_objective(x, target_r),
                x0,
                jac=lambda x: penalty_gradient(x, target_r),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if time.time() > deadline - 5:
                break
                
            new_points = res.x.reshape(-1, 3)
            new_points = np.clip(new_points, -0.5, 0.5)
            
            # Check if the packing is actually valid for target_r
            # Calculate the actual min radius
            actual_r = get_score(new_points)
            
            if actual_r >= target_r - 1e-6:
                # Success! We found a valid packing for target_r
                current_r = actual_r
                best_points = new_points
                best_score = actual_r
                # Keep step size or increase slightly
                # step_size = step_size * 1.1 
            else:
                # Failed to reach target_r
                # Maybe we can still improve current_r slightly?
                # If actual_r > current_r, update
                if actual_r > current_r:
                    current_r = actual_r
                    best_points = new_points
                    best_score = actual_r
                # Reduce step size
                step_size = step_size * 0.5
                if step_size < 1e-5:
                    break
                    
        except Exception:
            break

    return best_points
