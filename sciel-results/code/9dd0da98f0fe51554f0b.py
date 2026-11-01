import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 115  # Safety margin
    
    def _score(centers):
        """Compute the actual score (minimum radius) for a configuration."""
        c = centers
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist2, np.inf)
        min_dist = np.sqrt(np.min(dist2)) / 2.0
        
        abs_c = np.abs(c)
        max_coord = np.max(abs_c, axis=1)
        min_wall = 0.5 - np.max(max_coord)
        
        return min(min_dist, min_wall)
    
    def _smooth_min_distance(centers, beta):
        """
        Compute a smooth surrogate for the minimum pairwise distance using log-sum-exp.
        Returns (smooth_min_dist, gradient)
        """
        c = centers
        n = len(c)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist2, 0)
        
        # Only consider upper triangle to avoid double counting? 
        # For gradient, it's easier to work with full matrix and divide by 2 or use symmetry.
        # Let's use full matrix for simplicity in einsum, then divide weights by 2? 
        # No, soft min over all pairs i<j.
        # If we use full matrix, we count each pair twice.
        # sum_{i != j} exp(-d_ij/beta) = 2 * sum_{i<j} exp(-d_ij/beta).
        # The log term will have a +ln(2) shift.
        # The weights w_ij will be halved compared to the i<j sum.
        # The gradient sum will be correct if we sum over all j != i.
        # Let's stick to i<j for clarity and efficiency in memory, but vectorize the grad.
        
        iu = np.triu_indices(n, k=1)
        d2 = dist2[iu]
        
        if len(d2) == 0:
            return 0.0, np.zeros_like(c)
        
        d = np.sqrt(d2)
        min_d = np.min(d)
        exp_d = np.exp(-(d - min_d) / beta)
        sum_exp = np.sum(exp_d)
        log_sum_d = np.log(sum_exp)
        soft_min_d = min_d + beta * log_sum_d
        
        # Weights for i<j pairs
        weights = exp_d / sum_exp
        
        # Gradient calculation
        # grad[i] += w * (c[i] - c[j]) / d
        # grad[j] -= w * (c[i] - c[j]) / d
        
        grad = np.zeros_like(c)
        
        # Vectorized update
        # Get the pairs
        i_idx, j_idx = iu
        
        # Calculate unit vectors for these pairs
        # diff_vec = c[i] - c[j]
        diff_vec = c[i_idx] - c[j_idx]
        
        # Avoid division by zero
        d_safe = np.where(d > 1e-10, d, 1.0)
        unit_vecs = diff_vec / d_safe[:, np.newaxis]
        
        # Weighted unit vectors
        weighted_vecs = weights[:, np.newaxis] * unit_vecs
        
        # Accumulate gradients
        # np.add.at is efficient for this
        np.add.at(grad, i_idx, weighted_vecs)
        np.add.at(grad, j_idx, -weighted_vecs)
        
        return soft_min_d, grad
    
    def _smooth_wall_distance(centers, beta):
        """
        Compute smooth surrogate for minimum wall distance.
        """
        c = centers
        abs_c = np.abs(c)
        # max_coord is the max absolute value for each point
        max_coord = np.max(abs_c, axis=1)
        wall_dists = 0.5 - max_coord
        
        min_wall = np.min(wall_dists)
        exp_w = np.exp(-(wall_dists - min_wall) / beta)
        sum_exp_w = np.sum(exp_w)
        log_sum_w = np.log(sum_exp_w)
        soft_wall = min_wall + beta * log_sum_w
        
        # Gradient
        weights = exp_w / sum_exp_w
        
        grad = np.zeros_like(c)
        
        # For each point i, the gradient is -w_i * sign(c_ik) where k is argmax(abs(c_ik))
        # If there are ties, the gradient is ambiguous, but numerically we can pick one or average.
        # Let's find the argmax indices
        max_indices = np.argmax(abs_c, axis=1)
        
        # Signs
        signs = np.sign(c[np.arange(N), max_indices])
        
        # Update gradient
        # grad[i][k] -= w_i * sign
        grad[np.arange(N), max_indices] -= weights * signs
        
        return soft_wall, grad
    
    def _smooth_objective(centers, r, beta):
        """
        Smooth objective: penalize when soft_min_dist < 2r or soft_wall < r
        """
        soft_d, grad_d = _smooth_min_distance(centers, beta)
        soft_w, grad_w = _smooth_wall_distance(centers, beta)
        
        # Penalty terms
        pair_viol = max(0.0, 2*r - soft_d)
        wall_viol = max(0.0, r - soft_w)
        
        obj = pair_viol**2 + wall_viol**2
        
        # Gradient
        grad = np.zeros_like(centers)
        if pair_viol > 0:
            grad -= 2 * pair_viol * grad_d
        if wall_viol > 0:
            grad -= 2 * wall_viol * grad_w
        
        return obj, grad
    
    def _initial_grid():
        """Generate initial grid configuration."""
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()
    
    def _refine_lloyd(centers, r, max_iter=100, tol=1e-8):
        """
        Lloyd-style refinement: move each center away from closest neighbors.
        """
        for it in range(max_iter):
            if time.time() - start_time > timeout - 5:
                break
                
            c = centers.copy()
            for i in range(N):
                # Find closest other center
                diff = c - c[i]
                dist2 = np.sum(diff * diff, axis=1)
                dist2[i] = np.inf
                j = np.argmin(dist2)
                d = np.sqrt(dist2[j])
                
                if d < 2 * r:
                    # Move i away from j
                    direction = (c[i] - c[j]) / d
                    c[i] += 0.1 * direction * (2 * r - d)
            
            # Project back into cube
            c = np.clip(c, -0.5 + r, 0.5 - r)
            
            centers = c
            
            # Check convergence
            if np.allclose(centers, c, atol=tol):
                break
        
        return centers
    
    def minimize_for_radius(r, initial_centers, beta_schedule=None):
        """
        Try to find a configuration for radius r using smooth optimization.
        """
        if beta_schedule is None:
            beta_schedule = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]
        
        x0 = initial_centers.flatten()
        bounds = [(-0.5 + r - 0.01, 0.5 - r + 0.01)] * (3 * N)
        
        best_x = x0.copy()
        best_obj = np.inf
        
        for beta in beta_schedule:
            if time.time() - start_time > timeout - 5:
                break
            
            def objective(x):
                c = x.reshape(N, 3)
                obj, _ = _smooth_objective(c, r, beta)
                return obj
            
            def gradient(x):
                c = x.reshape(N, 3)
                _, grad = _smooth_objective(c, r, beta)
                return grad.flatten()
            
            result = minimize(
                objective,
                best_x,
                jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x.copy()
        
        # Final polish with tight tolerance
        if time.time() - start_time < timeout - 5:
            def final_objective(x):
                c = x.reshape(N, 3)
                obj, _ = _smooth_objective(c, r, 0.01)
                return obj
            
            def final_gradient(x):
                c = x.reshape(N, 3)
                _, grad = _smooth_objective(c, r, 0.01)
                return grad.flatten()
            
            result = minimize(
                final_objective,
                best_x,
                jac=final_gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'disp': False, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x.copy()
        
        best_centers = best_x.reshape(N, 3)
        
        # Check if feasible
        actual_score = _score(best_centers)
        success = actual_score >= r - 1e-8
        
        return success, best_centers, best_obj
    
    # Initial configuration
    current_centers = _initial_grid()
    
    # Binary search on radius
    r_low = 0.125
    r_high = 0.140
    
    best_centers = current_centers.copy()
    best_score = _score(best_centers)
    
    # Add a restart mechanism: if we fail to find a solution for r_mid, 
    # we might be stuck. We can try perturbing the current best and retrying.
    
    while time.time() - start_time < timeout and (r_high - r_low) > 1e-5:
        r_mid = (r_low + r_high) / 2.0
        
        success, new_centers, energy = minimize_for_radius(r_mid, current_centers)
        
        if success:
            r_low = r_mid
            current_centers = new_centers
            score = _score(new_centers)
            if score > best_score:
                best_score = score
                best_centers = new_centers
        else:
            r_high = r_mid
            # Update current_centers to the best found, even if not feasible
            current_centers = new_centers
            
            # If we failed, try a small random perturbation to escape local minima
            if time.time() - start_time < timeout - 10:
                # Perturb current_centers
                noise = np.random.normal(0, 0.01, size=(N, 3))
                perturbed = current_centers + noise
                # Clip to bounds roughly
                perturbed = np.clip(perturbed, -0.5 + r_mid, 0.5 - r_mid)
                
                success2, new_centers2, _ = minimize_for_radius(r_mid, perturbed)
                if success2:
                    r_low = r_mid
                    current_centers = new_centers2
                    score = _score(new_centers2)
                    if score > best_score:
                        best_score = score
                        best_centers = new_centers2
                else:
                    # If still failed, keep the less-bad one
                    current_centers = new_centers2 if _score(new_centers2) > _score(current_centers) else current_centers

        # Periodically try Lloyd refinement
        if int((r_high - r_low) * 100000) % 3 == 0:
            refined = _refine_lloyd(current_centers, r_low, max_iter=50)
            score = _score(refined)
            if score > best_score:
                best_score = score
                best_centers = refined
                current_centers = refined
    
    # Final refinement
    if time.time() - start_time < timeout - 5:
        final_refined = _refine_lloyd(best_centers, best_score, max_iter=200)
        final_score = _score(final_refined)
        if final_score > best_score:
            best_centers = final_refined
    
    return best_centers
# EVOLVE-BLOCK-END
