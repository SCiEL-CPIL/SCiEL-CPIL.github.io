import numpy as np
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 10s buffer for safety
    
    # Strategy: Discretize support into a grid and optimize weights using projected gradient ascent.
    # The objective P is convex in the weights (quadratic form w^T M w).
    # We fix the support points to a grid to reduce the search space dimensionality 
    # and rely on the convexity to find the global optimum for that grid.
    # We try a few different grids (densities/ranges) and pick the best.
    
    def compute_P_fast(support, weights):
        """
        Compute P[X1 + X2 + X3 < 2*X4] efficiently.
        support: shape (n,)
        weights: shape (n,)
        """
        n = len(support)
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        p = weights / w_sum
        
        # Z = X1 + X2
        # We need P[Z < 2*X4 - X3]
        # P = sum_{k,l} p_k p_l P[Z < 2*s_l - s_k]
        
        # Pairwise sums of support
        s_sums = support[:, None] + support[None, :]
        # Flatten and sort
        s_sums_flat = s_sums.flatten()
        sort_idx = np.argsort(s_sums_flat)
        s_sorted = s_sums_flat[sort_idx]
        
        # Corresponding probabilities for Z
        p_prods = p[:, None] * p[None, :]
        p_prods_flat = p_prods.flatten()
        p_sorted = p_prods_flat[sort_idx]
        
        # Prefix sums of probabilities for Z
        p_prefix = np.cumsum(p_sorted)
        p_prefix_pad = np.zeros(n * n + 1)
        p_prefix_pad[1:] = p_prefix
        
        # Thresholds T_{kl} = 2*s_l - s_k
        # T shape (n, n) where T[k, l] = 2 * s_l - s_k
        T = 2.0 * support[None, :] - support[:, None]
        
        # Find indices for strict inequality Z < T
        # searchsorted with side='left' gives index of first element >= T
        # So index is the count of elements < T
        idx = np.searchsorted(s_sorted, T, side='left')
        
        # Probability mass for Z < T
        prob_z_less = p_prefix_pad[idx]
        
        # Total probability
        # P = sum_{k, l} p_k * p_l * prob_z_less[k, l]
        total_prob = np.sum(p[:, None] * p[None, :] * prob_z_less)
        
        return total_prob

    def compute_gradient(support, weights):
        """
        Compute gradient of P with respect to weights.
        P = sum_{k,l} p_k p_l Q_{kl} where Q_{kl} = P[Z < 2*s_l - s_k]
        Note that Q is symmetric in a specific way? No, Q_{kl} depends on s_k, s_l.
        However, the structure is P = p^T M p where M_{kl} = Q_{kl}.
        Gradient dP/dp_i = 2 * sum_j M_{ij} p_j (if M is symmetric).
        Is M symmetric? M_{kl} = P[X1+X2 < 2*s_l - s_k].
        M_{lk} = P[X1+X2 < 2*s_k - s_l].
        They are not necessarily equal.
        Wait, let's look at the definition again.
        P = sum_{k, l} w_k w_l / S^2 * Q_{kl}
        Let u_i = w_i / S. P = sum u_k u_l Q_{kl} = u^T Q u.
        dP/du_i = sum_l Q_{il} u_l + sum_k u_k Q_{ki} = (Q u)_i + (Q^T u)_i.
        
        Let's compute Q and Q^T u efficiently.
        Q_{kl} = P[Z < 2*s_l - s_k].
        """
        n = len(support)
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return np.zeros(n)
        p = weights / w_sum
        
        s_sums = support[:, None] + support[None, :]
        s_sums_flat = s_sums.flatten()
        sort_idx = np.argsort(s_sums_flat)
        s_sorted = s_sums_flat[sort_idx]
        
        p_prods = p[:, None] * p[None, :]
        p_prods_flat = p_prods.flatten()
        p_sorted = p_prods_flat[sort_idx]
        
        p_prefix = np.cumsum(p_sorted)
        p_prefix_pad = np.zeros(n * n + 1)
        p_prefix_pad[1:] = p_prefix
        
        T = 2.0 * support[None, :] - support[:, None]
        idx = np.searchsorted(s_sorted, T, side='left')
        Q = p_prefix_pad[idx]
        
        # grad_i = (Q p)_i + (Q^T p)_i
        term1 = Q @ p
        term2 = Q.T @ p
        grad = term1 + term2
        
        # The objective is P(w) = (w^T Q w) / (sum w)^2
        # Let S = sum w.
        # d/dw_i [ (w^T Q w) / S^2 ] = [ 2 (Q w)_i S^2 - (w^T Q w) * 2 S * 1 ] / S^4
        # = [ 2 (Q w)_i / S - 2 P ] / S  ? No.
        # Let f = w^T Q w, g = S^2.
        # d(f/g)/dw_i = (f_i g - f g_i) / g^2
        # f_i = 2 (Q w)_i (since Q is not symmetric, d(w^T Q w)/dw = Qw + Q^T w)
        # Actually, let's just use the chain rule on u = w/S.
        # P(u) = u^T Q u.
        # u_i = w_i / S.
        # du_j / dw_i = (delta_{ij} S - w_i) / S^2
        # dP/dw_i = sum_j (dP/du_j) (du_j/dw_i)
        # dP/du_j = (Q u)_j + (Q^T u)_j = grad_u_j
        # So dP/dw_i = sum_j grad_u_j * (delta_{ij} S - w_i) / S^2
        # = [ S * grad_u_i - w_i * sum_j grad_u_j ] / S^2
        # Note sum_j grad_u_j = sum_j (Q u + Q^T u)_j = sum (Q+Q^T) u.
        # Also note that for any probability vector u, sum_j u_j (Q u)_j = u^T Q u = P.
        # And sum_j u_j (Q^T u)_j = u^T Q^T u = u^T Q u = P.
        # So sum_j grad_u_j = 2 P / S? No, u is normalized.
        # sum_j (Q u)_j = 1^T Q u.
        # Let's just compute it numerically to be safe.
        
        grad_u = term1 + term2
        sum_grad_u = np.sum(grad_u)
        
        # grad_w_i = (S * grad_u_i - w_i * sum_grad_u) / S^2
        grad_w = (w_sum * grad_u - weights * sum_grad_u) / (w_sum * w_sum)
        
        return grad_w

    best_score = -1.0
    best_support = None
    best_weights = None
    
    # Define a few grid configurations
    # Grid 1: Fine grid on [0, 1]
    # Grid 2: Coarser grid on [0, 2]
    # Grid 3: Log-spaced grid? No, linear is safer for convexity on the simplex if support is fixed.
    
    grids = [
        np.linspace(0, 1, 64),
        np.linspace(0, 2, 64),
        np.linspace(0, 1, 128),
        np.linspace(0, 0.5, 32),
        np.linspace(0, 1, 32),
    ]
    
    for grid in grids:
        if time.time() > deadline - 20:
            break
            
        n = len(grid)
        support = grid.copy()
        
        # Initialize weights
        # Start with uniform, or maybe concentrated?
        # Uniform is a safe start for convex optimization.
        weights = np.ones(n) / n
        
        # Projected Gradient Ascent
        # Step size needs to be adaptive or small.
        # Lipschitz constant of gradient?
        # Let's use a simple backtracking line search or fixed small step.
        # Since it's convex, any descent direction on -grad is good.
        # We maximize P, so we move in direction of grad.
        
        step_size = 0.1
        prev_score = compute_P_fast(support, weights)
        
        # Run optimization for a fixed number of iterations or time limit
        iter_limit = 500
        time_limit_iter = time.time() + 30
        
        for it in range(iter_limit):
            if time.time() > time_limit_iter:
                break
                
            grad = compute_gradient(support, weights)
            
            # Normalize gradient to control step size
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
                
            # Update
            # Projected gradient: w_new = w + step * grad
            # Then project onto simplex {w >= 0, sum w = 1}
            
            w_raw = weights + step_size * grad
            
            # Project onto simplex
            # Algorithm:
            # 1. Sort w_raw in descending order
            # 2. Find rho such that w_i > lambda for i <= rho
            # 3. lambda = (sum_{i<=rho} w_i - 1) / rho
            # 4. w_i = max(w_i - lambda, 0)
            
            sorted_w = np.sort(w_raw)[::-1]
            cumsum = np.cumsum(sorted_w)
            
            # Find rho
            # Condition: sorted_w[rho-1] > (cumsum[rho-1] - 1) / rho
            # Let's find the largest rho satisfying this.
            # We can use a binary search or just loop since n is small (64-128).
            rho = 0
            for r in range(1, n + 1):
                val = (cumsum[r-1] - 1.0) / r
                if sorted_w[r-1] > val:
                    rho = r
                else:
                    break
            
            if rho == 0:
                # All weights become 0? Should not happen if sum w_raw is large enough.
                # If sum w_raw < 1, projection is all zeros? No, projection of point with sum < 1 onto simplex.
                # Actually, if sum w_raw < 1, the projection is just w_raw normalized? No.
                # The simplex is sum w = 1, w >= 0.
                # If sum w_raw < 1, the closest point is w_raw / sum w_raw?
                # Wait, the projection of x onto {w: w>=0, sum w=1}.
                # If sum x < 1, the projection is x / sum x?
                # Let's check. Minimize ||w - x||^2 s.t. w>=0, sum w=1.
                # Lagrangian: L = ||w-x||^2 - mu(sum w - 1) - sum nu_i w_i.
                # w_i = x_i + mu/2 - nu_i/2.
                # If nu_i = 0, w_i = x_i + mu/2.
                # sum w = sum x + n mu/2 = 1 => mu = 2(1 - sum x)/n.
                # w_i = x_i + (1 - sum x)/n.
                # This is valid if w_i >= 0.
                # If some w_i < 0, we set them to 0 and re-solve for the rest.
                
                # Standard simplex projection algorithm handles this.
                # If rho=0, it means no element satisfies the condition?
                # That implies sorted_w[0] <= (sorted_w[0]-1)/1 => sorted_w[0] <= sorted_w[0]-1 => 0 <= -1, impossible.
                # So rho should be at least 1.
                
                # Let's re-verify the loop.
                # If sorted_w[0] is very small, e.g., 0.
                # val = (0-1)/1 = -1. 0 > -1 is True. So rho=1.
                # So rho should never be 0.
                
                pass
            
            if rho > 0:
                lambda_val = (cumsum[rho-1] - 1.0) / rho
                weights = np.maximum(w_raw - lambda_val, 0)
            else:
                # Fallback: normalize
                w_sum = np.sum(w_raw)
                if w_sum > 1e-9:
                    weights = w_raw / w_sum
                else:
                    weights = np.ones(n) / n
            
            # Ensure weights sum to 1 exactly (floating point cleanup)
            w_sum = np.sum(weights)
            if w_sum > 1e-9:
                weights = weights / w_sum
            else:
                weights = np.ones(n) / n
            
            current_score = compute_P_fast(support, weights)
            
            if current_score > prev_score:
                prev_score = current_score
                # Maybe increase step size slightly?
                # step_size *= 1.05
            else:
                # Decrease step size
                step_size *= 0.8
                if step_size < 1e-6:
                    break
            
            # Check for convergence
            if abs(current_score - prev_score) < 1e-8:
                pass # Could break, but let's let it run to ensure we've explored enough
        
        if prev_score > best_score:
            best_score = prev_score
            best_support = support.copy()
            best_weights = weights.copy()
            
        # If we found a good score, maybe refine with smaller steps?
        # For now, just move to next grid.
        
    # Fallback if no grid worked (should not happen)
    if best_support is None:
        n = 8
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
        best_score = compute_P_fast(best_support, best_weights)
        
    # Final cleanup
    best_support = np.maximum(best_support, 0)
    best_weights = np.maximum(best_weights, 0)
    w_sum = np.sum(best_weights)
    if w_sum > 1e-9:
        best_weights = best_weights / w_sum
    else:
        best_weights = np.ones(len(best_weights)) / len(best_weights)
        
    return best_support, best_weights
