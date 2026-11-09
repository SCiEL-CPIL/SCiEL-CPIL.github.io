import numpy as np
import time
import random

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    start_time = time.time()
    timeout = 550  # Safety margin for 600s limit
    
    # Pre-compute index mappings for projections
    # P[y, x] where y is row, x is col
    # x in 0..9, y in 0..9
    ys, xs = np.indices((n, n))
    
    # r = -1: x - y. Range -9..9. Shift by 9 to get 0..18.
    idx_D = (xs - ys + 9).flatten()
    # r = 0: x. Range 0..9.
    idx_M0 = xs.flatten()
    # r = 1: x + y. Range 0..18.
    idx_M1 = (xs + ys).flatten()
    # r = 2: x + 2y. Range 0..27.
    idx_M2 = (xs + 2 * ys).flatten()
    
    max_val_D = 19
    max_val_M0 = 10
    max_val_M1 = 19
    max_val_M2 = 28
    
    def compute_entropies_and_grads(P):
        """
        Computes entropies H_D, H_M0, H_M1, H_M2 and their gradients w.r.t log(P).
        P is flattened.
        Returns:
            Hs: dict of entropies
            grads: dict of gradients (flat arrays of size 100)
            active_M: index of the max entropy among M0, M1, M2
        """
        p = P.flatten()
        
        def entropy_and_grad(idx, max_val):
            # Accumulate probabilities
            counts = np.zeros(max_val)
            np.add.at(counts, idx, p)
            
            # Avoid log(0)
            mask = counts > 1e-12
            if not np.any(mask):
                return 0.0, np.zeros(len(p))
            
            p_counts = counts[mask]
            H = -np.sum(p_counts * np.log2(p_counts))
            
            # Gradient of H w.r.t p_counts
            # dH/dp_i = - (log2(p_i) + 1/ln(2))
            # Note: p_counts are sums of p_j.
            # We need dH/dp_j for the original p_j.
            # dH/dp_j = sum_{k: j in idx_k} dH/dp_counts_k
            # Since each p_j maps to exactly one bin, 
            # dH/dp_j = - (log2(counts[idx_j]) + 1/ln(2)) if counts[idx_j] > 0 else 0
            
            # Create gradient array
            grad_p = np.zeros(len(p))
            # For each element j in p, it contributes to bin idx[j]
            # The derivative is -log2(p_bin) - 1/ln(2)
            # We can vectorize this by mapping idx back to values
            # Let's compute log2(counts) for all bins, handling zeros
            log_counts = np.zeros(max_val)
            log_counts[mask] = np.log2(p_counts)
            
            # Map back to flat array
            grad_p = -(log_counts[idx] + 1.0/np.log(2))
            # Zero out where counts are 0? 
            # If counts[idx[j]] == 0, then p_j must be 0 (since p>=0). 
            # In that case, the derivative is technically undefined or 0 in the support limit.
            # Setting to 0 is safe for gradient ascent on log-probs if we handle the softmax carefully.
            grad_p[~mask[idx]] = 0.0
            
            return H, grad_p

        H_D, grad_D = entropy_and_grad(idx_D, max_val_D)
        H_M0, grad_M0 = entropy_and_grad(idx_M0, max_val_M0)
        H_M1, grad_M1 = entropy_and_grad(idx_M1, max_val_M1)
        H_M2, grad_M2 = entropy_and_grad(idx_M2, max_val_M2)
        
        Hs = {'D': H_D, 'M0': H_M0, 'M1': H_M1, 'M2': H_M2}
        grads = {'D': grad_D, 'M0': grad_M0, 'M1': grad_M1, 'M2': grad_M2}
        
        # Determine active M
        M_vals = [H_M0, H_M1, H_M2]
        max_M = max(M_vals)
        if max_M < 1e-9:
            return Hs, grads, -1, max_M, None
            
        # The gradient of M is the gradient of the max component
        # If there are ties, we can average or pick one. Picking one is standard subgradient.
        active_idx = np.argmax(M_vals)
        active_grad = [grad_M0, grad_M1, grad_M2][active_idx]
        
        return Hs, grads, active_idx, max_M, active_grad

    def evaluate(P):
        """
        Compute the score H_D / M.
        """
        H_D, _, _, M, _ = compute_entropies_and_grads(P)
        if M < 1e-9:
            return -1.0
        return H_D / M

    def optimize(theta_init, max_iters=2000):
        """
        Gradient ascent on log-probabilities.
        """
        theta = theta_init.copy()
        
        # Initial P
        exp_theta = np.exp(theta - np.max(theta)) # Stability
        P = exp_theta / np.sum(exp_theta)
        
        current_score = evaluate(P)
        
        # Learning rate schedule
        lr = 0.5
        lr_decay = 0.999
        momentum = 0.9
        v = np.zeros_like(theta)
        
        for i in range(max_iters):
            if time.time() - start_time > timeout:
                break
            
            # Compute gradients
            Hs, grads, active_idx, M, active_grad = compute_entropies_and_grads(P)
            
            if M < 1e-9:
                # Invalid state, try to recover by adding noise or breaking
                break
                
            score = Hs['D'] / M
            
            if score > current_score:
                current_score = score
            
            # Gradient of J = H_D / M
            # dJ = (1/M) dH_D - (H_D / M^2) dM
            # Note: grads are dH/dp. We need dH/dtheta.
            # dP/dtheta = P * (I - P^T 1) ... wait, for softmax:
            # P_i = e^{theta_i} / Z
            # dP_i/dtheta_j = P_i (delta_ij - P_j)
            # So dH/dtheta = sum_k (dH/dP_k) (dP_k/dtheta)
            # dH/dtheta = sum_k g_k P_k (delta_kj - P_j) = P_j (sum_k g_k P_k - sum_k g_k P_k P_j)
            # Let S = sum_k g_k P_k. Then dH/dtheta = P (S - S * P) = P * S * (1 - P)
            
            def grad_theta(grad_p):
                # grad_p is dH/dP (flat)
                S = np.dot(grad_p, P.flatten())
                # dH/dtheta = P * S * (1 - P)
                return P.flatten() * S * (1 - P.flatten())
            
            grad_D_theta = grad_theta(grads['D'])
            grad_M_theta = grad_theta(active_grad)
            
            # Combine
            grad_J_theta = (1.0 / M) * grad_D_theta - (Hs['D'] / (M * M)) * grad_M_theta
            
            # Update with momentum
            v = momentum * v + lr * grad_J_theta
            theta = theta + v
            
            # Update P
            exp_theta = np.exp(theta - np.max(theta))
            P = exp_theta / np.sum(exp_theta)
            
            # Decay learning rate
            lr *= lr_decay
            
            # Periodic restart of momentum or lr boost if stuck?
            if i % 500 == 0 and i > 0:
                # Check if stuck
                pass
        
        return P, current_score

    # Initializations
    best_P = np.ones((n, n)) / (n * n)
    best_score = evaluate(best_P)
    
    # 1. Uniform
    theta_uni = np.zeros((n, n))
    P_opt, score = optimize(theta_uni, max_iters=500)
    if score > best_score:
        best_score = score
        best_P = P_opt
        
    # 2. Random sparse diagonals
    for restart in range(100):
        if time.time() - start_time > timeout:
            break
            
        # Random permutation
        perm = np.random.permutation(n)
        theta = np.full((n, n), -10.0) # Low probability initially
        for x in range(n):
            y = perm[x]
            theta[y, x] = 0.0 # High probability
            
        P_opt, score = optimize(theta, max_iters=1000)
        if score > best_score:
            best_score = score
            best_P = P_opt
            
    # 3. Random dense initializations with noise
    for restart in range(50):
        if time.time() - start_time > timeout:
            break
        theta = np.random.normal(0, 1, (n, n))
        P_opt, score = optimize(theta, max_iters=1000)
        if score > best_score:
            best_score = score
            best_P = P_opt

    # Final check and normalization
    best_P = np.clip(best_P, 0, None)
    total = np.sum(best_P)
    if total < 1e-9:
        return np.ones((n, n)) / (n * n)
    best_P = best_P / total
    
    return best_P
