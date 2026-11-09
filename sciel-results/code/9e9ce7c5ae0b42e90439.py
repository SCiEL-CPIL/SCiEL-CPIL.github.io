import numpy as np
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy:
    1. Exploit the structural insight that the optimal measure is 3-point with support {0, 1, M}.
    2. Perform a high-resolution, vectorized 2D grid search over M (log-spaced) and w_1 (uniform).
    3. Use the best grid point as a starting point for a refined local search if time permits, 
       though the grid is expected to capture the global optimum given the smoothness in the region.
    4. Return the best valid measure found.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    def calc_prob_3point(s0, s1, s2, w0, w1, w2):
        """
        Calculate P[X1+X2+X3 < 2*X4] for a 3-point distribution.
        s = [s0, s1, s2], w = [w0, w1, w2]
        """
        s = np.array([s0, s1, s2], dtype=np.float64)
        w = np.array([w0, w1, w2], dtype=np.float64)
        
        # Normalize weights (should sum to 1, but ensure)
        w_sum = np.sum(w)
        if w_sum <= 1e-9:
            return 0.0
        w = w / w_sum
        
        # Compute all pairwise sums S2[i,j] = s_i + s_j and probabilities P2[i,j] = w_i w_j
        S2 = s[:, None] + s[None, :]
        P2 = w[:, None] * w[None, :]
        
        p_total = 0.0
        # Loop over k (X4) and l (X3)
        for k in range(3):
            for l in range(3):
                threshold = 2.0 * s[k] - s[l]
                # Sum P2 where S2 < threshold
                mask = S2 < threshold
                prob_sum = np.sum(P2[mask])
                p_total += w[k] * w[l] * prob_sum
                
        return p_total

    best_score = -np.inf
    best_params = None
    
    # Structural assumption: Support is {0, 1, M}.
    # Weights are (1 - w1 - w2, w1, w2).
    # Since the distribution is symmetric in a sense? No, 0 and 1 are distinct.
    # However, usually the mass is heavily on 0 and M, with small mass on 1?
    # Or maybe mass on 0 and 1, with small mass on M?
    # Let's search w1 in [0, 1] and w2 in [0, 1-w1].
    # To be efficient, we can fix the sum of w1 and w2? No, they are independent.
    
    # But wait, the previous best was around 1.27.
    # Let's look at the structure of the optimal solution for {0, 1, M}.
    # If M is large, the term 2*X4 - X3 is dominated by 2*X4 if X4=M.
    # If X4=M, condition is X1+X2+X3 < 2M. Since X1,X2,X3 <= M, this is always true unless X1=X2=X3=M.
    # So if X4=M, P[win] is high.
    # If X4=0, condition is X1+X2+X3 < 0, impossible.
    # If X4=1, condition is X1+X2+X3 < 2.
    
    # Let's define the parameters:
    # s0 = 0
    # s1 = 1
    # s2 = M
    # w0, w1, w2 are the weights.
    
    # We will search over M and (w1, w2).
    # To reduce dimensionality, we can parametrize weights using two variables.
    # Let's use a grid over M and w1, and optimize w2? Or just grid over w1 and w2?
    # 3D grid might be too slow if fine.
    # Let's try a 2D grid over M and w1, and for each (M, w1), find the best w2 analytically or via small 1D search.
    
    # Actually, let's just do a 3D grid but with coarse resolution first, then refine.
    # Or, notice that w0 = 1 - w1 - w2.
    
    # Let's use a 2D grid over M and w1, and assume w2 is small? 
    # No, let's be more systematic.
    
    # Range for M: 1e-1 to 1e4
    # Range for w1: 0 to 1
    # Range for w2: 0 to 1-w1
    
    # Let's use a smarter approach:
    # 1. Coarse 3D grid.
    # 2. Refine around the best point.
    
    # Coarse Grid
    M_vals = np.logspace(-1, 4, 200)
    w1_vals = np.linspace(0.01, 0.99, 100)
    # For each w1, w2 can range from 0.01 to 0.99-w1
    # This is variable size. Let's just sample w2 uniformly in [0, 1-w1] with a fixed number of points.
    w2_count = 50
    
    # To make it vectorizable, we can loop over M and w1, and vectorize over w2.
    # Or loop over M, and vectorize over (w1, w2).
    
    # Let's loop over M (200 iterations).
    # For each M, create a grid of w1 and w2.
    # w1 in [0, 1], w2 in [0, 1-w1].
    # We can sample w1, w2 from a triangular distribution? 
    # Simpler: Sample w1, w2 such that w1+w2 <= 1.
    # Let u1 ~ U(0,1), u2 ~ U(0,1).
    # If u1+u2 > 1, reflect? 
    # Or just use a fixed grid on the simplex.
    
    # Let's use a fixed number of points on the simplex for weights.
    # N_simplex points.
    N_w = 200
    # Generate N_w points on the simplex (w0, w1, w2) summing to 1.
    # We can do this by sampling from Dirichlet(1,1,1) or using a grid.
    # A grid on the simplex:
    # w1 from 0 to 1, w2 from 0 to 1-w1.
    # Let's create a meshgrid for w1 and w2.
    
    # To keep it fast, let's use a coarser grid first.
    M_coarse = np.logspace(-1, 4, 100)
    w1_coarse = np.linspace(0.0, 1.0, 50)
    w2_coarse = np.linspace(0.0, 1.0, 50)
    
    # We will iterate over M_coarse.
    # For each M, we will evaluate all (w1, w2) pairs where w1+w2 <= 1.
    
    for M in M_coarse:
        if time.time() > deadline - 10:
            break
            
        # Create meshgrid for w1, w2
        W1, W2 = np.meshgrid(w1_coarse, w2_coarse, indexing='ij')
        W0 = 1.0 - W1 - W2
        
        # Mask valid weights (w0, w1, w2 >= 0)
        mask = (W0 >= 0) & (W1 >= 0) & (W2 >= 0)
        
        if not np.any(mask):
            continue
            
        # Extract valid weights
        w1_valid = W1[mask]
        w2_valid = W2[mask]
        w0_valid = W0[mask]
        
        # Now we have arrays of weights.
        # We need to calculate P for each weight set with fixed s = [0, 1, M].
        # P = sum_{k,l} w_k w_l sum_{i,j} w_i w_j 1[s_i+s_j < 2s_k - s_l]
        
        # Let's precompute the condition matrix for fixed s.
        s = np.array([0.0, 1.0, M])
        S2 = s[:, None] + s[None, :]
        
        # For each k, l, the threshold is T_kl = 2*s_k - s_l.
        # The inner sum is sum_{i,j} w_i w_j 1[S2_ij < T_kl].
        # Let C_kl be the matrix of indicators 1[S2_ij < T_kl].
        # Then inner_sum = w^T C_kl w.
        # Total P = sum_{k,l} w_k w_l (w^T C_kl w).
        
        # We can vectorize this over the batch of weight vectors.
        # w_batch has shape (N, 3).
        # We want to compute for each row n:
        # P_n = sum_{k,l} w_n[k] w_n[l] (w_n^T C_kl w_n)
        
        # Let's compute C_kl for all k,l.
        C = np.zeros((3, 3, 3, 3))
        for k in range(3):
            for l in range(3):
                T = 2.0 * s[k] - s[l]
                C[k, l] = (S2 < T).astype(np.float64)
                
        # Now compute P for each w in w_batch.
        # w_batch shape (N, 3)
        w_batch = np.column_stack((w0_valid, w1_valid, w2_valid))
        
        # P_n = sum_{k,l} w_n[k] w_n[l] sum_{i,j} w_n[i] w_n[j] C_kl[i,j]
        # Let's expand the sum:
        # P_n = sum_{k,l,i,j} C_kl[i,j] w_n[k] w_n[l] w_n[i] w_n[j]
        # This is a 4th order tensor contraction.
        # We can do this efficiently with einsum or loops over k,l.
        
        # Loop over k, l (9 iterations)
        p_batch = np.zeros(len(w_batch))
        for k in range(3):
            for l in range(3):
                # Term is w_n[k] w_n[l] * (w_n^T C_kl w_n)
                # w_n^T C_kl w_n can be computed as einsum('ni,ij,nj->n', w_batch, C[k,l], w_batch)
                inner = np.einsum('ni,ij,nj->n', w_batch, C[k, l], w_batch)
                p_batch += w_batch[:, k] * w_batch[:, l] * inner
                
        # Find the best in this batch
        idx_max = np.argmax(p_batch)
        p_max = p_batch[idx_max]
        
        if p_max > best_score:
            best_score = p_max
            best_params = (M, w0_valid[idx_max], w1_valid[idx_max], w2_valid[idx_max])
            
    # Refinement Phase
    # If we found a good candidate, refine it.
    if best_params is not None:
        M_best, w0_best, w1_best, w2_best = best_params
        
        # Define a local objective function
        def objective(params):
            M, w0, w1, w2 = params
            if M < 0 or w0 < 0 or w1 < 0 or w2 < 0:
                return np.inf
            if w0 + w1 + w2 > 1.0 + 1e-6:
                return np.inf
            # Normalize weights if slightly off
            w_sum = w0 + w1 + w2
            if w_sum < 1e-9:
                return np.inf
            w0_n = w0 / w_sum
            w1_n = w1 / w_sum
            w2_n = w2 / w_sum
            
            p = calc_prob_3point(0.0, 1.0, M, w0_n, w1_n, w2_n)
            return -p
            
        # Initial guess
        init = np.array([M_best, w0_best, w1_best, w2_best])
        
        try:
            from scipy.optimize import minimize
            # Use Nelder-Mead for robustness
            res = minimize(objective, init, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < -best_score:
                M_res, w0_res, w1_res, w2_res = res.x
                # Ensure validity
                if M_res >= 0 and w0_res >= 0 and w1_res >= 0 and w2_res >= 0:
                    w_sum = w0_res + w1_res + w2_res
                    if w_sum > 1e-9:
                        w0_res /= w_sum
                        w1_res /= w_sum
                        w2_res /= w_sum
                        
                        p_res = calc_prob_3point(0.0, 1.0, M_res, w0_res, w1_res, w2_res)
                        if p_res > best_score:
                            best_score = p_res
                            best_params = (M_res, w0_res, w1_res, w2_res)
                            
        except Exception:
            pass

    # Finalize
    if best_params is None:
        # Fallback
        M_best = 10.0
        w0_best = 0.5
        w1_best = 0.3
        w2_best = 0.2
    else:
        M_best, w0_best, w1_best, w2_best = best_params
        
    support = np.array([0.0, 1.0, M_best], dtype=np.float64)
    weights = np.array([w0_best, w1_best, w2_best], dtype=np.float64)
    
    # Normalize weights
    w_sum = np.sum(weights)
    if w_sum > 1e-9:
        weights = weights / w_sum
    else:
        weights = np.ones(3) / 3.0
        
    # Ensure non-negative
    weights = np.maximum(weights, 0.0)
    support = np.maximum(support, 0.0)
    
    return support, weights

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print(s)
    print(w)
