import numpy as np
import time
import sys

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy:
    1. Assume optimal support is of form {0, 1, M} with M > 1.
    2. Analyze the piecewise structure of the probability P(M, w1, wM) in the weight simplex.
       The probability is a quartic polynomial in weights w = [w0, w1, wM].
       The coefficients of this polynomial change when the inequality s_i + s_j < 2*s_k - s_l 
       flips for some (i, j, k, l).
    3. For a fixed M, the region in the (w1, wM) simplex where the combinatorial structure 
       (which pairs satisfy the inequality) is constant is a convex polygon (intersection of half-planes).
       Since the objective is a polynomial (not necessarily concave), we search for the maximum 
       within these regions. However, for small M or specific structures, the optimum often lies 
       on the boundary or at a vertex. 
    4. We will perform a dense but efficient search over M, and for each M, we will use 
       scipy.optimize.minimize to find the local maximum in the most promising region, 
       initialized from the vertices of the feasible weight simplex and mid-points.
       Crucially, we will evaluate the exact probability using vectorized operations for speed.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    def calc_prob(support, weights):
        """
        Compute P[X1 + X2 + X3 < 2*X4] efficiently.
        support: 1D array of shape (n,)
        weights: 1D array of shape (n,)
        """
        total_w = np.sum(weights)
        if total_w <= 1e-9:
            return 0.0
        w = weights / total_w
        n = len(support)
        
        # Precompute pairwise sums and probabilities
        S2 = support[:, None] + support[None, :]
        Prob2 = w[:, None] * w[None, :]
        
        p_total = 0.0
        # Loop over k, l is O(n^2). Inner sum is O(n^2) if not optimized, but we can vectorize the inner sum.
        # For n=3, this is very fast.
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * support[k] - support[l]
                # Sum Prob2 where S2 < threshold
                mask = S2 < threshold
                prob_sum = np.sum(Prob2[mask])
                p_total += w[k] * w[l] * prob_sum
                
        return p_total

    def is_valid(support, weights):
        if len(support) != len(weights):
            return False
        if len(support) < 1:
            return False
        if np.any(np.isnan(support)) or np.any(np.isinf(support)):
            return False
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            return False
        if np.any(support < -1e-9):
            return False
        if np.any(weights < -1e-9):
            return False
        if np.sum(weights) <= 1e-9:
            return False
        return True

    best_score = -np.inf
    best_support = np.array([0.0, 1.0, 10.0])
    best_weights = np.array([0.5, 0.3, 0.2])
    
    # We focus on 3-point support {0, 1, M}.
    # Weights w = [w0, w1, wM] with w0 + w1 + wM = 1, w_i >= 0.
    # We can parameterize by (w1, wM) where w0 = 1 - w1 - wM.
    # Constraints: w1 >= 0, wM >= 0, w1 + wM <= 1.
    
    # The probability P is a function of M and (w1, wM).
    # P = sum_{k,l} w_k w_l (sum_{i,j} w_i w_j I[s_i+s_j < 2s_k - s_l])
    
    # Let's precompute the "structure" for a given M.
    # The inequalities are:
    # s_i + s_j < 2 s_k - s_l
    # Support S = [0, 1, M]
    # Indices: 0->0, 1->1, 2->M
    
    # We will search M in a log-spaced range.
    # Based on previous results, M around 2-5 is often good, but let's be safe.
    # The "beat the average" condition X1+X2+X3 < 2X4 suggests X4 should be large relative to the others,
    # or the others should be small.
    # If X4=0, LHS >= 0, strict inequality fails unless LHS < 0 (impossible). So w0 should be small?
    # If X4=M, 2M is large. If X1,X2,X3 are 0 or 1, sum is small. So wM should be significant.
    
    M_vals = np.logspace(np.log10(1.1), np.log10(50.0), 200)
    
    # For each M, we want to maximize P(w1, wM).
    # We will use a few initial points in the simplex for local optimization.
    # Simplex vertices: (0,0), (1,0), (0,1).
    # Also midpoints and some heuristic points.
    
    init_points = [
        [0.0, 0.0],  # w1=0, wM=0 -> w0=1 (Degenerate, P=0)
        [1.0, 0.0],  # w1=1, wM=0 -> w0=0
        [0.0, 1.0],  # w1=0, wM=1 -> w0=0
        [0.3, 0.3],
        [0.5, 0.2],
        [0.2, 0.5],
        [0.1, 0.1],
        [0.4, 0.4],
    ]
    
    try:
        from scipy.optimize import minimize
    except ImportError:
        # Fallback if scipy is not available, though it is stated as available
        minimize = None

    for M in M_vals:
        if time.time() > deadline - 10:
            break
            
        support = np.array([0.0, 1.0, M])
        
        # Define objective function for optimizer
        # Variables: x = [w1, wM]
        def obj(x):
            w1, wM = x
            w0 = 1.0 - w1 - wM
            if w0 < -1e-12 or w1 < -1e-12 or wM < -1e-12:
                return 1.0  # Large penalty
            w = np.array([w0, w1, wM])
            if np.any(w < 0):
                return 1.0
            p = calc_prob(support, w)
            return -p  # Minimize negative probability
        
        best_local_score = -np.inf
        best_local_w = None
        
        for init in init_points:
            # Ensure init is valid
            w0_init = 1.0 - init[0] - init[1]
            if w0_init < -1e-9:
                continue
                
            if minimize:
                try:
                    res = minimize(obj, init, method='Nelder-Mead', 
                                   options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
                    if res.fun < 1.0:  # Check if valid
                        score = -res.fun
                        if score > best_local_score:
                            best_local_score = score
                            best_local_w = res.x
                except Exception:
                    pass
            else:
                # Simple grid fallback if no scipy
                pass

        # Also check the vertices and some grid points directly to ensure we don't miss global max
        # due to optimizer failing
        grid_w1 = np.linspace(0.0, 1.0, 20)
        grid_wM = np.linspace(0.0, 1.0, 20)
        for w1 in grid_w1:
            for wM in grid_wM:
                w0 = 1.0 - w1 - wM
                if w0 < -1e-9:
                    continue
                w = np.array([w0, w1, wM])
                p = calc_prob(support, w)
                if p > best_local_score:
                    best_local_score = p
                    best_local_w = np.array([w1, wM])

        if best_local_score > best_score:
            best_score = best_local_score
            w1_opt, wM_opt = best_local_w
            w0_opt = 1.0 - w1_opt - wM_opt
            # Clamp small negatives
            w0_opt = max(0.0, w0_opt)
            w1_opt = max(0.0, w1_opt)
            wM_opt = max(0.0, wM_opt)
            s_sum = w0_opt + w1_opt + wM_opt
            if s_sum > 1e-9:
                w0_opt /= s_sum
                w1_opt /= s_sum
                wM_opt /= s_sum
            else:
                w0_opt, w1_opt, wM_opt = 1/3, 1/3, 1/3
            
            best_support = support.copy()
            best_weights = np.array([w0_opt, w1_opt, wM_opt])

    # If no valid result found, use fallback
    if best_score <= -np.inf:
        best_support = np.array([0.0, 1.0, 10.0])
        best_weights = np.array([0.5, 0.3, 0.2])

    # Final validity check
    if not is_valid(best_support, best_weights):
        best_support = np.array([0.0, 1.0, 10.0])
        best_weights = np.array([0.5, 0.3, 0.2])

    support_out = np.asarray(best_support, dtype=np.float64).flatten()
    weights_out = np.asarray(best_weights, dtype=np.float64).flatten()
    
    return support_out, weights_out

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print(s)
    print(w)
