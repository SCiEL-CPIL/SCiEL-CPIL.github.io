import numpy as np
from scipy.optimize import differential_evolution
import time

def compute_P(support, weights):
    n = len(support)
    if n == 0:
        return 0.0
    
    # Normalize weights
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    
    # Precompute pairwise sums s_i + s_j
    sums_ij = support[:, None] + support[None, :]
    
    total = 0.0
    # Loop over k (X3) and l (X4)
    for k in range(n):
        for l in range(n):
            T = 2 * support[l] - support[k]
            # We need sum_{i,j} w_i w_j * 1[s_i + s_j < T]
            # mask[i,j] is 1 if sums_ij[i,j] < T
            mask = sums_ij < T
            # val = sum_i sum_j w_i w_j mask[i,j]
            # This can be computed as w @ (w * mask).sum(axis=0) or similar
            # Let M be mask. We want 1^T (w_i w_j M_ij) 1 = sum_i w_i (sum_j w_j M_ij)
            # Let v_j = sum_i w_i M_ij? No.
            # sum_i sum_j w_i w_j M_ij
            # Let A_ij = w_i w_j M_ij. Sum is A.sum().
            # Vectorized: (w[:, None] * w[None, :] * mask).sum()
            # This creates an n x n array each time.
            # Alternative:
            # For each i, sum_j w_j M_ij is a vector v_i.
            # Then sum_i w_i v_i.
            # v = (mask * w[None, :]).sum(axis=1)
            # val = (w * v).sum()
            
            # Let's use the broadcasting way, it's concise and likely fast for small n
            # But allocating n*n arrays in a loop of n*n is expensive.
            # Let's stick to the previous method but optimized:
            # mask is boolean n x n.
            # val = np.sum(w[:, None] * w[None, :] * mask)
            # This is 1 multiplication of n x n arrays and a sum.
            
            val = np.sum(w[:, None] * w[None, :] * mask)
            total += w[k] * w[l] * val
            
    return total

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    deadline = time.time() + 600
    best_score = 0.0
    best_support = None
    best_weights = None
    
    # We will try different n. DE is global, so it's robust.
    # Dimensions: n-1 support values (s_0..s_{n-2}, s_{n-1}=1 fixed) + n weight logits.
    # Total dim = 2n - 1.
    
    for n in range(3, 11):
        if time.time() > deadline - 120:
            break
            
        def objective(vars):
            if time.time() > deadline - 10:
                return 1e10 # Penalty for timeout
            
            # Unpack
            s = np.zeros(n)
            s[0:n-1] = vars[0:n-1]
            s[n-1] = 1.0
            
            # Ensure s is sorted? The problem doesn't require sorted support,
            # but the indices in weights must match.
            # If we don't sort, the optimizer might find a permutation that is equivalent
            # but with different indices. This is fine as long as weights match.
            # However, DE explores the space of (s, w). If s is unsorted, it's just a different point in the space.
            # To avoid duplicate evaluations of the same measure (permutation symmetry),
            # we could sort s and w together. But this introduces non-smoothness (sorting).
            # For DE, non-smoothness is less of an issue than for gradient methods.
            # However, sorting might help convergence by reducing the effective dimensionality of the search space (symmetry breaking).
            # Let's try sorting s and w together in the objective.
            
            # Sort indices
            idx = np.argsort(s)
            s_sorted = s[idx]
            w_logits = vars[n-1:2*n-1]
            w_logits_sorted = w_logits[idx]
            
            # Softmax for weights
            w_max = np.max(w_logits_sorted)
            w = np.exp(w_logits_sorted - w_max)
            w_sum = np.sum(w)
            if w_sum == 0:
                return 1e10
            w = w / w_sum
            
            # Ensure s >= 0
            s_sorted = np.maximum(s_sorted, 0.0)
            
            # Check validity
            if np.any(np.isnan(s_sorted)) or np.any(np.isnan(w)):
                return 1e10
            
            P = compute_P(s_sorted, w)
            return -P
        
        # Bounds for DE
        # s_i in [0, 1] for i < n-1
        # w_logits in [-10, 10] (arbitrary wide range)
        bounds = [(0, 1)] * (n-1) + [(-10, 10)] * n
        
        # Initial population size and max iterations
        # DE scales as population_size * maxiter * dim
        # Let's be conservative with time
        maxiter = 500
        popsize = 15
        
        try:
            res = differential_evolution(
                objective, 
                bounds, 
                maxiter=maxiter, 
                popsize=popsize,
                tol=1e-5,
                seed=42,
                init='sobol' # Better initialization
            )
            
            if res.fun < 1e9:
                P_val = -res.fun
                if P_val > best_score:
                    best_score = P_val
                    
                    # Extract best solution
                    s = np.zeros(n)
                    s[0:n-1] = res.x[0:n-1]
                    s[n-1] = 1.0
                    w_logits = res.x[n-1:2*n-1]
                    
                    # Apply same sorting as in objective
                    idx = np.argsort(s)
                    s_sorted = s[idx]
                    w_logits_sorted = w_logits[idx]
                    
                    w_max = np.max(w_logits_sorted)
                    w = np.exp(w_logits_sorted - w_max)
                    w = w / np.sum(w)
                    s_sorted = np.maximum(s_sorted, 0.0)
                    
                    best_support = s_sorted
                    best_weights = w
        except Exception as e:
            pass # Continue to next n if DE fails

    if best_support is None:
        # Fallback
        n = 4
        support = np.array([0.0, 0.5, 1.0, 1.0])
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        return support, weights
        
    return best_support, best_weights
