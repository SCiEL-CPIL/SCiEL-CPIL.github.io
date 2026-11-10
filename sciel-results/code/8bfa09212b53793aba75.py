# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights (need not be normalized;
            the evaluator normalizes them to sum to 1).
    """
    start_time = time.time()
    deadline = start_time + 550.0  # Safety margin
    
    # 1. Define Support
    # Use a log-spaced support to allow for large ratios between values.
    # Include 0? No, logspace doesn't include 0. 
    # The condition is strict inequality. 0 is a valid support point.
    # Let's mix: some small integers and log-spaced large values.
    n_small = 10
    n_large = 90
    n = n_small + n_large
    
    s_small = np.arange(n_small, dtype=np.float64) # 0, 1, ..., 9
    # Log spaced from 10 to 10000
    s_large = np.logspace(np.log10(10.0), np.log10(10000.0), n_large, dtype=np.float64)
    support = np.concatenate([s_small, s_large])
    support.sort()
    n = len(support)
    
    # 2. Precompute Pairwise Sums and Sorting Indices
    # S_ij = support[i] + support[j]
    S = support[:, None] + support[None, :]
    S_flat = S.flatten()
    
    # Get indices to sort S_flat
    sort_idx = np.argsort(S_flat)
    S_sorted = S_flat[sort_idx]
    
    # 3. Objective Function
    # We optimize over z in R^n, w = softmax(z)
    # This ensures w > 0 and sum(w) = 1.
    
    best_z = np.zeros(n)
    best_score = -np.inf
    best_w = np.ones(n) / n
    
    def objective(z):
        nonlocal best_z, best_score, best_w
        
        # Check deadline
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded")
            
        # Softmax
        z_max = np.max(z)
        exp_z = np.exp(z - z_max)
        w = exp_z / np.sum(exp_z)
        
        # Compute P
        # W_ij = w_i * w_j
        W = np.outer(w, w)
        W_flat = W.flatten()
        
        # Map weights to sorted order
        W_sorted = W_flat[sort_idx]
        
        # Prefix sums of weights
        PrefixW = np.cumsum(W_sorted)
        
        # Thresholds T_kl = 2*support[l] - support[k]
        # We need P[X1+X2 < T_kl] for each k,l
        # T is n x n
        T = 2.0 * support[None, :] - support[:, None]
        T_flat = T.flatten()
        
        # Count of S_ij < T_kl
        # np.searchsorted returns index where T_flat would be inserted to maintain order
        # side='left' means we count elements < T_flat
        counts = np.searchsorted(S_sorted, T_flat, side='left')
        
        # Get sum of weights for those counts
        # If count is 0, sum is 0.
        # If count is c, sum is PrefixW[c-1].
        # Use take with mode='clip' to handle 0 index safely? 
        # np.take(PrefixW, counts-1, mode='clip') 
        # If counts=0, index=-1, clip to 0? No, clip to 0 gives PrefixW[0] which is wrong.
        # We need to handle 0 explicitly or use a padded array.
        PaddedPrefixW = np.concatenate([[0.0], PrefixW])
        P_flat = PaddedPrefixW[counts]
        
        # P_kl = P_flat
        P_matrix = P_flat.reshape(n, n)
        
        # Total P = sum_{k,l} w_k * w_l * P_kl
        # Note: w_k w_l is W[k,l]
        P_val = np.sum(W * P_matrix)
        
        # Update best
        if P_val > best_score:
            best_score = P_val
            best_z = z.copy()
            best_w = w.copy()
            
        return -P_val
    
    # 4. Optimization
    # Use Nelder-Mead as it doesn't require gradients and is robust for this dimension.
    # maxiter set high, but deadline will stop it.
    try:
        res = scipy.optimize.minimize(
            objective, 
            np.zeros(n), 
            method='Nelder-Mead', 
            options={
                'maxiter': 10000, 
                'xatol': 1e-4, 
                'fatol': 1e-6
            }
        )
    except TimeoutError:
        pass
    except Exception as e:
        # If optimization fails, return initial
        pass
    
    # If no improvement was made, return initial
    if best_score == -np.inf:
        best_w = np.ones(n) / n
        best_z = np.zeros(n)
        
    # Ensure weights are valid
    # best_w is already normalized
    # support is non-negative
    
    return support, best_w

# EVOLVE-BLOCK-END
