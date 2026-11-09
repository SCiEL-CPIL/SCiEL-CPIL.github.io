# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_P(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] exactly.
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    # Check validity
    if len(s) != len(w):
        return -np.inf
    if np.any(s < -1e-9) or np.any(w < -1e-9):
        return -np.inf
    if np.any(np.isnan(s)) or np.any(np.isinf(s)) or np.any(np.isnan(w)) or np.any(np.isinf(w)):
        return -np.inf
        
    total = w.sum()
    if total <= 1e-9:
        return -np.inf
        
    p = w / total
    n = len(s)
    
    # Compute all pair sums and their probabilities
    # pair_sums[i, j] = s[i] + s[j]
    # pair_probs[i, j] = p[i] * p[j]
    pair_sums = s[:, None] + s[None, :]
    pair_probs = p[:, None] * p[None, :]
    
    # Flatten and sort by pair sum
    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()
    
    order = np.argsort(flat_sums, kind='mergesort')
    sorted_sums = flat_sums[order]
    sorted_probs = flat_probs[order]
    
    # Cumulative probabilities
    # cum_probs[i] is sum of sorted_probs[0..i]
    # We need sum of probs where sum < threshold
    # searchsorted(sorted_sums, threshold, side='left') gives index of first >= threshold
    # So sum is cum_probs[idx-1] if idx > 0 else 0
    cum_probs = np.cumsum(sorted_probs)
    
    # Thresholds for (k, l) where k is index for X3, l is index for X4
    # threshold = 2*s[l] - s[k]
    # We need F(threshold) for each (k, l)
    # Vectorized approach:
    # thresholds is shape (n, n)
    thresholds = 2.0 * s[None, :] - s[:, None] # thresholds[k, l] = 2*s[l] - s[k]
    
    # Find indices for all thresholds at once
    # searchsorted works on 1D array, but we can flatten thresholds
    flat_thresholds = thresholds.flatten()
    indices = np.searchsorted(sorted_sums, flat_thresholds, side='left')
    
    # Get cumulative probabilities
    # If index is 0, sum is 0. Else sum is cum_probs[index-1]
    # Use np.where or masking
    sums_arr = np.zeros_like(flat_thresholds)
    mask = indices > 0
    sums_arr[mask] = cum_probs[indices[mask] - 1]
    
    # Reshape back to (n, n)
    F_values = sums_arr.reshape(n, n) # F_values[k, l] = P[X1+X2 < 2*s[l]-s[k]]
    
    # Weight matrix for X3, X4
    weight_matrix = p[:, None] * p[None, :] # weight_matrix[k, l] = p[k]*p[l]
    
    P = np.sum(weight_matrix * F_values)
    return P

def _objective_3point(params):
    """
    Objective function for 3-point distribution.
    params: [p0, p1, p2, a, b]
    support: [0, a, b]
    weights: [p0, p1, p2]
    """
    p0, p1, p2, a, b = params
    
    # Basic validity checks
    if p0 < 0 or p1 < 0 or p2 < 0:
        return 1e10
    if a < 0 or b < 0:
        return 1e10
    if a > b:
        return 1e10
        
    # Normalize weights to sum to 1
    total_p = p0 + p1 + p2
    if total_p <= 1e-9:
        return 1e10
        
    support = np.array([0.0, a, b], dtype=np.float64)
    weights = np.array([p0, p1, p2], dtype=np.float64)
    
    P = _compute_P(support, weights)
    if P == -np.inf:
        return 1e10
        
    return -P

def _objective_4point(params):
    """
    Objective function for 4-point distribution.
    params: [p0, p1, p2, p3, a, b, c]
    support: [0, a, b, c]
    """
    p0, p1, p2, p3, a, b, c = params
    
    if any(x < 0 for x in [p0, p1, p2, p3, a, b, c]):
        return 1e10
    if a > b or b > c:
        return 1e10
        
    total_p = p0 + p1 + p2 + p3
    if total_p <= 1e-9:
        return 1e10
        
    support = np.array([0.0, a, b, c], dtype=np.float64)
    weights = np.array([p0, p1, p2, p3], dtype=np.float64)
    
    P = _compute_P(support, weights)
    if P == -np.inf:
        return 1e10
        
    return -P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 580.0 # Safety margin
    
    best_P = -np.inf
    best_support = None
    best_weights = None
    
    # 1. Analytic 2-point optimum
    # P(p) = (1-p)p^2(3-2p)
    # Optimal p approx 0.578
    p_opt_2 = (15 - np.sqrt(33)) / 16
    support_2 = np.array([0.0, 1.0], dtype=np.float64)
    weights_2 = np.array([p_opt_2, 1 - p_opt_2], dtype=np.float64)
    P_2 = _compute_P(support_2, weights_2)
    
    if P_2 > best_P:
        best_P = P_2
        best_support = support_2.copy()
        best_weights = weights_2.copy()
        
    # 2. Optimize 3-point distribution
    # Parameters: [p0, p1, p2, a, b]
    # Initial guesses
    guesses_3 = [
        # Near 2-point optimum
        [p_opt_2 * 0.9, p_opt_2 * 0.1, 1 - p_opt_2, 0.5, 1.0],
        [p_opt_2 * 0.8, p_opt_2 * 0.2, 1 - p_opt_2, 0.2, 1.0],
        [p_opt_2 * 0.7, p_opt_2 * 0.3, 1 - p_opt_2, 0.5, 2.0],
        # Spread out
        [0.5, 0.3, 0.2, 1.0, 3.0],
        [0.4, 0.4, 0.2, 1.0, 5.0],
        [0.3, 0.3, 0.4, 1.0, 10.0],
        # Small a
        [0.6, 0.2, 0.2, 0.1, 1.0],
        # Large b
        [0.5, 0.3, 0.2, 1.0, 100.0],
    ]
    
    for guess in guesses_3:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _objective_3point, 
                guess, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            P_val = -res.fun
            if P_val > best_P:
                best_P = P_val
                p0, p1, p2, a, b = res.x
                # Ensure non-negative
                p0 = max(0, p0); p1 = max(0, p1); p2 = max(0, p2)
                a = max(0, a); b = max(0, b)
                if a > b: a, b = b, a
                support = np.array([0.0, a, b], dtype=np.float64)
                weights = np.array([p0, p1, p2], dtype=np.float64)
                # Re-evaluate to ensure validity and exact P
                P_check = _compute_P(support, weights)
                if P_check != -np.inf and P_check > best_P:
                    best_P = P_check
                    best_support = support.copy()
                    best_weights = weights.copy()
        except Exception:
            pass

    # 3. Optimize 4-point distribution
    # Parameters: [p0, p1, p2, p3, a, b, c]
    # Start from best 3-point if available, or 2-point
    if best_support is not None and len(best_support) == 3:
        s0, s1, s2 = best_support
        w0, w1, w2 = best_weights
        guesses_4 = [
            [w0, w1*0.5, w1*0.5, w2, s0, s1, s2], # Split middle weight
            [w0*0.5, w0*0.5, w1, w2, 0.0, s1, s2], # Split first weight (but s0=0 fixed) -> [0, s1, s2] with split p0? No, support is fixed 0.
            # Better: Add a new point
            [w0*0.9, w0*0.1, w1, w2, 0.0, s1, s2*2.0],
            [w0, w1*0.8, w1*0.2, w2, s0, s1*0.5, s2],
            [w0, w1, w2*0.8, w2*0.2, s0, s1, s2*1.5],
        ]
    else:
        guesses_4 = [
            [0.4, 0.3, 0.2, 0.1, 0.5, 1.0, 2.0],
            [0.3, 0.3, 0.25, 0.15, 0.5, 1.0, 3.0],
            [0.4, 0.3, 0.2, 0.1, 1.0, 2.0, 5.0],
            [0.3, 0.3, 0.2, 0.2, 1.0, 3.0, 10.0],
        ]
        
    for guess in guesses_4:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _objective_4point, 
                guess, 
                method='Nelder-Mead', 
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            P_val = -res.fun
            if P_val > best_P:
                best_P = P_val
                p0, p1, p2, p3, a, b, c = res.x
                p0 = max(0, p0); p1 = max(0, p1); p2 = max(0, p2); p3 = max(0, p3)
                a = max(0, a); b = max(0, b); c = max(0, c)
                # Sort support and corresponding weights
                support = np.array([0.0, a, b, c], dtype=np.float64)
                weights = np.array([p0, p1, p2, p3], dtype=np.float64)
                
                # Ensure sorted support and consistent weights
                order = np.argsort(support)
                support = support[order]
                weights = weights[order]
                
                P_check = _compute_P(support, weights)
                if P_check != -np.inf and P_check > best_P:
                    best_P = P_check
                    best_support = support.copy()
                    best_weights = weights.copy()
        except Exception:
            pass

    # Fallback if no valid solution found
    if best_support is None:
        return np.array([0.0, 1.0], dtype=np.float64), np.array([p_opt_2, 1-p_opt_2], dtype=np.float64)

    # Final validity check
    if len(best_support) != len(best_weights):
        return np.array([0.0, 1.0], dtype=np.float64), np.array([p_opt_2, 1-p_opt_2], dtype=np.float64)
    if np.any(best_support < -1e-9) or np.any(best_weights < -1e-9) or np.sum(best_weights) <= 1e-9:
        return np.array([0.0, 1.0], dtype=np.float64), np.array([p_opt_2, 1-p_opt_2], dtype=np.float64)
    if np.any(np.isnan(best_support)) or np.any(np.isinf(best_support)) or np.any(np.isnan(best_weights)) or np.any(np.isinf(best_weights)):
        return np.array([0.0, 1.0], dtype=np.float64), np.array([p_opt_2, 1-p_opt_2], dtype=np.float64)

    return best_support, best_weights

# EVOLVE-BLOCK-END
