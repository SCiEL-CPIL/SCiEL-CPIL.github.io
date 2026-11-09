# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

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
    pair_sums = s[:, None] + s[None, :]
    pair_probs = p[:, None] * p[None, :]
    
    # Flatten and sort by pair sum
    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()
    
    order = np.argsort(flat_sums, kind='mergesort')
    sorted_sums = flat_sums[order]
    sorted_probs = flat_probs[order]
    
    # Cumulative probabilities
    cum_probs = np.cumsum(sorted_probs)
    
    # Thresholds for (k, l) where k is index for X3, l is index for X4
    # threshold = 2*s[l] - s[k]
    # Vectorized approach:
    thresholds = 2.0 * s[None, :] - s[:, None] # thresholds[k, l] = 2*s[l] - s[k]
    
    # Find indices for all thresholds at once
    flat_thresholds = thresholds.flatten()
    indices = np.searchsorted(sorted_sums, flat_thresholds, side='left')
    
    # Get cumulative probabilities
    sums_arr = np.zeros_like(flat_thresholds)
    mask = indices > 0
    sums_arr[mask] = cum_probs[indices[mask] - 1]
    
    # Reshape back to (n, n)
    F_values = sums_arr.reshape(n, n) # F_values[k, l] = P[X1+X2 < 2*s[l]-s[k]]
    
    # Weight matrix for X3, X4
    weight_matrix = p[:, None] * p[None, :] # weight_matrix[k, l] = p[k]*p[l]
    
    P = np.sum(weight_matrix * F_values)
    return P

def _two_point_P(p):
    """
    Analytic P for two-point measure {0, a} with prob p at 0.
    P = (1-p) * p^2 * (3 - 2p)
    """
    q = 1.0 - p
    return q * p * p * (3.0 - 2.0 * p)

def _objective_3point(params):
    """
    Objective function for 3-point distribution.
    params: [p0, p1, p2, a, b]
    support: [0, a, b]
    weights: [p0, p1, p2]
    """
    p0, p1, p2, a, b = params
    
    # Basic validity checks
    if p0 < -1e-9 or p1 < -1e-9 or p2 < -1e-9:
        return 1e10
    if a < -1e-9 or b < -1e-9:
        return 1e10
        
    # Clamp to valid range
    p0 = max(0.0, p0)
    p1 = max(0.0, p1)
    p2 = max(0.0, p2)
    a = max(0.0, a)
    b = max(0.0, b)
    
    if a > b:
        a, b = b, a
        
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
    
    if any(x < -1e-9 for x in [p0, p1, p2, p3, a, b, c]):
        return 1e10
        
    p0 = max(0.0, p0)
    p1 = max(0.0, p1)
    p2 = max(0.0, p2)
    p3 = max(0.0, p3)
    a = max(0.0, a)
    b = max(0.0, b)
    c = max(0.0, c)
    
    # Ensure sorted support
    support_vals = [0.0, a, b, c]
    support_vals.sort()
    # Note: reordering support changes which weight corresponds to which support point.
    # The parameterization assumes support is [0, a, b, c] with weights [p0, p1, p2, p3].
    # If we sort support, we must also sort weights accordingly.
    # However, the objective function takes params and constructs support/weights.
    # If a > b, the support array [0, a, b, c] is not sorted, but _compute_P doesn't require sorted support.
    # It just computes pairwise sums. So we don't strictly need to sort, but it helps with numerical stability or logic?
    # Actually, _compute_P is invariant to permutation of (s, w) pairs.
    # So we can just pass them as is, provided they are non-negative.
    
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
    # Optimize p in (0, 1)
    res_2p = minimize_scalar(lambda p: -_two_point_P(p), bounds=(1e-6, 1.0 - 1e-6), method='bounded')
    p_opt_2 = res_2p.x
    support_2 = np.array([0.0, 1.0], dtype=np.float64)
    weights_2 = np.array([p_opt_2, 1 - p_opt_2], dtype=np.float64)
    P_2 = _compute_P(support_2, weights_2)
    
    if P_2 > best_P:
        best_P = P_2
        best_support = support_2.copy()
        best_weights = weights_2.copy()
        
    # 2. Optimize 3-point distribution
    # Parameters: [p0, p1, p2, a, b]
    # We use multiple initial guesses to cover the landscape.
    # Key insight: The 2-point optimum has mass at 0 and 1.
    # A 3-point distribution might benefit from a point between 0 and 1, or > 1.
    
    guesses_3 = [
        # Near 2-point optimum, split mass
        [p_opt_2 * 0.9, p_opt_2 * 0.1, 1 - p_opt_2, 0.5, 1.0],
        [p_opt_2 * 0.5, p_opt_2 * 0.5, 1 - p_opt_2, 0.5, 1.0],
        [p_opt_2, 0.1, 0.9 - p_opt_2, 0.5, 1.0],
        # Spread out
        [0.5, 0.3, 0.2, 1.0, 3.0],
        [0.4, 0.4, 0.2, 1.0, 5.0],
        [0.3, 0.3, 0.4, 1.0, 10.0],
        # Small a
        [0.6, 0.2, 0.2, 0.1, 1.0],
        # Large b
        [0.5, 0.3, 0.2, 1.0, 100.0],
        # Symmetric-ish
        [0.4, 0.2, 0.4, 1.0, 2.0],
        [0.5, 0.1, 0.4, 0.5, 2.0],
    ]
    
    for guess in guesses_3:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _objective_3point, 
                guess, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            P_val = -res.fun
            if P_val > best_P:
                p0, p1, p2, a, b = res.x
                p0 = max(0, p0); p1 = max(0, p1); p2 = max(0, p2)
                a = max(0, a); b = max(0, b)
                if a > b: a, b = b, a
                support = np.array([0.0, a, b], dtype=np.float64)
                weights = np.array([p0, p1, p2], dtype=np.float64)
                P_check = _compute_P(support, weights)
                if P_check != -np.inf and P_check > best_P:
                    best_P = P_check
                    best_support = support.copy()
                    best_weights = weights.copy()
        except Exception:
            pass

    # 3. Optimize 4-point distribution
    # Parameters: [p0, p1, p2, p3, a, b, c]
    
    guesses_4 = []
    
    # Generate guesses based on current best
    if best_support is not None:
        s_best = best_support
        w_best = best_weights
        # Extend best solution
        if len(s_best) == 2:
            guesses_4.append([w_best[0]*0.8, w_best[0]*0.2, w_best[1]*0.5, w_best[1]*0.5, 0.5, s_best[1], s_best[1]*2])
            guesses_4.append([w_best[0]*0.5, w_best[0]*0.5, w_best[1]*0.5, w_best[1]*0.5, s_best[1]*0.5, s_best[1], s_best[1]*1.5])
        elif len(s_best) == 3:
            # Split one of the weights
            for i in range(3):
                w_split = list(w_best)
                w_new = list(w_best)
                w_new[i] *= 0.5
                w_new.append(w_best[i] * 0.5)
                s_new = list(s_best)
                # Add a new support point near the existing one or in a gap
                if i == 0:
                    s_new.append(s_best[1] * 0.5)
                elif i == 1:
                    s_new.append(s_best[2] * 0.5)
                else:
                    s_new.append(s_best[2] * 1.5)
                guesses_4.append(w_new + s_new)
    
    # Add some generic guesses
    guesses_4.extend([
        [0.4, 0.3, 0.2, 0.1, 0.5, 1.0, 2.0],
        [0.3, 0.3, 0.25, 0.15, 0.5, 1.0, 3.0],
        [0.4, 0.3, 0.2, 0.1, 1.0, 2.0, 5.0],
        [0.3, 0.3, 0.2, 0.2, 1.0, 3.0, 10.0],
        [0.5, 0.2, 0.2, 0.1, 0.5, 1.0, 2.0],
        [0.2, 0.3, 0.3, 0.2, 1.0, 2.0, 4.0],
    ])
    
    for guess in guesses_4:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _objective_4point, 
                guess, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            P_val = -res.fun
            if P_val > best_P:
                p0, p1, p2, p3, a, b, c = res.x
                p0 = max(0, p0); p1 = max(0, p1); p2 = max(0, p2); p3 = max(0, p3)
                a = max(0, a); b = max(0, b); c = max(0, c)
                
                support = np.array([0.0, a, b, c], dtype=np.float64)
                weights = np.array([p0, p1, p2, p3], dtype=np.float64)
                
                # Note: _compute_P handles unsorted support, but let's sort for consistency in output
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
