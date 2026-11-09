import numpy as np
import time
from scipy.optimize import minimize

def _evaluate_measure(support, weights):
    """
    Compute P = Prob[X1 + X2 + X3 < 2*X4] exactly in float64.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    p = weights / w_sum
    
    # Ensure support is sorted for consistency, though not strictly required for the sum
    # But sorting helps if we were to do searchsorted, but here we do explicit sum
    # The formula: P = sum_k sum_l w_k w_l * (sum_{i,j: s_i+s_j < 2s_l-s_k} w_i w_j)
    
    # For small n, direct double loop is fine and avoids array overhead
    # n^2 * n^2 = n^4 operations. For n=10, 10000 ops. Trivial.
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * support[l] - support[k]
            inner_sum = 0.0
            for i in range(n):
                for j in range(n):
                    if support[i] + support[j] < threshold:
                        inner_sum += p[i] * p[j]
            P += p[k] * p[l] * inner_sum
    return float(P)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4].
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_score = -1.0
    best_support = None
    best_weights = None
    
    def update_best(sup, wts, score):
        nonlocal best_score, best_support, best_weights
        if score > best_score:
            best_score = score
            best_support = sup.copy()
            best_weights = wts.copy()
            
    def time_left():
        return deadline - time.time()

    # Strategy: Optimize reduced parameter spaces for specific support structures
    
    # Structure 1: Support [0, a, 1]
    # Params: [a, w_a, w_1]
    # w_0 = 1 - w_a - w_1
    def obj_3pt(params):
        a = params[0]
        w_a = params[1]
        w_1 = params[2]
        w_0 = 1.0 - w_a - w_1
        
        # Constraints: 0 <= a <= 1, w_a >= 0, w_1 >= 0, w_0 >= 0
        if a < 0 or a > 1 or w_a < 0 or w_1 < 0 or w_0 < 0:
            return 1.0 # Penalty
        
        support = np.array([0.0, a, 1.0])
        weights = np.array([w_0, w_a, w_1])
        
        # Check for duplicate support points (degenerate)
        # If a == 0 or a == 1, it's effectively 2-point, which is fine
        # But _evaluate_measure handles duplicates correctly
        
        try:
            P = _evaluate_measure(support, weights)
            return -P
        except Exception:
            return 1.0

    # Strategy 2: Support [0, a, b, 1]
    # Params: [a, b, w_a, w_b, w_1]
    # w_0 = 1 - w_a - w_b - w_1
    def obj_4pt(params):
        a = params[0]
        b = params[1]
        w_a = params[2]
        w_b = params[3]
        w_1 = params[4]
        w_0 = 1.0 - w_a - w_b - w_1
        
        if a < 0 or a > 1 or b < 0 or b > 1:
            return 1.0
        if w_a < 0 or w_b < 0 or w_1 < 0 or w_0 < 0:
            return 1.0
            
        support = np.array([0.0, a, b, 1.0])
        weights = np.array([w_0, w_a, w_b, w_1])
        
        try:
            P = _evaluate_measure(support, weights)
            return -P
        except Exception:
            return 1.0

    # Run optimization for 3-point structure
    np.random.seed(42)
    
    # Initializations for 3-point
    inits_3 = [
        [0.5, 0.2, 0.2], # Random
        [0.4, 0.3, 0.3],
        [0.6, 0.1, 0.5],
        [0.2, 0.5, 0.1],
        [0.8, 0.2, 0.2],
        [0.0, 0.5, 0.5], # Degenerate 2-point
        [1.0, 0.5, 0.5], # Degenerate 2-point
    ]
    
    for x0 in inits_3:
        if time_left() < 30:
            break
        try:
            res = minimize(obj_3pt, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < 0: # Valid solution
                a, w_a, w_1 = res.x
                w_0 = 1.0 - w_a - w_1
                if w_0 >= 0:
                    support = np.array([0.0, a, 1.0])
                    weights = np.array([w_0, w_a, w_1])
                    score = _evaluate_measure(support, weights)
                    update_best(support, weights, score)
        except Exception:
            pass

    # Run optimization for 4-point structure
    inits_4 = [
        [0.3, 0.7, 0.1, 0.1, 0.3],
        [0.2, 0.8, 0.2, 0.2, 0.2],
        [0.5, 0.5, 0.1, 0.1, 0.4],
        [0.1, 0.9, 0.3, 0.1, 0.2],
        [0.4, 0.6, 0.2, 0.3, 0.1],
    ]
    
    for x0 in inits_4:
        if time_left() < 30:
            break
        try:
            res = minimize(obj_4pt, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < 0:
                a, b, w_a, w_b, w_1 = res.x
                w_0 = 1.0 - w_a - w_b - w_1
                if w_0 >= 0:
                    support = np.array([0.0, a, b, 1.0])
                    weights = np.array([w_0, w_a, w_b, w_1])
                    score = _evaluate_measure(support, weights)
                    update_best(support, weights, score)
        except Exception:
            pass

    # Fallback: The analytical 2-point solution
    # p = (1 + sqrt(33)) / 16
    p_opt = (1.0 + np.sqrt(33.0)) / 16.0
    support_2 = np.array([0.0, 1.0])
    weights_2 = np.array([1.0 - p_opt, p_opt])
    score_2 = _evaluate_measure(support_2, weights_2)
    update_best(support_2, weights_2, score_2)

    # Final check
    if best_support is None:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])

    return best_support, best_weights
