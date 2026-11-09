import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _calc_P(support, weights):
    """
    Calculate P[X1 + X2 + X3 < 2*X4] for discrete distribution.
    Uses vectorized numpy operations for speed.
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    n = len(s)
    if n == 0:
        return 0.0
        
    # Normalize weights
    w_sum = np.sum(w)
    if w_sum <= 1e-12:
        return 0.0
    w = w / w_sum
    
    # Precompute pairwise sums and weights
    # S_ij = s_i + s_j
    S = s[:, None] + s[None, :]
    # W_ij = w_i * w_j
    W = w[:, None] * w[None, :]
    
    # Threshold matrix T_kl = 2*s_l - s_k
    # We need sum over i,j such that S_ij < T_kl of W_ij
    # For small n, we can do this in a loop over k,l but vectorize the inner sum
    # Or use a 4D approach if n is small enough
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            mask = S < threshold
            P += w[k] * w[l] * np.sum(W[mask])
            
    return P

def _calc_P_fast(support, weights):
    """
    Faster calculation for small n using broadcasting.
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    n = len(s)
    if n == 0:
        return 0.0
        
    w_sum = np.sum(w)
    if w_sum <= 1e-12:
        return 0.0
    w = w / w_sum
    
    # Pairwise sums S[i,j] = s[i] + s[j]
    S = s[:, None] + s[None, :]
    # Pairwise weights W[i,j] = w[i] * w[j]
    W = w[:, None] * w[None, :]
    
    # Thresholds T[k,l] = 2*s[l] - s[k]
    # We want P = sum_{k,l} w[k]w[l] * sum_{i,j} W[i,j] * 1[S[i,j] < T[k,l]]
    
    # If n is small (<= 15), use 4D broadcasting
    if n <= 15:
        # S shape (n, n) -> (1, 1, n, n)
        # T shape (n, n) -> (n, n, 1, 1)
        # W shape (n, n) -> (1, 1, n, n)
        S_4d = S[None, None, :, :]
        T_4d = 2.0 * s[None, :] - s[:, None]  # shape (n, n)
        T_4d = T_4d[:, :, None, None]
        W_4d = W[None, None, :, :]
        
        mask = S_4d < T_4d  # shape (n, n, n, n)
        inner = np.sum(mask * W_4d, axis=(2, 3))  # shape (n, n)
        
        # P = sum_{k,l} w[k] w[l] inner[k,l]
        P = np.sum(w[:, None] * w[None, :] * inner)
        return P
    else:
        return _calc_P(s, w)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    best_score = -1.0
    best_support = None
    best_weights = None
    
    def update_best(s, w, score):
        nonlocal best_score, best_support, best_weights
        if score > best_score:
            best_score = score
            best_support = s.copy()
            best_weights = w.copy()
    
    # 1. Analytic 2-point baseline
    # For support {0, 1} with weights (1-p, p):
    # P = p * [ (1-p)^3 + 3*p*(1-p)^2 ] = p(1-p)^2(1+2p)
    # Maximize f(p) = p(1-p)^2(1+2p)
    # f'(p) = (1-p)(1-p-4p^2)
    # Roots: p=1 or 4p^2+p-1=0 -> p = (-1+sqrt(17))/8
    p_opt = (-1.0 + np.sqrt(17.0)) / 8.0
    s2 = np.array([0.0, 1.0])
    w2 = np.array([1.0 - p_opt, p_opt])
    score2 = _calc_P_fast(s2, w2)
    update_best(s2, w2, score2)
    
    # 2. Differential Evolution for 3-point {0, t, 1}
    def opt3(params):
        t, w1, w2 = params
        w0 = 1.0 - w1 - w2
        if w0 < -1e-9 or w1 < -1e-9 or w2 < -1e-9:
            return 1.0
        if t < -1e-9 or t > 1.0 + 1e-9:
            return 1.0
        s = np.array([0.0, t, 1.0])
        w = np.array([w0, w1, w2])
        return -_calc_P_fast(s, w)
    
    if time.time() < deadline:
        try:
            bounds3 = [(1e-6, 1.0), (1e-6, 1.0), (1e-6, 1.0)]
            res3 = differential_evolution(opt3, bounds3, maxiter=200, tol=1e-9,
                                           popsize=20, seed=42, polish=True,
                                           init='sobol')
            if -res3.fun > best_score:
                t, w1, w2 = res3.x
                w0 = 1.0 - w1 - w2
                if w0 > 0 and w1 > 0 and w2 > 0:
                    s3 = np.array([0.0, t, 1.0])
                    w3 = np.array([w0, w1, w2])
                    score3 = -res3.fun
                    update_best(s3, w3, score3)
        except Exception:
            pass
    
    # 3. Differential Evolution for 4-point {0, a, b, 1}
    def opt4(params):
        a, b, w1, w2, w3 = params
        w0 = 1.0 - w1 - w2 - w3
        if w0 < -1e-9 or w1 < -1e-9 or w2 < -1e-9 or w3 < -1e-9:
            return 1.0
        if a < -1e-9 or b < -1e-9 or a >= b - 1e-9 or b > 1.0 + 1e-9:
            return 1.0
        s = np.array([0.0, a, b, 1.0])
        w = np.array([w0, w1, w2, w3])
        return -_calc_P_fast(s, w)
    
    if time.time() < deadline:
        try:
            bounds4 = [(1e-6, 0.999), (0.001, 1.0), (1e-6, 1.0), (1e-6, 1.0), (1e-6, 1.0)]
            res4 = differential_evolution(opt4, bounds4, maxiter=200, tol=1e-9,
                                           popsize=20, seed=42, polish=True,
                                           init='sobol')
            if -res4.fun > best_score:
                a, b, w1, w2, w3 = res4.x
                w0 = 1.0 - w1 - w2 - w3
                if w0 > 0 and w1 > 0 and w2 > 0 and w3 > 0 and a < b:
                    s4 = np.array([0.0, a, b, 1.0])
                    w4 = np.array([w0, w1, w2, w3])
                    score4 = -res4.fun
                    update_best(s4, w4, score4)
        except Exception:
            pass
    
    # 4. Differential Evolution for 5-point {0, a, b, c, 1}
    def opt5(params):
        a, b, c, w1, w2, w3, w4 = params
        w0 = 1.0 - w1 - w2 - w3 - w4
        if w0 < -1e-9 or w1 < -1e-9 or w2 < -1e-9 or w3 < -1e-9 or w4 < -1e-9:
            return 1.0
        if a < -1e-9 or b < -1e-9 or c < -1e-9 or a >= b - 1e-9 or b >= c - 1e-9 or c > 1.0 + 1e-9:
            return 1.0
        s = np.array([0.0, a, b, c, 1.0])
        w = np.array([w0, w1, w2, w3, w4])
        return -_calc_P_fast(s, w)
    
    if time.time() < deadline:
        try:
            bounds5 = [(1e-6, 0.999), (0.001, 0.999), (0.002, 1.0),
                       (1e-6, 1.0), (1e-6, 1.0), (1e-6, 1.0), (1e-6, 1.0)]
            res5 = differential_evolution(opt5, bounds5, maxiter=150, tol=1e-9,
                                           popsize=15, seed=42, polish=True,
                                           init='sobol')
            if -res5.fun > best_score:
                a, b, c, w1, w2, w3, w4 = res5.x
                w0 = 1.0 - w1 - w2 - w3 - w4
                if w0 > 0 and w1 > 0 and w2 > 0 and w3 > 0 and w4 > 0 and a < b < c:
                    s5 = np.array([0.0, a, b, c, 1.0])
                    w5 = np.array([w0, w1, w2, w3, w4])
                    score5 = -res5.fun
                    update_best(s5, w5, score5)
        except Exception:
            pass
    
    # 5. Local refinement with Nelder-Mead on the best solution found so far
    if time.time() < deadline and best_support is not None:
        n_best = len(best_support)
        
        if n_best == 3:
            def refine3(params):
                t, w1, w2 = params
                w0 = 1.0 - w1 - w2
                if w0 < -1e-9 or w1 < -1e-9 or w2 < -1e-9 or t < -1e-9 or t > 1.0 + 1e-9:
                    return 1.0
                s = np.array([0.0, t, 1.0])
                w = np.array([w0, w1, w2])
                return -_calc_P_fast(s, w)
            
            # Initial guess from current best
            t_init = best_support[1]
            w1_init = best_weights[1]
            w2_init = best_weights[2]
            
            try:
                res_ref = minimize(refine3, [t_init, w1_init, w2_init],
                                   method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
                if -res_ref.fun > best_score:
                    t, w1, w2 = res_ref.x
                    w0 = 1.0 - w1 - w2
                    if w0 > 0 and w1 > 0 and w2 > 0:
                        s3 = np.array([0.0, t, 1.0])
                        w3 = np.array([w0, w1, w2])
                        score3 = -res_ref.fun
                        update_best(s3, w3, score3)
            except Exception:
                pass
                
        elif n_best == 4:
            def refine4(params):
                a, b, w1, w2, w3 = params
                w0 = 1.0 - w1 - w2 - w3
                if w0 < -1e-9 or w1 < -1e-9 or w2 < -1e-9 or w3 < -1e-9:
                    return 1.0
                if a < -1e-9 or b < -1e-9 or a >= b - 1e-9 or b > 1.0 + 1e-9:
                    return 1.0
                s = np.array([0.0, a, b, 1.0])
                w = np.array([w0, w1, w2, w3])
                return -_calc_P_fast(s, w)
            
            a_init = best_support[1]
            b_init = best_support[2]
            w1_init = best_weights[1]
            w2_init = best_weights[2]
            w3_init = best_weights[3]
            
            try:
                res_ref = minimize(refine4, [a_init, b_init, w1_init, w2_init, w3_init],
                                   method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
                if -res_ref.fun > best_score:
                    a, b, w1, w2, w3 = res_ref.x
                    w0 = 1.0 - w1 - w2 - w3
                    if w0 > 0 and w1 > 0 and w2 > 0 and w3 > 0 and a < b:
                        s4 = np.array([0.0, a, b, 1.0])
                        w4 = np.array([w0, w1, w2, w3])
                        score4 = -res_ref.fun
                        update_best(s4, w4, score4)
            except Exception:
                pass
                
        elif n_best == 5:
            def refine5(params):
                a, b, c, w1, w2, w3, w4 = params
                w0 = 1.0 - w1 - w2 - w3 - w4
                if w0 < -1e-9 or w1 < -1e-9 or w2 < -1e-9 or w3 < -1e-9 or w4 < -1e-9:
                    return 1.0
                if a < -1e-9 or b < -1e-9 or c < -1e-9 or a >= b - 1e-9 or b >= c - 1e-9 or c > 1.0 + 1e-9:
                    return 1.0
                s = np.array([0.0, a, b, c, 1.0])
                w = np.array([w0, w1, w2, w3, w4])
                return -_calc_P_fast(s, w)
            
            a_init = best_support[1]
            b_init = best_support[2]
            c_init = best_support[3]
            w1_init = best_weights[1]
            w2_init = best_weights[2]
            w3_init = best_weights[3]
            w4_init = best_weights[4]
            
            try:
                res_ref = minimize(refine5, [a_init, b_init, c_init, w1_init, w2_init, w3_init, w4_init],
                                   method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
                if -res_ref.fun > best_score:
                    a, b, c, w1, w2, w3, w4 = res_ref.x
                    w0 = 1.0 - w1 - w2 - w3 - w4
                    if w0 > 0 and w1 > 0 and w2 > 0 and w3 > 0 and w4 > 0 and a < b < c:
                        s5 = np.array([0.0, a, b, c, 1.0])
                        w5 = np.array([w0, w1, w2, w3, w4])
                        score5 = -res_ref.fun
                        update_best(s5, w5, score5)
            except Exception:
                pass
    
    # 6. Try a 3-point support with a large value {0, 1, M}
    # This can decouple the success conditions
    if time.time() < deadline:
        def opt3_large(params):
            M, w1, w2 = params
            w0 = 1.0 - w1 - w2
            if w0 < -1e-9 or w1 < -1e-9 or w2 < -1e-9:
                return 1.0
            if M < 1.0 - 1e-9 or M > 100.0:
                return 1.0
            s = np.array([0.0, 1.0, M])
            w = np.array([w0, w1, w2])
            return -_calc_P_fast(s, w)
        
        try:
            bounds3l = [(1.0, 50.0), (1e-6, 1.0), (1e-6, 1.0)]
            res3l = differential_evolution(opt3_large, bounds3l, maxiter=100, tol=1e-9,
                                            popsize=15, seed=42, polish=True,
                                            init='sobol')
            if -res3l.fun > best_score:
                M, w1, w2 = res3l.x
                w0 = 1.0 - w1 - w2
                if w0 > 0 and w1 > 0 and w2 > 0:
                    s3l = np.array([0.0, 1.0, M])
                    w3l = np.array([w0, w1, w2])
                    score3l = -res3l.fun
                    update_best(s3l, w3l, score3l)
        except Exception:
            pass
    
    # Ensure validity and return
    if best_support is None:
        # Fallback to 2-point analytic
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([1.0 - p_opt, p_opt])
    
    best_support = np.asarray(best_support, dtype=np.float64)
    best_weights = np.asarray(best_weights, dtype=np.float64)
    best_support = best_support.flatten()
    best_weights = best_weights.flatten()
    
    # Final validity check
    if len(best_support) != len(best_weights) or len(best_support) < 1:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([1.0 - p_opt, p_opt])
    
    if np.any(np.isnan(best_support)) or np.any(np.isinf(best_support)):
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([1.0 - p_opt, p_opt])
    
    if np.any(best_support < -1e-9):
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([1.0 - p_opt, p_opt])
    
    if np.any(np.isnan(best_weights)) or np.any(np.isinf(best_weights)):
        best_weights = np.array([1.0 - p_opt, p_opt])
    
    if np.any(best_weights < -1e-9):
        best_weights = np.array([1.0 - p_opt, p_opt])
    
    if np.sum(best_weights) <= 1e-9:
        best_weights = np.array([1.0 - p_opt, p_opt])
    
    return best_support, best_weights
