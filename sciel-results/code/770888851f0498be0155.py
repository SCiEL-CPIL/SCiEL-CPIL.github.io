import numpy as np
import time
from scipy.optimize import minimize

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: Co-optimize support points and weights for a 3-point measure {a, b, c}.
    Unlike fixed support strategies, this allows the geometry of the support to adapt 
    to maximize the inequality probability.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    def calc_prob_exact(support, weights):
        """
        Calculate P[X1 + X2 + X3 < 2*X4] for a 3-point support.
        support = [s0, s1, s2]
        weights = [w0, w1, w2]
        """
        s0, s1, s2 = support
        w0, w1, w2 = weights
        
        # Normalize weights to ensure they sum to 1 for probability calculation
        w_sum = w0 + w1 + w2
        if w_sum <= 1e-12:
            return 0.0
        p0, p1, p2 = w0/w_sum, w1/w_sum, w2/w_sum
        
        # P = sum_{k,l} p_k p_l * (sum_{i,j} p_i p_j * I(s_i + s_j < 2*s_l - s_k))
        # Since n=3, we can enumerate the 9 (k,l) pairs and for each, the 9 (i,j) pairs.
        # Total 81 checks. This is very fast.
        
        p = 0.0
        for k in range(3):
            for l in range(3):
                threshold = 2 * support[l] - support[k]
                # Inner sum: sum_{i,j} p_i p_j I(s_i + s_j < threshold)
                inner_sum = 0.0
                for i in range(3):
                    for j in range(3):
                        if support[i] + support[j] < threshold:
                            inner_sum += p[i] * p[j]
                p += p[k] * p[l] * inner_sum
        return p

    def objective(params):
        """
        Params: [log_a, log_b, log_c, log_w0, log_w1, log_w2]
        We use log-transforms to ensure positivity and handle scale invariance roughly.
        Actually, let's just use raw values with bounds in minimize, or map from unconstrained.
        Let's map unconstrained u -> [a, b, c, w0, w1, w2]
        a = exp(u0), b = exp(u1), c = exp(u2)
        w0 = exp(u3), w1 = exp(u4), w2 = exp(u5)
        
        Constraint: a < b < c. We can enforce this by ordering or just let the optimizer find a local min where they are ordered.
        However, the function is symmetric under permutation of points? No, the weights are tied to specific points.
        But the set {s_i} is unordered in the definition of mu.
        So we can assume WLOG a < b < c.
        To ensure a < b < c, we can define:
        a = exp(u0)
        b = a * exp(u1)  => b > a
        c = b * exp(u2)  => c > b
        
        Weights:
        w0 = exp(u3)
        w1 = exp(u4)
        w2 = exp(u5)
        
        We need to return -P because minimize finds minima.
        """
        if time.time() > deadline:
            return 1e10 # Penalize timeout
        
        a = np.exp(params[0])
        b = a * np.exp(params[1])
        c = b * np.exp(params[2])
        
        # If values are too large, overflow or precision issues?
        # The metric is scale invariant?
        # If we scale support by k, s_i + s_j < 2 s_l - s_k becomes k(s_i/s_k + ...) < 2(s_l/s_k) ...
        # Let s_i = k * t_i.
        # k t_i + k t_j < 2 k t_l - k t_k  =>  t_i + t_j < 2 t_l - t_k.
        # So the probability P is scale invariant.
        # Therefore, we can fix one scale parameter, e.g., a = 1.
        # This reduces dimensionality and improves conditioning.
        
        # Revised parameterization:
        # a = 1.0
        # b = exp(params[1])  (b > 1)
        # c = exp(params[2])  (c > 1, and we should ensure c > b? Not strictly required for the formula, but for unique identification of 'large' vs 'small' tiers it helps. 
        # Actually, if c < b, the roles swap. The formula handles any order.
        # But to avoid redundant local minima, let's just let b and c be independent > 1.
        
        # Wait, if a=1, b can be < 1? No, a is the smallest?
        # The support points are just a set. The labels 0,1,2 are arbitrary.
        # However, the weights are attached to specific points.
        # If we fix a=1, we are fixing the scale.
        # Let's allow a to vary slightly? No, scale invariance means we can fix a=1.
        # But what if the optimal support has a=0?
        # If a=0, then 0 is in the support.
        # Let's check if a=0 is possible in log-space. No.
        # So we must handle 0 separately or allow a to be very small.
        # Given the previous best score used 0, it's likely optimal to have 0 in the support.
        # Let's parameterize:
        # s0 = 0.0 (Fixed? Or small?)
        # If we fix s0=0, we lose generality if s0>0 is better.
        # Let's use a soft lower bound or just search over [0, 100].
        
        # Let's stick to the previous successful structure but allow movement.
        # Strategy: Search over (s0, s1, s2, w0, w1, w2) with s0=0?
        # The previous program found a good score with s0=0.
        # Let's assume s0=0 is optimal or near-optimal and optimize s1, s2, w0, w1, w2.
        # This reduces search space from 6 to 5 dims.
        
        s0 = 0.0
        s1 = np.exp(params[0]) # s1 > 0
        s2 = np.exp(params[1]) # s2 > 0
        
        # Ensure s1 != s2? If s1 == s2, it's effectively 2-point.
        # Weights
        w0 = np.exp(params[2])
        w1 = np.exp(params[3])
        w2 = np.exp(params[4])
        
        support = np.array([s0, s1, s2])
        weights = np.array([w0, w1, w2])
        
        p = calc_prob_exact(support, weights)
        return -p

    # Initial guesses based on current best:
    # support: [0, 1, 10]
    # weights: optimized around [0.5, 0.3, 0.2] roughly?
    # Let's run a coarse grid first to find a good starting point for the optimizer.
    
    best_score = -np.inf
    best_params = None
    best_support = None
    best_weights = None
    
    # Coarse Grid Search
    # s1 in [0.1, 5.0]
    # s2 in [1.0, 20.0]
    # Weights: normalize (w0, w1, w2).
    # Let's sample weights on a simplex.
    
    # To be fast, we will do a limited grid.
    s1_vals = np.linspace(0.1, 5.0, 10)
    s2_vals = np.linspace(1.0, 20.0, 10)
    
    # Weight grid: 5x5 on the 2D simplex (w0, w1, w2)
    w0_vals = np.linspace(0.1, 0.9, 5)
    w1_vals = np.linspace(0.1, 0.9, 5)
    
    for s1 in s1_vals:
        for s2 in s2_vals:
            if s2 < s1 * 1.01: continue # Ensure distinct
            
            for w0 in w0_vals:
                for w1 in w1_vals:
                    w2 = 1.0 - w0 - w1
                    if w2 < 0.01: continue
                    
                    support = np.array([0.0, s1, s2])
                    weights = np.array([w0, w1, w2])
                    p = calc_prob_exact(support, weights)
                    
                    if p > best_score:
                        best_score = p
                        best_support = support.copy()
                        best_weights = weights.copy()
                        
    if time.time() > deadline - 30:
        return best_support, best_weights

    # Local Optimization using scipy
    # Convert best_support/weights to log-params for objective function
    # s1 = exp(p0) => p0 = log(s1)
    # s2 = exp(p1) => p1 = log(s2)
    # w0 = exp(p2) => p2 = log(w0)
    # w1 = exp(p3) => p3 = log(w1)
    # w2 = exp(p4) => p4 = log(w2)
    
    x0 = np.array([
        np.log(best_support[1]),
        np.log(best_support[2]),
        np.log(best_weights[0]),
        np.log(best_weights[1]),
        np.log(best_weights[2])
    ])
    
    # Bounds for the optimizer (optional, but helpful)
    # s1, s2 > 0. w0, w1, w2 > 0.
    # We don't strictly bound them, but the exp ensures positivity.
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        
        if time.time() < deadline - 10:
            # Extract result
            if res.fun < -best_score: # res.fun is -P, so smaller is better
                p_res = -res.fun
                s1_res = np.exp(res.x[0])
                s2_res = np.exp(res.x[1])
                w0_res = np.exp(res.x[2])
                w1_res = np.exp(res.x[3])
                w2_res = np.exp(res.x[4])
                
                support_res = np.array([0.0, s1_res, s2_res])
                weights_res = np.array([w0_res, w1_res, w2_res])
                
                # Verify validity and score
                # Check for NaN/Inf
                if not (np.any(np.isnan(support_res)) or np.any(np.isnan(weights_res)) or 
                        np.any(np.isinf(support_res)) or np.any(np.isinf(weights_res))):
                    if np.min(support_res) >= -1e-9 and np.min(weights_res) >= -1e-9 and np.sum(weights_res) > 1e-9:
                        best_score = p_res
                        best_support = support_res
                        best_weights = weights_res
                        
    except Exception:
        pass

    # Final validity check
    if best_support is None:
        # Fallback
        best_support = np.array([0.0, 1.0, 10.0])
        best_weights = np.array([0.5, 0.3, 0.2])
        
    # Ensure weights sum to 1? The evaluator normalizes them, but it's cleaner if they do.
    # The contract says w_i = weights[i] / sum(weights). So raw weights are fine.
    # But we should ensure no negatives.
    if np.any(best_weights < 0):
        best_weights = np.abs(best_weights)
        
    return best_support, best_weights
