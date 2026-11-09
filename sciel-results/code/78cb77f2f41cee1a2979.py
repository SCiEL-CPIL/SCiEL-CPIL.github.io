import numpy as np
import time
import random
from scipy.optimize import minimize, differential_evolution

def _calc_prob_fast(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] for a discrete measure.
    Vectorized for speed.
    Returns 0 if invalid.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    # Basic validity check for speed
    if np.any(np.isnan(support)) or np.any(np.isnan(weights)):
        return 0.0
    if np.any(np.isinf(support)) or np.any(np.isinf(weights)):
        return 0.0
    if np.any(support < -1e-9) or np.any(weights < -1e-9):
        return 0.0
        
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
        
    p = weights / w_sum
    
    # Pairwise sums matrix: S[i,j] = s_i + s_j
    S = support[:, None] + support[None, :]
    
    # Weights outer product: W[i,j] = p_i * p_j
    W = p[:, None] * p[None, :]
    
    S_flat = S.flatten()
    W_flat = W.flatten()
    
    # Pre-sort S_flat and corresponding W_flat for fast prefix sums
    sort_idx = np.argsort(S_flat)
    S_sorted = S_flat[sort_idx]
    W_sorted = W_flat[sort_idx]
    W_cumsum = np.cumsum(W_sorted)
    W_cumsum_padded = np.concatenate(([0.0], W_cumsum))
    
    # Vectorize over k: for each k, T_k = 2*s_k - s_l for all l
    T_matrix = 2.0 * support[:, None] - support[None, :]
    T_flat = T_matrix.flatten()
    
    # Find indices in S_sorted where S_sorted < T_flat
    indices = np.searchsorted(S_sorted, T_flat, side='left')
    
    # Sum of weights for each (k,l) pair
    sums_for_kl = W_cumsum_padded[indices]
    sums_for_kl = sums_for_kl.reshape(n, n)
    
    # prob3_k = sum_l p_l * sums_for_kl[k,l]
    prob3 = sums_for_kl @ p  # shape (n,)
    
    total_prob = np.dot(p, prob3)
    
    return float(total_prob)

def _calc_prob_3pt(p0, p1, pM):
    """
    Exact probability for support {0, 1, M} with weights p0, p1, pM.
    Assumes p0+p1+pM=1, p_i >= 0, M > 0.
    """
    if p0 < 0 or p1 < 0 or pM < 0:
        return 0.0
    s = np.array([0.0, 1.0, M], dtype=np.float64)
    w = np.array([p0, p1, pM], dtype=np.float64)
    return _calc_prob_fast(s, w)

def _validate(support, weights):
    if len(support) != len(weights):
        return False
    if len(support) < 1:
        return False
    if np.any(np.isnan(support)) or np.any(np.isnan(weights)):
        return False
    if np.any(np.isinf(support)) or np.any(np.isinf(weights)):
        return False
    if np.any(support < -1e-9):
        return False
    if np.any(weights < -1e-9):
        return False
    if np.sum(weights) <= 1e-9:
        return False
    return True

def _params_to_measure(params, n):
    """
    Convert unconstrained parameters to valid support and weights.
    params[0:n] -> log(support)
    params[n:2n] -> raw weights (softmax or abs+normalize)
    """
    log_s = params[:n]
    raw_w = params[n:]
    
    # Support: ensure non-negative. log_s can be any real number.
    # exp(log_s) is always positive.
    support = np.exp(log_s)
    
    # Weights: ensure positive. 
    # Using exp(raw_w) ensures positivity.
    exp_w = np.exp(raw_w - np.max(raw_w)) # for numerical stability
    weights = exp_w / np.sum(exp_w)
    
    return support, weights

def _objective(params, n):
    support, weights = _params_to_measure(params, n)
    if not _validate(support, weights):
        return -1.0
    prob = _calc_prob_fast(support, weights)
    return -prob # Minimizing negative probability

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game using multi-phase optimization with diverse initialization.
    """
    start_time = time.time()
    deadline = start_time + 585.0  # Safety margin for 600s limit
    
    best_prob = -1.0
    best_support = None
    best_weights = None
    
    # Helper to update best
    def try_update(support, weights, prob):
        nonlocal best_prob, best_support, best_weights
        if prob > best_prob:
            best_prob = prob
            best_support = support.copy()
            best_weights = weights.copy()
    
    # Phase 0: Analytical/Grid Search for 3-point support {0, 1, M}
    # This is extremely fast and likely finds a near-optimal solution immediately.
    # We optimize p1 and pM (p0 = 1 - p1 - pM) and M.
    
    # Grid search for M
    # M likely in range [0.5, 10]
    M_values = np.linspace(0.1, 10.0, 1000)
    
    for M in M_values:
        # For fixed M, maximize over p1, pM
        # We can use a small grid for p1, pM or analytical if possible.
        # Given the piecewise linear nature, the optimum for fixed M is often at boundaries or specific intersections.
        # Let's do a fine grid for p1, pM.
        # p1 + pM <= 1.
        # p1 in [0, 1], pM in [0, 1-p1].
        
        # To be fast, we sample a grid for p1 and pM.
        # 100x100 grid per M is 10k evaluations. 1000 Ms -> 10M evals.
        # _calc_prob_fast is vectorized but n=3 is small.
        # Let's reduce grid size to be safe, or use random sampling + local refine.
        
        # Actually, let's just do a coarse grid first, then refine best candidates.
        p1_grid = np.linspace(0, 1, 50)
        pM_grid = np.linspace(0, 1, 50)
        
        for p1 in p1_grid:
            for pM in pM_grid:
                if p1 + pM > 1.0:
                    continue
                p0 = 1.0 - p1 - pM
                prob = _calc_prob_3pt(p0, p1, pM)
                if prob > best_prob:
                    best_prob = prob
                    best_support = np.array([0.0, 1.0, M], dtype=np.float64)
                    best_weights = np.array([p0, p1, pM], dtype=np.float64)
                    
    # Refine the best 3-point candidate found in Phase 0
    if best_support is not None and len(best_support) == 3:
        # Use Nelder-Mead on the 3-point parameters: [log(M), log(p1), log(pM)]
        # Note: p0 is determined.
        # We need to handle constraints p1+pM <= 1.
        # Let's parameterize p1 = exp(a)/(1+exp(a)+exp(b)), pM = exp(b)/(1+exp(a)+exp(b)), p0 = 1/(1+exp(a)+exp(b))
        # This ensures they sum to 1 and are positive.
        
        best_M = best_support[2]
        best_p1 = best_weights[1]
        best_pM = best_weights[2]
        
        # Inverse transform to a, b
        # p1 = e^a / D, pM = e^b / D, p0 = 1/D
        # p1/p0 = e^a => a = log(p1/p0)
        # pM/p0 = e^b => b = log(pM/p0)
        
        def obj_3pt(params):
            a, b, log_M = params
            M = np.exp(log_M)
            if M < 1e-6:
                return 1.0
            e_a = np.exp(a)
            e_b = np.exp(b)
            D = 1.0 + e_a + e_b
            p0 = 1.0 / D
            p1 = e_a / D
            pM = e_b / D
            return -_calc_prob_3pt(p0, p1, pM)
        
        # Initial params
        p0_init = best_weights[0]
        if p0_init < 1e-9:
            p0_init = 1e-9
            # Adjust others
            best_weights[0] = p0_init
            rest_sum = 1.0 - p0_init
            best_weights[1] *= rest_sum
            best_weights[2] *= rest_sum
            best_weights = best_weights / np.sum(best_weights)
            p0_init = best_weights[0]
            best_p1 = best_weights[1]
            best_pM = best_weights[2]
            
        a_init = np.log(best_p1 / p0_init)
        b_init = np.log(best_pM / p0_init)
        log_M_init = np.log(best_M)
        
        x0 = np.array([a_init, b_init, log_M_init])
        
        try:
            res = minimize(obj_3pt, x0, method='Nelder-Mead', options={'maxfev': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            a_opt, b_opt, log_M_opt = res.x
            M_opt = np.exp(log_M_opt)
            e_a = np.exp(a_opt)
            e_b = np.exp(b_opt)
            D = 1.0 + e_a + e_b
            p0_opt = 1.0 / D
            p1_opt = e_a / D
            pM_opt = e_b / D
            
            s_opt = np.array([0.0, 1.0, M_opt], dtype=np.float64)
            w_opt = np.array([p0_opt, p1_opt, pM_opt], dtype=np.float64)
            
            prob_opt = _calc_prob_fast(s_opt, w_opt)
            try_update(s_opt, w_opt, prob_opt)
        except Exception:
            pass
    
    # Phase 1: General search for n > 3 if time permits
    # The 3-point solution is often very strong. We use remaining time to see if n=4 or 5 beats it.
    
    if time.time() < deadline - 30:
        # Initialize with some diverse starting points for general n
        initializations = [
            ([0.1, 0.5, 1.0, 2.0, 4.0], [0.1, 0.2, 0.3, 0.25, 0.15]),
            ([0.0, 1.0, 3.0], [0.3, 0.4, 0.3]),
            ([0.5, 1.0, 2.0, 4.0, 8.0], [0.15, 0.2, 0.3, 0.2, 0.15]),
            ([0.2, 0.8, 1.5, 3.0], [0.25, 0.3, 0.25, 0.2]),
            ([1.0, 2.0], [0.5, 0.5]),
        ]
        
        for s_init, w_init in initializations:
            s = np.array(s_init, dtype=np.float64)
            w = np.array(w_init, dtype=np.float64)
            if _validate(s, w):
                prob = _calc_prob_fast(s, w)
                try_update(s, w, prob)
        
        # Simulated Annealing for general n
        phase1_deadline = start_time + (deadline - start_time) * 0.7
        T_start = 1.0
        T_end = 0.005
        cooling_rate = 0.997
        
        # Start from best found so far
        current_support = best_support.copy()
        current_weights = best_weights.copy()
        current_prob = best_prob
        T = T_start
        step_scale = 1.0
        
        while time.time() < phase1_deadline:
            move_type = random.random()
            
            if move_type < 0.55:
                n = len(current_support)
                log_s = np.log(np.maximum(current_support, 1e-10))
                log_s += np.random.normal(0, 0.25 * step_scale, size=n)
                new_support = np.exp(log_s)
                new_support = np.maximum(new_support, 0.0)
                
                new_weights = current_weights * np.exp(np.random.normal(0, 0.15 * step_scale, size=n))
                new_weights = np.maximum(new_weights, 1e-10)
                new_weights = new_weights / np.sum(new_weights)
                
                new_prob = _calc_prob_fast(new_support, new_weights)
                
            elif move_type < 0.78 and len(current_support) < 10:
                n = len(current_support)
                new_support = current_support.copy()
                new_weights = current_weights.copy()
                
                insert_idx = random.randint(0, n)
                if insert_idx == 0:
                    new_val = max(current_support[0] * np.exp(np.random.normal(0, 0.4)), 0.0)
                elif insert_idx == n:
                    new_val = max(current_support[-1] * np.exp(np.random.normal(0, 0.4)), 0.0)
                else:
                    left = current_support[insert_idx - 1]
                    right = current_support[insert_idx]
                    new_val = np.exp(np.log(max(left, 1e-10)) + np.random.uniform(0, 1) * (np.log(max(right, 1e-10)) - np.log(max(left, 1e-10))))
                    new_val = max(new_val, 0.0)
                
                new_support = np.insert(new_support, insert_idx, new_val)
                
                if insert_idx == 0:
                    w_new = new_weights[0] * 0.4
                    new_weights[0] -= w_new
                    new_weights = np.insert(new_weights, 0, w_new)
                elif insert_idx == n:
                    w_new = new_weights[-1] * 0.4
                    new_weights[-1] -= w_new
                    new_weights = np.insert(new_weights, -1, w_new)
                else:
                    w_new = (new_weights[insert_idx - 1] + new_weights[insert_idx]) * 0.2
                    new_weights[insert_idx - 1] -= w_new
                    new_weights[insert_idx] -= w_new
                    new_weights = np.insert(new_weights, insert_idx, w_new)
                
                new_weights = np.maximum(new_weights, 1e-10)
                new_weights = new_weights / np.sum(new_weights)
                
                new_prob = _calc_prob_fast(new_support, new_weights)
                
            elif move_type < 0.92 and len(current_support) > 2:
                n = len(current_support)
                remove_idx = random.randint(0, n - 1)
                
                new_support = np.delete(current_support, remove_idx)
                new_weights = np.delete(current_weights, remove_idx)
                
                if len(new_weights) > 0:
                    w_removed = current_weights[remove_idx]
                    if remove_idx == 0:
                        new_weights[0] += w_removed
                    elif remove_idx == n - 1:
                        new_weights[-1] += w_removed
                    else:
                        new_weights[remove_idx - 1] += w_removed * 0.5
                        new_weights[remove_idx] += w_removed * 0.5
                
                new_weights = np.maximum(new_weights, 1e-10)
                new_weights = new_weights / np.sum(new_weights)
                
                new_prob = _calc_prob_fast(new_support, new_weights)
            
            else:
                n = len(current_support)
                new_support = current_support * np.exp(np.random.normal(0, 0.8, size=n))
                new_support = np.maximum(new_support, 0.0)
                new_weights = np.abs(np.random.normal(1.0, 0.4, size=n)) + 0.05
                new_weights = new_weights / np.sum(new_weights)
                
                new_prob = _calc_prob_fast(new_support, new_weights)
            
            delta = new_prob - current_prob
            if delta > 0 or random.random() < np.exp(delta / max(T, 1e-10)):
                current_support = new_support
                current_weights = new_weights
                current_prob = new_prob
                try_update(current_support, current_weights, current_prob)
                step_scale = min(step_scale * 1.005, 2.5)
            else:
                step_scale = max(step_scale * 0.995, 0.2)
            
            T *= cooling_rate
            if T < T_end:
                T = T_end
    
    # Fallback
    if best_support is None:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([0.5, 0.5], dtype=np.float64)
    else:
        if not _validate(best_support, best_weights):
            best_support = np.array([0.0, 1.0], dtype=np.float64)
            best_weights = np.array([0.5, 0.5], dtype=np.float64)
            
        w_sum = np.sum(best_weights)
        if w_sum > 1e-9:
            best_weights = best_weights / w_sum
        else:
            best_weights = np.ones(len(best_support)) / len(best_support)
            
        best_support = np.maximum(best_support, 0.0)

    return best_support.astype(np.float64), best_weights.astype(np.float64)

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print("Support:", s)
    print("Weights:", w)
    print("Probability:", _calc_prob_fast(s, w))
