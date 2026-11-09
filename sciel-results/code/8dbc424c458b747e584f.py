# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _evaluate_measure(support, weights):
    """
    Compute P = Prob[X1 + X2 + X3 < 2*X4] exactly in float64.
    X1, X2, X3, X4 are i.i.d. from the discrete measure defined by (support, weights).
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    p = weights / w_sum
    
    # For small n, use direct computation which avoids overhead
    if n <= 16:
        # Direct: for each (k,l), compute CDF(2*s_l - s_k) by summing over (i,j)
        P = 0.0
        for l in range(n):
            for k in range(n):
                threshold = 2.0 * support[l] - support[k]
                # Sum p_i * p_j over (i,j) with s_i + s_j < threshold
                cdf_val = 0.0
                for i in range(n):
                    for j in range(n):
                        if support[i] + support[j] < threshold:
                            cdf_val += p[i] * p[j]
                P += p[k] * p[l] * cdf_val
        return float(P)
    else:
        # For larger n, use the sort+searchsorted approach
        if not np.all(support[:-1] <= support[1:]):
            order = np.argsort(support)
            support = support[order]
            p = p[order]
        
        pair_sums = support[:, None] + support[None, :]
        pair_probs = p[:, None] * p[None, :]
        
        pair_sums_flat = pair_sums.ravel()
        pair_probs_flat = pair_probs.ravel()
        
        sort_idx = np.argsort(pair_sums_flat, kind='stable')
        sorted_sums = pair_sums_flat[sort_idx]
        sorted_probs = pair_probs_flat[sort_idx]
        cum_probs = np.cumsum(sorted_probs)
        
        thresholds = 2.0 * support[None, :] - support[:, None]
        T_flat = thresholds.ravel()
        
        idx = np.searchsorted(sorted_sums, T_flat, side='left')
        
        cdf_vals = np.zeros(len(idx), dtype=np.float64)
        mask = idx > 0
        cdf_vals[mask] = cum_probs[idx[mask] - 1]
        
        cdf_vals = cdf_vals.reshape(n, n)
        
        prob_kl = p[:, None] * p[None, :]
        
        P = np.sum(prob_kl * cdf_vals)
        
        return float(P)


def _evaluate_from_params(params, n):
    """
    Evaluate measure from raw parameters: [s_0..s_{n-1}, w_0..w_{n-1}]
    """
    support = params[:n]
    weights = params[n:]
    
    # Clamp support to non-negative and sort
    support = np.maximum(support, 0.0)
    support = np.sort(support)
    
    # Clamp weights to non-negative and normalize
    weights = np.maximum(weights, 0.0)
    w_sum = np.sum(weights)
    if w_sum < 1e-9:
        return 0.0
    p = weights / w_sum
    
    try:
        P = _evaluate_measure(support, p)
    except Exception:
        return 0.0
    
    return P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4].
    """
    start_time = time.time()
    deadline = start_time + 595  # Safety margin before 630s limit
    
    best_score = -1.0
    best_support = None
    best_weights = None
    
    def update_best(sup, wts, score):
        nonlocal best_score, best_support, best_weights
        if score > best_score:
            best_score = score
            best_support = np.array(sup, dtype=np.float64)
            best_weights = np.array(wts, dtype=np.float64)
    
    def time_left():
        return deadline - time.time()
    
    # Analytic 2-point optimum: support {0, 1}, weight p* on 1
    # P = p*(1-p)^2*(1+2p), maximize over p
    # f'(p) = 1 - 9p^2 + 8p^3 = 0 => 8p^3 - 9p^2 + 1 = 0
    # (p-1)(8p^2 - p - 1) = 0 => p = (1+sqrt(33))/16
    p_opt_two = (1.0 + np.sqrt(33.0)) / 16.0
    support_two = np.array([0.0, 1.0])
    weights_two = np.array([1.0 - p_opt_two, p_opt_two])
    score_two = _evaluate_measure(support_two, weights_two)
    update_best(support_two, weights_two, score_two)
    
    # Also try a few other 2-point configurations for robustness
    for p_try in [0.3, 0.4, 0.42, 0.4215, 0.45, 0.5, 0.55, 0.6]:
        w_try = np.array([1.0 - p_try, p_try])
        s_try = np.array([0.0, 1.0])
        sc_try = _evaluate_measure(s_try, w_try)
        update_best(s_try, w_try, sc_try)
    
    # For differential evolution: minimize -P
    def objective(params, n):
        P = _evaluate_from_params(params, n)
        return -P
    
    # Try various sizes of support - focus more on smaller n for speed
    for n in [3, 4, 5, 6]:
        if time_left() < 90:
            break
        
        # Bounds for support: [0, 3]
        # Bounds for weights: [0, 1]
        bounds = [(0.0, 3.0)] * n + [(0.0, 1.0)] * n
        
        # More restarts for smaller n (faster evaluations with direct method)
        n_restarts = 8 if n == 3 else (6 if n == 4 else (4 if n == 5 else 3))
        
        for restart in range(n_restarts):
            if time_left() < 30:
                break
            
            # Different initial populations
            if restart == 0:
                # Start near the 2-point solution, spread out
                init_s = np.sort(np.linspace(0.0, 1.0, n))
                init_w = np.ones(n) / n
            elif restart == 1:
                # Random initialization
                init_s = np.sort(np.random.uniform(0, 2, n))
                init_s[0] = 0.0
                init_w = np.random.dirichlet(np.ones(n) * 2)
            elif restart == 2:
                # Another random initialization
                init_s = np.sort(np.random.uniform(0, 3, n))
                init_s[0] = 0.0
                init_w = np.random.dirichlet(np.ones(n))
            elif restart == 3:
                # Concentrated initialization
                init_s = np.sort(np.concatenate([[0.0], np.random.uniform(0.5, 1.5, n-1)]))
                init_w = np.random.dirichlet(np.ones(n) * 5)
            elif restart == 4:
                # Wide spread
                init_s = np.sort(np.concatenate([[0.0], np.random.uniform(1.0, 3.0, n-1)]))
                init_w = np.random.dirichlet(np.ones(n) * 1.5)
            elif restart == 5:
                # Near optimal 2-point, add small perturbation
                init_s = np.sort(np.concatenate([[0.0, 1.0], np.random.uniform(0.5, 1.5, n-2)]))
                init_w = np.concatenate([[1.0 - p_opt_two, p_opt_two], np.random.dirichlet(np.ones(n-2) * 0.5)])
                w_sum_init = np.sum(init_w)
                init_w = init_w / w_sum_init
            elif restart == 6:
                # Geometric spacing
                init_s = np.sort(np.geomspace(0.01, 2.0, n))
                init_s[0] = 0.0
                init_w = np.random.dirichlet(np.ones(n) * 3)
            else:
                # Uniform with emphasis on [0,1]
                init_s = np.sort(np.random.uniform(0, 1, n))
                init_s[0] = 0.0
                init_w = np.random.dirichlet(np.ones(n) * 4)
            
            x0 = np.concatenate([init_s, init_w])
            
            # Run differential evolution with adjusted parameters
            maxiter = 300 if n <= 4 else 200
            popsize = 30 if n <= 4 else 25
            
            try:
                result = differential_evolution(
                    objective,
                    bounds,
                    args=(n,),
                    x0=x0,
                    seed=42 + restart * 100 + n * 1000,
                    maxiter=maxiter,
                    popsize=popsize,
                    tol=1e-10,
                    polish=False,
                    init='sobol' if restart == 0 else 'latinhypercube',
                    mutation=(0.5, 1.0),
                    recombination=0.9
                )
            except Exception:
                continue
            
            x_opt = result.x
            P_opt = _evaluate_from_params(x_opt, n)
            
            support_opt = np.maximum(x_opt[:n], 0.0)
            support_opt = np.sort(support_opt)
            w_opt = np.maximum(x_opt[n:], 0.0)
            w_sum = np.sum(w_opt)
            if w_sum < 1e-9:
                continue
            p_opt = w_opt / w_sum
            
            update_best(support_opt, p_opt, P_opt)
    
    # Try to grow the solution: add a new point and optimize (limited)
    if best_support is not None and time_left() > 90:
        n_best = len(best_support)
        n_grow = n_best + 1
        
        # Insert a new point between existing ones or at the end
        candidates = []
        if n_best >= 2:
            mid = (best_support[-1] - best_support[0]) / 2
            candidates.append(mid)
        candidates.append(best_support[-1] + 0.5)
        candidates.append(0.1)
        
        for new_point in candidates[:2]:  # Limit to 2 candidates
            if time_left() < 45:
                break
            
            # Build initial support by inserting new point
            init_s = np.sort(np.concatenate([best_support, [new_point]]))
            # Start with small weight on new point
            init_w = np.concatenate([best_weights, [0.05]])
            w_sum_init = np.sum(init_w)
            init_w = init_w / w_sum_init
            
            x0 = np.concatenate([init_s, init_w])
            bounds = [(0.0, 3.0)] * n_grow + [(0.0, 1.0)] * n_grow
            
            try:
                result = differential_evolution(
                    objective,
                    bounds,
                    args=(n_grow,),
                    x0=x0,
                    seed=42 + n_grow * 10000,
                    maxiter=200,
                    popsize=25,
                    tol=1e-10,
                    polish=False,
                    init='latinhypercube',
                    mutation=(0.5, 1.0),
                    recombination=0.9
                )
            except Exception:
                continue
            
            x_opt = result.x
            P_opt = _evaluate_from_params(x_opt, n_grow)
            
            support_opt = np.maximum(x_opt[:n_grow], 0.0)
            support_opt = np.sort(support_opt)
            w_opt = np.maximum(x_opt[n_grow:], 0.0)
            w_sum = np.sum(w_opt)
            if w_sum < 1e-9:
                continue
            p_opt = w_opt / w_sum
            
            update_best(support_opt, p_opt, P_opt)
    
    # Final local polish with Nelder-Mead on the best solution - high precision
    if best_support is not None and time_left() > 20:
        n_best = len(best_support)
        
        def nm_objective(params):
            support = params[:n_best]
            log_w = params[n_best:]
            
            support = np.maximum(support, 0.0)
            support = np.sort(support)
            
            log_w_shifted = log_w - np.max(log_w)
            w = np.exp(log_w_shifted)
            w_sum = np.sum(w)
            if w_sum < 1e-9:
                return 0.0
            p = w / w_sum
            
            try:
                P = _evaluate_measure(support, p)
            except Exception:
                return 0.0
            
            return -P
        
        log_w_best = np.log(np.maximum(best_weights, 1e-15))
        x_best = np.concatenate([best_support, log_w_best])
        
        # Multiple polish passes with increasing precision
        for polish_pass in range(4):
            if time_left() < 10:
                break
            
            maxiter = 3000 if polish_pass == 0 else 8000
            xatol = 1e-12 if polish_pass == 0 else (1e-14 if polish_pass == 1 else (1e-15 if polish_pass == 2 else 1e-16))
            fatol = 1e-14 if polish_pass == 0 else (1e-16 if polish_pass == 1 else (1e-17 if polish_pass == 2 else 1e-18))
            
            try:
                result_final = minimize(
                    nm_objective,
                    x_best,
                    method='Nelder-Mead',
                    options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
                )
            except Exception:
                result_final = None
            
            if result_final is not None:
                x_best = result_final.x
                support_final = np.maximum(x_best[:n_best], 0.0)
                support_final = np.sort(support_final)
                log_w_final = x_best[n_best:]
                log_w_shifted_final = log_w_final - np.max(log_w_final)
                w_final = np.exp(log_w_shifted_final)
                w_sum_final = np.sum(w_final)
                if w_sum_final >= 1e-9:
                    p_final = w_final / w_sum_final
                    score_final = _evaluate_measure(support_final, p_final)
                    update_best(support_final, p_final, score_final)
        
        # Coordinate descent as final refinement
        if time_left() > 15:
            for coord_pass in range(2):
                if time_left() < 10:
                    break
                for dim in range(n_best * 2):
                    if time_left() < 5:
                        break
                    # Try small perturbations in each dimension
                    for delta in [1e-6, 1e-7, 1e-8, -1e-6, -1e-7, -1e-8]:
                        x_trial = x_best.copy()
                        x_trial[dim] += delta
                        val = nm_objective(x_trial)
                        if val < nm_objective(x_best):
                            x_best = x_trial
                            support_final = np.maximum(x_best[:n_best], 0.0)
                            support_final = np.sort(support_final)
                            log_w_final = x_best[n_best:]
                            log_w_shifted_final = log_w_final - np.max(log_w_final)
                            w_final = np.exp(log_w_shifted_final)
                            w_sum_final = np.sum(w_final)
                            if w_sum_final >= 1e-9:
                                p_final = w_final / w_sum_final
                                score_final = _evaluate_measure(support_final, p_final)
                                update_best(support_final, p_final, score_final)
    
    # Fallback
    if best_support is None:
        best_support = support_two
        best_weights = weights_two
    
    # Ensure validity: no NaN, no inf, non-negative, positive sum
    best_support = np.nan_to_num(best_support, nan=0.0, posinf=0.0, neginf=0.0)
    best_weights = np.nan_to_num(best_weights, nan=0.0, posinf=0.0, neginf=0.0)
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    w_sum = np.sum(best_weights)
    if w_sum < 1e-9:
        best_support = support_two
        best_weights = weights_two
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
