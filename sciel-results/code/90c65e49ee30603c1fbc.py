# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _evaluate_measure(support, weights):
    """
    Compute P = Prob[X1 + X2 + X3 < 2*X4] exactly in float64.
    Returns the probability as a float.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    # Normalize weights
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    p = weights / w_sum
    
    # Build all pairwise sums s_i + s_j with probabilities p_i * p_j
    pair_sums = support[:, None] + support[None, :]
    pair_probs = p[:, None] * p[None, :]
    
    pair_sums_flat = pair_sums.ravel()
    pair_probs_flat = pair_probs.ravel()
    
    # Sort by pair_sums
    sort_idx = np.argsort(pair_sums_flat)
    sorted_sums = pair_sums_flat[sort_idx]
    sorted_probs = pair_probs_flat[sort_idx]
    
    # Cumulative sum of probs
    cum_probs = np.cumsum(sorted_probs)
    
    # thresholds shape (n, n): thresholds[k, l] = 2*s_l - s_k
    thresholds = 2.0 * support[None, :] - support[:, None]
    T_flat = thresholds.ravel()
    
    # Find index where sorted_sums < T_flat (strict inequality)
    idx = np.searchsorted(sorted_sums, T_flat, side='left')
    
    cdf_vals = np.zeros(len(idx), dtype=np.float64)
    mask = idx > 0
    cdf_vals[mask] = cum_probs[idx[mask] - 1]
    
    cdf_vals = cdf_vals.reshape(n, n)
    
    # p_k p_l
    prob_kl = p[:, None] * p[None, :]
    
    P = np.sum(prob_kl * cdf_vals)
    
    return float(P)


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4].
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    best_score = -1.0
    best_support = None
    best_weights = None
    
    def update_best(sup, wts, score):
        nonlocal best_score, best_support, best_weights
        if score > best_score:
            best_score = score
            best_support = sup.copy()
            best_weights = wts.copy()
            print(f"New best: {score:.6f} at n={len(sup)}")
    
    def time_left():
        return deadline - time.time()
    
    def objective(params, n):
        """
        params: [s_0, s_1, ..., s_{n-1}, log_w_0, log_w_1, ..., log_w_{n-1}]
        Returns negative P (for minimization).
        """
        support = params[:n].copy()
        log_w = params[n:].copy()
        
        # Ensure support is sorted and non-negative
        support = np.maximum(support, 0.0)
        support = np.sort(support)
        
        # Convert log weights to probabilities via softmax
        # Subtract max for numerical stability
        log_w_shifted = log_w - np.max(log_w)
        w = np.exp(log_w_shifted)
        w_sum = np.sum(w)
        if w_sum < 1e-9:
            return 1.0  # Large penalty
        p = w / w_sum
        
        # Check for degenerate cases
        if np.any(w < -1e-9) or np.any(support < -1e-9):
            return 1.0
        
        try:
            P = _evaluate_measure(support, p)
        except Exception:
            return 1.0
        
        return -P
    
    # Strategy 1: Quick two-point scan to get a good starting point
    for p_val in np.linspace(0.0, 1.0, 501):
        if time_left() < 60:
            break
        support = np.array([0.0, 1.0])
        weights = np.array([1.0 - p_val, p_val])
        score = _evaluate_measure(support, weights)
        update_best(support, weights, score)
    
    # Strategy 2: Nelder-Mead optimization for n=3,4,5 with multiple starts
    np.random.seed(42)
    
    for n in [3, 4, 5]:
        if time_left() < 120:
            break
        
        # Number of restarts based on time available
        n_restarts = 8 if n == 3 else (6 if n == 4 else 4)
        
        for restart in range(n_restarts):
            if time_left() < 60:
                break
            
            # Generate diverse initializations
            if restart == 0:
                # Start near the best two-point solution extended
                if best_support is not None and len(best_support) == 2:
                    support_init = np.array([0.0, 1.0] + list(np.linspace(0.5, 2.0, n-2)))
                    support_init = np.sort(support_init)
                else:
                    support_init = np.sort(np.array([0.0, 1.0] + list(np.linspace(0.2, 1.8, n-2))))
                log_w_init = np.zeros(n)
            elif restart == 1:
                # Concentrated near 0
                support_init = np.sort(np.array([0.0, 0.1, 0.5] + list(np.random.uniform(0, 1, n-3))))
                log_w_init = np.random.randn(n)
            elif restart == 2:
                # Spread out
                support_init = np.sort(np.random.uniform(0, 3, n))
                support_init[0] = 0.0  # Ensure at least one point at 0
                log_w_init = np.random.randn(n)
            elif restart == 3:
                # Bimodal
                support_init = np.sort(np.concatenate([
                    np.random.uniform(0, 0.5, n//2),
                    np.random.uniform(1.5, 2.5, n - n//2)
                ]))
                support_init[0] = 0.0
                log_w_init = np.random.randn(n)
            else:
                # Random
                support_init = np.sort(np.random.uniform(0, 2, n))
                support_init[0] = 0.0
                log_w_init = np.random.randn(n)
            
            # Ensure support is valid
            support_init = np.maximum(support_init, 0.0)
            support_init = np.sort(support_init)
            
            x0 = np.concatenate([support_init, log_w_init])
            
            # Run Nelder-Mead
            result = minimize(
                objective,
                x0,
                args=(n,),
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if time_left() < 30:
                break
            
            # Extract best from result
            x_opt = result.x
            support_opt = np.maximum(x_opt[:n], 0.0)
            support_opt = np.sort(support_opt)
            log_w_opt = x_opt[n:]
            log_w_shifted = log_w_opt - np.max(log_w_opt)
            w_opt = np.exp(log_w_shifted)
            w_sum = np.sum(w_opt)
            if w_sum < 1e-9:
                continue
            p_opt = w_opt / w_sum
            
            score = _evaluate_measure(support_opt, p_opt)
            update_best(support_opt, p_opt, score)
            
            # Also try to refine the result with a tighter tolerance
            if time_left() > 60:
                result2 = minimize(
                    objective,
                    x_opt,
                    args=(n,),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 300,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                
                x_opt2 = result2.x
                support_opt2 = np.maximum(x_opt2[:n], 0.0)
                support_opt2 = np.sort(support_opt2)
                log_w_opt2 = x_opt2[n:]
                log_w_shifted2 = log_w_opt2 - np.max(log_w_opt2)
                w_opt2 = np.exp(log_w_shifted2)
                w_sum2 = np.sum(w_opt2)
                if w_sum2 < 1e-9:
                    continue
                p_opt2 = w_opt2 / w_sum2
                
                score2 = _evaluate_measure(support_opt2, p_opt2)
                update_best(support_opt2, p_opt2, score2)
    
    # Strategy 3: If we have a good solution, do a final local refinement
    if best_support is not None and time_left() > 90:
        n_best = len(best_support)
        
        # Create a parameter vector from the best solution
        log_w_best = np.log(np.maximum(best_weights, 1e-15))
        x_best = np.concatenate([best_support, log_w_best])
        
        # Run a final Nelder-Mead with tight tolerances
        result_final = minimize(
            objective,
            x_best,
            args=(n_best,),
            method='Nelder-Mead',
            options={
                'maxiter': 1000,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        
        x_final = result_final.x
        support_final = np.maximum(x_final[:n_best], 0.0)
        support_final = np.sort(support_final)
        log_w_final = x_final[n_best:]
        log_w_shifted_final = log_w_final - np.max(log_w_final)
        w_final = np.exp(log_w_shifted_final)
        w_sum_final = np.sum(w_final)
        if w_sum_final < 1e-9:
            pass
        else:
            p_final = w_final / w_sum_final
            score_final = _evaluate_measure(support_final, p_final)
            update_best(support_final, p_final, score_final)
    
    # Fallback if nothing found
    if best_support is None:
        support = np.array([0.0, 1.0])
        weights = np.array([0.5, 0.5])
        best_support = support
        best_weights = weights
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
