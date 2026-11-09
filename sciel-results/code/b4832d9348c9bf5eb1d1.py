# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, linprog, minimize

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
    
    # Build all pairwise sums
    pair_sums = support[:, None] + support[None, :]
    pair_probs = p[:, None] * p[None, :]
    
    pair_sums_flat = pair_sums.ravel()
    pair_probs_flat = pair_probs.ravel()
    
    sort_idx = np.argsort(pair_sums_flat)
    sorted_sums = pair_sums_flat[sort_idx]
    sorted_probs = pair_probs_flat[sort_idx]
    
    cum_probs = np.cumsum(sorted_probs)
    
    # thresholds[k, l] = 2*s_l - s_k
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

def _compute_gradient(support, weights):
    """
    Compute the gradient of P with respect to weights.
    P = sum_{k,l,i,j} w_k w_l w_i w_j I_{ij,kl}
    dP/dw_m = sum_{k,l,i,j} I_{ij,kl} * (delta_{mk} w_l w_i w_j + delta_{ml} w_k w_i w_j + delta_{mi} w_k w_l w_j + delta_{mj} w_k w_l w_i)
    Due to symmetry, we can compute this efficiently.
    """
    n = len(support)
    w_sum = np.sum(weights)
    p = weights / w_sum
    
    # Precompute indicators and sums
    # I_{ij,kl} = 1 if s_i + s_j < 2 s_l - s_k
    # This is a 4-tensor. For n=4, n^4 = 256, which is small.
    # We can compute the gradient by finite differences or analytical formula.
    # Analytical:
    # Let A_{kl} = sum_{ij: s_i+s_j < 2s_l-s_k} p_i p_j
    # Then P = sum_{kl} p_k p_l A_{kl}
    # dP/dp_m = sum_{kl} (delta_{mk} p_l A_{kl} + p_k delta_{ml} A_{kl} + p_k p_l dA_{kl}/dp_m)
    # dA_{kl}/dp_m = sum_{ij} I_{ij,kl} (delta_{mi} p_j + delta_{mj} p_i)
    
    # For small n, finite difference is safer and faster to implement correctly.
    eps = 1e-6
    grad = np.zeros(n)
    base_P = _evaluate_measure(support, weights)
    
    for m in range(n):
        w_plus = weights.copy()
        w_plus[m] += eps
        # Renormalize? No, P is homogeneous of degree 4? 
        # P(w) = P(w * c) for any c? 
        # Prob[X1..X4] uses normalized weights.
        # So P depends on direction of w.
        # If we change w_m, we change the distribution.
        # We should evaluate P with the new normalized weights.
        w_plus_norm = w_plus / np.sum(w_plus)
        P_plus = _evaluate_measure(support, w_plus_norm)
        grad[m] = (P_plus - base_P) / eps
        
    return grad

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4].
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s limit
    
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
    
    # 1. Analytic 2-point optimum
    p_opt_two = (1 + np.sqrt(17)) / 8.0
    support_two = np.array([0.0, 1.0])
    weights_two = np.array([1.0 - p_opt_two, p_opt_two])
    score_two = _evaluate_measure(support_two, weights_two)
    update_best(support_two, weights_two, score_two)
    
    # 2. Known good 3-point configuration
    support_three = np.array([0.0, 1.0, 10.0])
    # Optimize weights for this support using a quick grid or DE
    # For now, use a reasonable guess
    weights_three_init = np.array([0.5, 0.3, 0.2])
    score_three = _evaluate_measure(support_three, weights_three_init)
    update_best(support_three, weights_three_init, score_three)
    
    # 3. Global Search with Differential Evolution
    # We search for 4 points.
    n_points = 4
    # Parameters: s0, s1, s2, s3, w0, w1, w2, w3
    # Constraints: s0 <= s1 <= s2 <= s3, s_i >= 0
    #               sum(w) = 1, w_i >= 0
    
    def de_objective(params):
        s = params[:n_points]
        w = params[n_points:]
        
        # Enforce ordering
        s = np.sort(s)
        s = np.maximum(s, 0.0)
        
        # Normalize weights
        w = np.maximum(w, 0.0)
        w_sum = np.sum(w)
        if w_sum < 1e-9:
            return 1.0
        p = w / w_sum
        
        try:
            P = _evaluate_measure(s, p)
        except Exception:
            return 1.0
        
        return -P
    
    # Bounds: s in [0, 50], w in [0, 1]
    bounds = [(0, 50)] * n_points + [(0, 1)] * n_points
    
    # Initial population: include the known good points
    # DE doesn't take a single initial point, but we can use a small population
    # and rely on the global search.
    # To speed up, we use a small popsize and maxiter.
    
    if time_left() > 300:
        try:
            result_de = differential_evolution(
                de_objective,
                bounds,
                popsize=10,
                maxiter=50,
                seed=42,
                tol=1e-4,
                polish=True,  # Use Nelder-Mead to polish the best result
                polish_options={'maxiter': 200}
            )
            
            if result_de.success or result_de.fun < 1.0:
                best_params = result_de.x
                best_s = np.sort(np.maximum(best_params[:n_points], 0.0))
                best_w = np.maximum(best_params[n_points:], 0.0)
                best_w_sum = np.sum(best_w)
                if best_w_sum > 1e-9:
                    best_w = best_w / best_w_sum
                    score_de = _evaluate_measure(best_s, best_w)
                    update_best(best_s, best_w, score_de)
        except Exception:
            pass
    
    # 4. Local Refinement using LP (Directive Compliance)
    # For the best support found, refine weights using LP-based gradient step.
    if best_support is not None and time_left() > 60:
        s_best = best_support
        w_best = best_weights
        n = len(s_best)
        
        # Ensure w_best is normalized
        w_best = w_best / np.sum(w_best)
        
        for _ in range(20): # 20 iterations of LP-based refinement
            if time_left() < 30:
                break
            
            # Compute gradient
            grad = _compute_gradient(s_best, w_best)
            
            # Solve LP: Max grad . w s.t. sum(w)=1, w>=0
            # c = -grad (minimize -grad . w)
            c = -grad
            A_eq = np.ones((1, n))
            b_eq = [1.0]
            
            try:
                res_lp = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, 1)] * n, method='highs')
                if res_lp.success:
                    w_new = res_lp.x
                    # Update weights
                    # Simple projection or step
                    w_best = w_new
                    score_lp = _evaluate_measure(s_best, w_best)
                    update_best(s_best, w_best, score_lp)
                else:
                    break
            except Exception:
                break
        
        # Final Nelder-Mead polish on both support and weights
        if time_left() > 30:
            def nm_objective(params):
                s = params[:n]
                w = params[n:]
                s = np.sort(np.maximum(s, 0.0))
                w = np.maximum(w, 0.0)
                w_sum = np.sum(w)
                if w_sum < 1e-9:
                    return 1.0
                p = w / w_sum
                try:
                    P = _evaluate_measure(s, p)
                except Exception:
                    return 1.0
                return -P
            
            x0 = np.concatenate([s_best, w_best])
            result_nm = minimize(
                nm_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            x_final = result_nm.x
            s_final = np.sort(np.maximum(x_final[:n], 0.0))
            w_final = np.maximum(x_final[n:], 0.0)
            w_final_sum = np.sum(w_final)
            if w_final_sum > 1e-9:
                w_final = w_final / w_final_sum
                score_final = _evaluate_measure(s_final, w_final)
                update_best(s_final, w_final, score_final)
    
    # Fallback
    if best_support is None:
        best_support = support_two
        best_weights = weights_two
    
    # Ensure validity
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
