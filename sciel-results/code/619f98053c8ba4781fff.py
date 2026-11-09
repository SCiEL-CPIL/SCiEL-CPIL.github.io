import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _evaluate_measure(support, weights):
    """
    Compute P = Prob[X1 + X2 + X3 < 2*X4] exactly in float64.
    Uses sorted pairwise sums + cumulative probabilities + searchsorted for O(n^2 log n) per call.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    p = weights / w_sum
    
    # Build all pairwise sums and probabilities
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


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin before 630s kill

    def time_left():
        return deadline - time.time()

    best_score = -1.0
    best_support = None
    best_weights = None
    
    def update_best(sup, wts, score):
        nonlocal best_score, best_support, best_weights
        if score > best_score:
            best_score = score
            best_support = np.array(sup, dtype=np.float64)
            best_weights = np.array(wts, dtype=np.float64)
    
    # Analytic 2-point optimum
    p_opt_two = (1.0 + np.sqrt(17.0)) / 8.0
    support_two = np.array([0.0, 1.0])
    weights_two = np.array([1.0 - p_opt_two, p_opt_two])
    score_two = _evaluate_measure(support_two, weights_two)
    update_best(support_two, weights_two, score_two)

    # Differential evolution for n=3, 4, 5, 6
    # Using a general parameterization: support points and weights
    # To handle the simplex constraint on weights, we optimize log-weights or use softmax
    # Here we use a simple approach: optimize raw weights and normalize, with bounds to keep them positive
    # But DE works better with unconstrained or box-constrained variables.
    # Let's stick to the box-constrained approach used in the current program but with the faster evaluator.
    
    for n in [3, 4, 5, 6]:
        if time_left() < 60:
            break
        
        # We fix support[0]=0 and support[-1]=1 to reduce dimensionality?
        # The current program does this. It's a reasonable assumption that optimal support lies in [0,1] 
        # or can be scaled. Actually, the problem is scale-invariant?
        # P[X1+X2+X3 < 2X4]. If we scale all X by c, the inequality becomes c(X1'+X2'+X3') < 2c(X4'), 
        # so X1'+X2'+X3' < 2X4'. The probability is invariant under scaling.
        # So we can fix the support range. The current program fixes [0,1].
        # This is a good structural choice to reduce search space.
        
        # Parameters: interior support points (n-2) and interior weights (n-1)
        # w0 is determined by sum-to-1.
        
        def objective(params):
            n_int_s = n - 2
            n_int_w = n - 1
            s_int = params[:n_int_s]
            w_int = params[n_int_s:]
            
            # Check constraints
            if np.any(s_int < -1e-9) or np.any(s_int > 1 + 1e-9):
                return 1.0
            if np.any(w_int < -1e-9):
                return 1.0
            w0 = 1.0 - np.sum(w_int)
            if w0 < -1e-9:
                return 1.0
            
            s = np.concatenate([[0.0], s_int, [1.0]])
            # Ensure sorted
            if np.any(np.diff(s) < -1e-9):
                return 1.0
            
            w = np.concatenate([[w0], w_int])
            
            try:
                P = _evaluate_measure(s, w)
            except Exception:
                return 1.0
            
            return -P
        
        # Bounds for interior support: [0, 1]
        # Bounds for interior weights: [0, 1] (sum will be <= 1 if w0 >= 0)
        # Actually, if sum(w_int) > 1, w0 becomes negative, which is penalized.
        # It's better to bound w_int such that their sum is likely <= 1.
        # Let's use [0, 1] for each, the penalty handles the rest.
        
        bounds = [(1e-6, 0.999)] * (n - 2) + [(1e-6, 0.999)] * (n - 1)
        
        # Multiple restarts for robustness
        n_restarts = 3 if n <= 4 else 2
        
        for restart in range(n_restarts):
            if time_left() < 30:
                break
            
            # Initialize near the best solution found so far if possible
            if restart == 0 and best_support is not None and len(best_support) == n:
                s_init = best_support[1:-1]
                w_init = best_weights[1:]
                x0 = np.concatenate([s_init, w_init])
            elif restart == 0:
                # Initialize with a reasonable spread
                s_init = np.linspace(0.1, 0.9, n - 2)
                w_init = np.ones(n - 1) / n
                x0 = np.concatenate([s_init, w_init])
            else:
                # Random initialization
                s_init = np.sort(np.random.uniform(0.05, 0.95, n - 2))
                w_init = np.random.dirichlet(np.ones(n - 1)) * 0.9
                x0 = np.concatenate([s_init, w_init])
            
            try:
                result = differential_evolution(
                    objective,
                    bounds,
                    x0=x0,
                    seed=42 + restart * 100,
                    maxiter=200,
                    popsize=15,
                    tol=1e-8,
                    polish=True,
                    init='sobol' if restart == 0 else 'latinhypercube'
                )
            except Exception:
                continue
            
            if time_left() < 10:
                break
            
            x_opt = result.x
            n_int_s = n - 2
            n_int_w = n - 1
            s_int = x_opt[:n_int_s]
            w_int = x_opt[n_int_s:]
            
            # Validate
            if np.any(s_int < -1e-9) or np.any(s_int > 1 + 1e-9):
                continue
            if np.any(w_int < -1e-9):
                continue
            w0 = 1.0 - np.sum(w_int)
            if w0 < -1e-9:
                continue
            
            s = np.concatenate([[0.0], s_int, [1.0]])
            if np.any(np.diff(s) < -1e-9):
                continue
            
            w = np.concatenate([[w0], w_int])
            
            score = _evaluate_measure(s, w)
            update_best(s, w, score)
        
        # Local refinement with SLSQP on the best solution of this size
        if best_support is not None and len(best_support) == n and time_left() > 30:
            def slsqp_objective(params):
                n_int_s = n - 2
                n_int_w = n - 1
                s_int = params[:n_int_s]
                w_int = params[n_int_s:]
                
                if np.any(s_int < -1e-9) or np.any(s_int > 1 + 1e-9):
                    return 1e10
                if np.any(w_int < -1e-9):
                    return 1e10
                w0 = 1.0 - np.sum(w_int)
                if w0 < -1e-9:
                    return 1e10
                
                s = np.concatenate([[0.0], s_int, [1.0]])
                if np.any(np.diff(s) < -1e-9):
                    return 1e10
                
                w = np.concatenate([[w0], w_int])
                
                try:
                    P = _evaluate_measure(s, w)
                except Exception:
                    return 1e10
                
                return -P
            
            s_init = best_support[1:-1]
            w_init = best_weights[1:]
            x0 = np.concatenate([s_init, w_init])
            
            # Constraints for SLSQP:
            # s_int sorted: s_int[i] - s_int[i-1] >= 0 (with 0 and 1 boundaries)
            # w_int >= 0, w0 >= 0
            
            cons = []
            # s_int[0] >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=0: x[i]})
            # s_int[-1] <= 1
            cons.append({'type': 'ineq', 'fun': lambda x, i=n-3: 1.0 - x[i]})
            # s_int sorted
            for i in range(1, n - 2):
                cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[i-1]})
            
            # w_int >= 0
            for i in range(n - 1):
                idx = (n - 2) + i
                cons.append({'type': 'ineq', 'fun': lambda x, i=idx: x[i]})
            # w0 >= 0 => 1 - sum(w_int) >= 0
            cons.append({'type': 'ineq', 'fun': lambda x: 1.0 - np.sum(x[n-2:])})
            
            try:
                result_slq = minimize(
                    slsqp_objective,
                    x0,
                    method='SLSQP',
                    constraints=cons,
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
            except Exception:
                continue
            
            if time_left() < 5:
                break
            
            x_final = result_slq.x
            n_int_s = n - 2
            n_int_w = n - 1
            s_int = x_final[:n_int_s]
            w_int = x_final[n_int_s:]
            
            if np.any(s_int < -1e-9) or np.any(s_int > 1 + 1e-9):
                continue
            if np.any(w_int < -1e-9):
                continue
            w0 = 1.0 - np.sum(w_int)
            if w0 < -1e-9:
                continue
            
            s = np.concatenate([[0.0], s_int, [1.0]])
            if np.any(np.diff(s) < -1e-9):
                continue
            
            w = np.concatenate([[w0], w_int])
            
            score = _evaluate_measure(s, w)
            update_best(s, w, score)

    # Final Nelder-Mead polish on the overall best solution
    if best_support is not None and time_left() > 30:
        n_best = len(best_support)
        
        def nm_objective(params):
            n_int_s = n_best - 2
            n_int_w = n_best - 1
            s_int = params[:n_int_s]
            w_int = params[n_int_s:]
            
            if np.any(s_int < -1e-9) or np.any(s_int > 1 + 1e-9):
                return 1e10
            if np.any(w_int < -1e-9):
                return 1e10
            w0 = 1.0 - np.sum(w_int)
            if w0 < -1e-9:
                return 1e10
            
            s = np.concatenate([[0.0], s_int, [1.0]])
            if np.any(np.diff(s) < -1e-9):
                return 1e10
            
            w = np.concatenate([[w0], w_int])
            
            try:
                P = _evaluate_measure(s, w)
            except Exception:
                return 1e10
            
            return -P
        
        s_init = best_support[1:-1]
        w_init = best_weights[1:]
        x0 = np.concatenate([s_init, w_init])
        
        try:
            result_nm = minimize(
                nm_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
        except Exception:
            pass
        else:
            x_final = result_nm.x
            n_int_s = n_best - 2
            n_int_w = n_best - 1
            s_int = x_final[:n_int_s]
            w_int = x_final[n_int_s:]
            
            if np.any(s_int < -1e-9) or np.any(s_int > 1 + 1e-9):
                pass
            elif np.any(w_int < -1e-9):
                pass
            else:
                w0 = 1.0 - np.sum(w_int)
                if w0 < -1e-9:
                    pass
                else:
                    s = np.concatenate([[0.0], s_int, [1.0]])
                    if np.any(np.diff(s) < -1e-9):
                        pass
                    else:
                        w = np.concatenate([[w0], w_int])
                        score = _evaluate_measure(s, w)
                        update_best(s, w, score)

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
