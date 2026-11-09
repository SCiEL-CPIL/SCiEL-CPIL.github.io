import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

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
    
    # Ensure sorted
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


def _param_to_measure(params, n):
    """
    Convert parameter vector to (support, weights).
    params[0:n-1] are gaps (non-negative), support[0]=0, support[i]=sum(gaps[:i])
    params[n:2*n-1] are log-weights (softmax)
    """
    gaps = np.maximum(params[0:n-1], 0.0)
    support = np.zeros(n)
    if n > 1:
        support[1:] = np.cumsum(gaps)
    
    log_w = params[n-1:2*n-1]
    log_w_shifted = log_w - np.max(log_w)
    w = np.exp(log_w_shifted)
    w_sum = np.sum(w)
    if w_sum < 1e-15:
        w = np.ones(n) / n
    else:
        w = w / w_sum
    
    return support, w


def _objective(params, n, deadline):
    """Objective function for optimization."""
    if time.time() > deadline:
        return 1e10
    
    support, w = _param_to_measure(params, n)
    
    if np.any(np.isnan(support)) or np.any(np.isnan(w)):
        return 1e10
    
    P = _evaluate_measure(support, w)
    return -P


def _local_optimize(support_init, weights_init, n, deadline, max_iter=5000):
    """Local optimization using L-BFGS-B."""
    # Convert to parameter space
    gaps = np.zeros(n-1)
    if n > 1:
        gaps[0] = support_init[1] - support_init[0]
        for i in range(1, n-1):
            gaps[i] = support_init[i+1] - support_init[i]
    
    log_w = np.log(np.maximum(weights_init, 1e-15))
    
    params_init = np.concatenate([gaps, log_w])
    
    # Bounds: gaps >= 0, log_w in [-20, 20]
    bounds = [(0, 50.0)] * (n-1) + [(-20.0, 20.0)] * n
    
    best_params = params_init.copy()
    best_obj = _objective(params_init, n, deadline)
    
    # Try L-BFGS-B
    try:
        result = minimize(
            _objective,
            params_init,
            args=(n, deadline),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
    except Exception:
        pass
    
    # Try Nelder-Mead as backup
    try:
        result2 = minimize(
            _objective,
            best_params,
            args=(n, deadline),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if result2.fun < best_obj:
            best_obj = result2.fun
            best_params = result2.x
    except Exception:
        pass
    
    support_final, w_final = _param_to_measure(best_params, n)
    score = -best_obj
    return support_final, w_final, score


def _generate_initial_candidates(n, deadline):
    """Generate diverse initial candidates for n-point distributions."""
    candidates = []
    
    # 1. Uniformly spaced with random weights
    for trial in range(20):
        support = np.linspace(0, 1.0 + np.random.random(), n)
        weights = np.random.dirichlet(np.ones(n))
        candidates.append((support, weights))
    
    # 2. Geometric spacing
    for ratio in [0.5, 0.7, 0.8, 1.2, 1.5]:
        support = np.array([0.0])
        for i in range(1, n):
            support.append(support[-1] * ratio + 0.1)
        weights = np.random.dirichlet(np.ones(n))
        candidates.append((support, weights))
    
    # 3. Concentrated at 0 and 1
    support = np.zeros(n)
    support[-1] = 1.0
    if n > 1:
        support[-2] = 0.5
    weights = np.zeros(n)
    weights[0] = 0.3
    weights[-1] = 0.5
    weights[1:-1] = 0.2 / max(1, n-2)
    candidates.append((support, weights))
    
    # 4. Based on 2-point optimum extended
    p_opt = (1.0 + np.sqrt(33.0)) / 16.0
    support = np.zeros(n)
    support[-1] = 1.0
    weights = np.zeros(n)
    weights[0] = 1.0 - p_opt
    weights[-1] = p_opt
    if n > 2:
        # Add small mass at intermediate points
        for i in range(1, n-1):
            support[i] = i / (n - 1)
            weights[i] = 0.01
            weights[0] -= 0.01
        weights = np.maximum(weights, 0.001)
        weights /= np.sum(weights)
    candidates.append((support, weights))
    
    # 5. Random with structure: one large, rest small
    for trial in range(10):
        support = np.sort(np.concatenate([np.random.uniform(0, 0.3, n-1), [np.random.uniform(0.5, 2.0)]]))
        weights = np.random.dirichlet(np.ones(n))
        # Bias weight toward the large support point
        weights[-1] += 0.3
        weights /= np.sum(weights)
        candidates.append((support, weights))
    
    # 6. Symmetric-ish: mass at 0, mid, high
    if n >= 3:
        for a in [0.3, 0.5, 0.7]:
            support = np.zeros(n)
            support[0] = 0.0
            support[-1] = 1.0
            for i in range(1, n-1):
                support[i] = a * i / (n-1)
            weights = np.array([0.4, 0.3, 0.3] if n == 3 else np.ones(n) / n)
            if n > 3:
                weights = np.concatenate([np.array([0.4, 0.3]), np.ones(n-3) * 0.3 / (n-3), np.array([0.3])])
                weights /= np.sum(weights)
            candidates.append((support, weights))
    
    # 7. Fine grid near 0 with one outlier
    for outlier_pos in [1.0, 2.0, 3.0]:
        support = np.zeros(n)
        support[-1] = outlier_pos
        if n > 1:
            support[1:-1] = np.linspace(0.01, 0.3, n-2)
        weights = np.zeros(n)
        weights[0] = 0.5
        weights[-1] = 0.4
        if n > 2:
            weights[1:-1] = 0.1 / (n-2)
        candidates.append((support, weights))
    
    return candidates


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
            best_support = np.array(sup, dtype=np.float64)
            best_weights = np.array(wts, dtype=np.float64)
    
    def time_left():
        return deadline - time.time()
    
    # Start with analytic 2-point optimum
    p_opt_two = (1.0 + np.sqrt(33.0)) / 16.0
    support_two = np.array([0.0, 1.0])
    weights_two = np.array([1.0 - p_opt_two, p_opt_two])
    score_two = _evaluate_measure(support_two, weights_two)
    update_best(support_two, weights_two, score_two)
    
    # Try various n from 2 to 8
    for n in range(2, 9):
        if time_left() < 40:
            break
        
        # Generate initial candidates
        candidates = _generate_initial_candidates(n, deadline)
        
        # Also add the current best (resized) as a candidate
        if best_support is not None:
            if len(best_support) < n:
                # Pad with small values
                pad_support = np.concatenate([best_support, np.linspace(best_support[-1]+0.1, best_support[-1]+0.5, n-len(best_support))])
                pad_weights = np.concatenate([best_weights, np.ones(n-len(best_support)) * 0.01])
                pad_weights /= np.sum(pad_weights)
                candidates.append((pad_support, pad_weights))
            elif len(best_support) > n:
                # Keep top n by weight
                idx = np.argsort(best_weights)[::-1][:n]
                candidates.append((best_support[idx], best_weights[idx] / np.sum(best_weights[idx])))
            else:
                candidates.append((best_support.copy(), best_weights.copy()))
        
        # Evaluate all candidates and pick the best few for local optimization
        candidate_scores = []
        for sup, wts in candidates:
            if time.time() > deadline - 20:
                break
            sc = _evaluate_measure(sup, wts)
            candidate_scores.append((sc, sup, wts))
        
        candidate_scores.sort(key=lambda x: -x[0])
        
        # Local optimize the top candidates
        num_to_optimize = min(8, len(candidate_scores))
        for sc, sup, wts in candidate_scores[:num_to_optimize]:
            if time_left() < 20:
                break
            
            sup_opt, wts_opt, score_opt = _local_optimize(sup, wts, n, deadline)
            update_best(sup_opt, wts_opt, score_opt)
        
        # Also try a few random restarts
        for _ in range(5):
            if time_left() < 15:
                break
            support_rand = np.sort(np.random.uniform(0, 1.5, n))
            weights_rand = np.random.dirichlet(np.ones(n))
            sup_opt, wts_opt, score_opt = _local_optimize(support_rand, weights_rand, n, deadline)
            update_best(sup_opt, wts_opt, score_opt)
    
    # Final polish on the best solution found
    if best_support is not None and time_left() > 10:
        n_best = len(best_support)
        sup_final, wts_final, score_final = _local_optimize(
            best_support, best_weights, n_best, deadline, max_iter=10000
        )
        update_best(sup_final, wts_final, score_final)
    
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
