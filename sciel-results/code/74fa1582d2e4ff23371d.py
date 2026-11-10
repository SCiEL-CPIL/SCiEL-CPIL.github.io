# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _compute_P_fast(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] exactly for a discrete measure.
    P = sum_{k,l} w_k * w_l * F(2*s_l - s_k)
    where F(t) = P[X1+X2 < t] = sum_{i,j: s_i+s_j < t} w_i * w_j
    Uses sorted support and prefix sums for O(n^2 log n) computation.
    """
    n = len(support)
    if n == 0:
        return 0.0
    w = np.asarray(weights, dtype=np.float64)
    s = np.asarray(support, dtype=np.float64)
    total = w.sum()
    if total <= 1e-9:
        return 0.0
    w = w / total
    
    idx = np.argsort(s)
    s = s[idx]
    w = w[idx]
    
    # Build sorted pair sums with their weights
    pair_sums = s[:, None] + s[None, :]
    pair_weights = w[:, None] * w[None, :]
    
    # Flatten and sort by pair sum
    flat_sums = pair_sums.ravel()
    flat_wts = pair_weights.ravel()
    order = np.argsort(flat_sums, kind='stable')
    sorted_sums = flat_sums[order]
    sorted_wts = flat_wts[order]
    cum_wts = np.concatenate([[0.0], np.cumsum(sorted_wts)])
    
    # F(t) = sum of pair_weights where pair_sum < t
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            pos = np.searchsorted(sorted_sums, threshold, side='left')
            inner = cum_wts[pos]
            P += w[k] * w[l] * inner
    return P


def _pareto_support_weighted(alpha_mass, alpha_tail, scale, n_tail, conc):
    """
    Generate support with non-uniform tail weights.
    conc > 0 concentrates weight toward larger values.
    """
    u_vals = np.linspace(0, 1, n_tail + 2)[1:-1]
    u_vals = np.clip(u_vals, 1e-12, 1 - 1e-12)
    x_vals = scale * (1.0 - u_vals) ** (-1.0 / alpha_tail)
    
    raw_w = (1.0 - u_vals) ** conc
    raw_w = raw_w / raw_w.sum()
    w_tail = alpha_mass * raw_w
    
    support = np.concatenate([[0.0], x_vals])
    weights = np.concatenate([[1.0 - alpha_mass], w_tail])
    return support, weights


def _optimize_weights_fixed_support(support, max_iter=500):
    """
    Optimize weights for a fixed support to maximize P.
    Uses log-parameterization and L-BFGS-B.
    """
    n = len(support)
    
    def _neg_P(log_w):
        w = np.exp(log_w)
        w = w / w.sum()
        return -_compute_P_fast(support, w)
    
    best_val = np.inf
    best_log_w = None
    
    inits = [
        np.zeros(n) + np.log(1.0/n),
        np.log(np.concatenate([[0.5], np.full(n-1, 0.5/(n-1))])) if n > 1 else np.array([0.0]),
        np.log(np.concatenate([[0.3], np.full(n-1, 0.7/(n-1))])) if n > 1 else np.array([0.0]),
        np.log(np.concatenate([[0.7], np.full(n-1, 0.3/(n-1))])) if n > 1 else np.array([0.0]),
        np.linspace(np.log(0.1), np.log(1.0), n),
    ]
    
    for init in inits:
        try:
            res = minimize(_neg_P, init, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_log_w = res.x
        except Exception:
            continue
    
    if best_log_w is None:
        w = np.full(n, 1.0/n)
        return w, _compute_P_fast(support, w)
    
    w = np.exp(best_log_w)
    w = w / w.sum()
    return w, -best_val


def _joint_optimize(n_points, time_budget=30.0):
    """
    Jointly optimize support positions and weights.
    Support is parameterized as: s_0 = 0, s_i = exp(log_positions[i-1]) for i >= 1.
    Weights are parameterized in log space.
    """
    start = time.time()
    
    def _neg_P(params):
        if time.time() - start > time_budget:
            return 1.0
        log_pos = params[:n_points-1]
        log_w = params[n_points-1:]
        
        s = np.concatenate([[0.0], np.exp(log_pos)])
        s = np.sort(s)
        w = np.exp(log_w)
        w = w / w.sum()
        
        if np.any(~np.isfinite(s)) or np.any(~np.isfinite(w)):
            return 1.0
        if np.any(s < -1e-9) or np.any(w < -1e-9):
            return 1.0
        
        return -_compute_P_fast(s, w)
    
    best_val = np.inf
    best_params = None
    
    n_inits = min(8, max(3, int(time_budget / 5)))
    for i in range(n_inits):
        if time.time() - start > time_budget * 0.8:
            break
        
        if i == 0:
            log_pos_init = np.linspace(0.0, 0.5, n_points-1)
            log_w_init = np.full(n_points-1, np.log(0.7/(n_points-1)))
            log_w_init = np.concatenate([[np.log(0.3)], log_w_init])
        elif i == 1:
            log_pos_init = np.linspace(-0.5, 0.5, n_points-1)
            log_w_init = np.full(n_points, np.log(1.0/n_points))
        elif i == 2:
            log_pos_init = np.linspace(0.0, 1.0, n_points-1)
            log_w_init = np.concatenate([[np.log(0.5)], np.full(n_points-1, np.log(0.5/(n_points-1)))])
        else:
            rng = np.random.RandomState(i * 123 + 456)
            log_pos_init = rng.uniform(-0.5, 1.0, size=n_points-1)
            log_w_init = rng.uniform(-2.0, 0.0, size=n_points)
        
        x0 = np.concatenate([log_pos_init, log_w_init])
        
        try:
            res = minimize(_neg_P, x0, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            continue
    
    if best_params is None:
        return None, None, 0.0
    
    log_pos = best_params[:n_points-1]
    log_w = best_params[n_points-1:]
    s = np.concatenate([[0.0], np.exp(log_pos)])
    s = np.sort(s)
    w = np.exp(log_w)
    w = w / w.sum()
    
    return s, w, -best_val


def _polish(support, weights, time_budget=20.0):
    """
    High-precision polish: jointly optimize support and weights with tight tolerances.
    """
    start = time.time()
    n = len(support)
    
    def _neg_P(params):
        if time.time() - start > time_budget:
            return 1.0
        delta_s = params[:n-1]
        delta_w = params[n-1:]
        
        s = np.copy(support)
        s[1:] = s[1:] * np.exp(delta_s)
        s = np.sort(s)
        
        w = weights * np.exp(delta_w)
        w = w / w.sum()
        
        if np.any(~np.isfinite(s)) or np.any(~np.isfinite(w)):
            return 1.0
        if np.any(s < -1e-9) or np.any(w < -1e-9):
            return 1.0
        
        return -_compute_P_fast(s, w)
    
    x0 = np.zeros(2 * n - 1)
    try:
        res = minimize(_neg_P, x0, method='L-BFGS-B',
                      options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12})
        if res.fun < 0:
            delta_s = res.x[:n-1]
            delta_w = res.x[n-1:]
            s = np.copy(support)
            s[1:] = s[1:] * np.exp(delta_s)
            s = np.sort(s)
            w = weights * np.exp(delta_w)
            w = w / w.sum()
            P = -res.fun
            return s, w, P
    except Exception:
        pass
    
    return support, weights, _compute_P_fast(support, weights)


def _polish_weights(support, weights, time_budget=15.0):
    """
    Polish only the weights for a fixed support with very tight tolerances.
    """
    start = time.time()
    n = len(support)
    
    def _neg_P(log_w):
        if time.time() - start > time_budget:
            return 1.0
        w = np.exp(log_w)
        w = w / w.sum()
        return -_compute_P_fast(support, w)
    
    log_w0 = np.log(np.maximum(weights, 1e-15))
    try:
        res = minimize(_neg_P, log_w0, method='L-BFGS-B',
                      options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-14})
        if res.fun < 0:
            w = np.exp(res.x)
            w = w / w.sum()
            P = -res.fun
            return support, w, P
    except Exception:
        pass
    
    return support, weights, _compute_P_fast(support, weights)


def beat_the_average_measure() -> tuple:
    """
    Find a discrete measure maximizing P[X1+X2+X3 < 2*X4].
    
    Strategy:
    1. Analytic two-point baseline
    2. Pareto mixture with optimized parameters (main search)
    3. Joint optimization of support + weights
    4. Differential evolution for global search
    5. High-precision polish (weights-only and joint)
    """
    deadline = time.time() + 580
    best_score = -1.0
    best_support = None
    best_weights = None
    
    def _update_best(P, support, weights):
        nonlocal best_score, best_support, best_weights
        if P > best_score:
            best_score = P
            best_support = np.array(support, dtype=np.float64).copy()
            best_weights = np.array(weights, dtype=np.float64).copy()
    
    def _time_ok():
        return time.time() < deadline
    
    # ===== 1. Two-point analytic baseline =====
    q_star = (1.0 + np.sqrt(33.0)) / 16.0
    P_two = q_star * (1 - q_star)**2 * (1 + 2 * q_star)
    _update_best(P_two, np.array([0.0, 1.0]), np.array([1 - q_star, q_star]))
    
    # ===== 2. Pareto mixture search =====
    def _P_pareto(params):
        if not _time_ok():
            return -1.0
        alpha_mass, alpha_tail, scale, n_tail, conc = params
        alpha_mass = float(alpha_mass)
        alpha_tail = float(alpha_tail)
        scale = float(scale)
        n_tail = int(round(n_tail))
        conc = float(conc)
        
        if alpha_mass < 0.05 or alpha_mass > 0.95:
            return -1.0
        if alpha_tail < 0.3 or alpha_tail > 20.0:
            return -1.0
        if scale < 0.01 or scale > 1000.0:
            return -1.0
        if n_tail < 2 or n_tail > 12:
            return -1.0
        if conc < -2.0 or conc > 5.0:
            return -1.0
        
        try:
            sup, wt = _pareto_support_weighted(alpha_mass, alpha_tail, scale, n_tail, conc)
            if len(sup) < 1 or np.any(~np.isfinite(sup)) or np.any(~np.isfinite(wt)):
                return -1.0
            if np.any(sup < -1e-9) or np.any(wt < -1e-9):
                return -1.0
            if wt.sum() <= 1e-9:
                return -1.0
            return _compute_P_fast(sup, wt)
        except Exception:
            return -1.0
    
    for n_tail in [2, 3, 4, 5, 6, 8, 10]:
        if not _time_ok():
            break
        
        bounds_p = [(0.1, 0.9), (0.3, 10.0), (0.1, 100.0), 
                    (float(n_tail), float(n_tail)), (-2.0, 3.0)]
        
        try:
            res = differential_evolution(
                lambda x: -_P_pareto(x),
                bounds_p,
                maxiter=80,
                popsize=20,
                seed=42 + n_tail,
                tol=1e-8,
                polish=True,
                init='sobol'
            )
            if -res.fun > -0.5:
                alpha_mass, alpha_tail, scale, _, conc = res.x
                sup, wt = _pareto_support_weighted(float(alpha_mass), float(alpha_tail), 
                                                   float(scale), n_tail, float(conc))
                wt = wt / wt.sum()
                P_val = _compute_P_fast(sup, wt)
                _update_best(P_val, sup, wt)
        except Exception:
            pass
        
        for am0 in [0.3, 0.5, 0.7]:
            for at0 in [0.5, 1.0, 2.0, 4.0]:
                for sc0 in [0.5, 1.0, 3.0, 10.0]:
                    for c0 in [0.0, 0.5, 1.0]:
                        if not _time_ok():
                            break
                        x0 = np.array([am0, at0, sc0, float(n_tail), c0])
                        try:
                            res = minimize(lambda x: -_P_pareto(x), x0,
                                         method='L-BFGS-B',
                                         bounds=[(0.1, 0.9), (0.3, 10.0), (0.1, 100.0),
                                                 (float(n_tail), float(n_tail)), (-2.0, 3.0)],
                                         options={'maxiter': 200, 'ftol': 1e-14})
                            P_val = -res.fun
                            if P_val > -0.5:
                                alpha_mass, alpha_tail, scale, _, conc = res.x
                                sup, wt = _pareto_support_weighted(float(alpha_mass), 
                                                                   float(alpha_tail), 
                                                                   float(scale), n_tail, float(conc))
                                wt = wt / wt.sum()
                                P_val = _compute_P_fast(sup, wt)
                                _update_best(P_val, sup, wt)
                        except Exception:
                            continue
    
    # ===== 3. Joint optimization with different n =====
    for n in [3, 4, 5, 6, 7, 8]:
        if not _time_ok():
            break
        
        remaining = deadline - time.time()
        if remaining < 5:
            break
        budget = min(25.0, remaining * 0.3)
        
        s, w, P = _joint_optimize(n, time_budget=budget)
        if s is not None and P > -0.5:
            _update_best(P, s, w)
    
    # ===== 4. Structured families =====
    for n in [3, 4, 5, 6]:
        if not _time_ok():
            break
        for log_r in np.linspace(-1.0, 0.5, 15):
            if not _time_ok():
                break
            r = np.exp(log_r)
            sup = np.concatenate([[0.0], r ** np.arange(1, n)])
            if sup[-1] > 1000:
                continue
            w, P = _optimize_weights_fixed_support(sup, max_iter=300)
            if P > -0.5:
                _update_best(P, sup, w)
    
    for n in [4, 5, 6, 7]:
        if not _time_ok():
            break
        for lo in [-2.0, -1.0, -0.5]:
            for hi in [0.0, 0.5, 1.0]:
                if not _time_ok():
                    break
                sup = np.concatenate([[0.0], np.exp(np.linspace(lo, hi, n-1))])
                if sup[-1] > 1000:
                    continue
                w, P = _optimize_weights_fixed_support(sup, max_iter=300)
                if P > -0.5:
                    _update_best(P, sup, w)
    
    # ===== 5. Differential evolution on compact parameterization =====
    if _time_ok():
        n_de = 5
        
        def _neg_P_de(params):
            if not _time_ok():
                return 1.0
            a, b = params[0], params[1]
            log_w = params[2:]
            
            s = np.concatenate([[0.0], np.exp(a * np.arange(1, n_de) + b)])
            s = np.sort(s)
            w = np.exp(log_w)
            w = w / w.sum()
            
            if np.any(~np.isfinite(s)) or np.any(~np.isfinite(w)):
                return 1.0
            if np.any(s < -1e-9) or np.any(w < -1e-9):
                return 1.0
            if s[-1] > 1000:
                return 1.0
            
            return -_compute_P_fast(s, w)
        
        bounds_de = [(-0.5, 0.5), (-2.0, 1.0)] + [(-3.0, 0.5)] * (n_de - 1)
        try:
            res = differential_evolution(_neg_P_de, bounds_de,
                                         maxiter=100, popsize=25,
                                         seed=999, tol=1e-8, polish=True)
            if res.fun < 0:
                a, b = res.x[0], res.x[1]
                log_w = res.x[2:]
                s = np.concatenate([[0.0], np.exp(a * np.arange(1, n_de) + b)])
                s = np.sort(s)
                w = np.exp(log_w)
                w = w / w.sum()
                P_val = -res.fun
                _update_best(P_val, s, w)
        except Exception:
            pass
    
    # ===== 6. Final polish: weights-only then joint =====
    if _time_ok() and best_support is not None:
        remaining = deadline - time.time()
        if remaining > 10:
            # First: polish weights only (faster, more iterations possible)
            budget_w = min(20.0, remaining * 0.4)
            s, w, P = _polish_weights(best_support, best_weights, time_budget=budget_w)
            if P > -0.5:
                _update_best(P, s, w)
        
        # Then: joint polish
        if _time_ok():
            remaining = deadline - time.time()
            if remaining > 10:
                budget_j = min(30.0, remaining * 0.7)
                s, w, P = _polish(best_support, best_weights, time_budget=budget_j)
                if P > -0.5:
                    _update_best(P, s, w)
    
    # ===== Final validation =====
    if best_support is None or best_weights is None:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([1 - q_star, q_star], dtype=np.float64)
    
    best_support = np.asarray(best_support, dtype=np.float64).flatten()
    best_weights = np.asarray(best_weights, dtype=np.float64).flatten()
    
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    if best_weights.sum() <= 1e-9:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([1 - q_star, q_star], dtype=np.float64)
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
