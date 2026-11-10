import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import softmax

def _compute_P(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] for a discrete measure.
    support: 1D array of shape (n,)
    weights: 1D array of shape (n,)
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    # Normalize weights just in case
    total_w = w.sum()
    if total_w <= 1e-12:
        return 0.0
    w = w / total_w
    
    # 1. Compute all pairwise sums for X1 + X2
    ss = s[:, None] + s[None, :]
    ww = w[:, None] * w[None, :]
    
    # Flatten and sort
    pair_sums = ss.flatten()
    pair_weights = ww.flatten()
    
    order = np.argsort(pair_sums, kind='stable')
    sorted_sums = pair_sums[order]
    sorted_weights = pair_weights[order]
    cum_weights = np.cumsum(sorted_weights)
    
    # 2. Compute P = sum_{k,l} w_k w_l F(2*s_l - s_k)
    thresholds = 2.0 * s[None, :] - s[:, None]
    
    # For each threshold, find cumulative weight of pair_sums < threshold
    idxs = np.searchsorted(sorted_sums, thresholds, side='left')
    idxs_clipped = np.clip(idxs - 1, 0, len(cum_weights) - 1)
    probs = np.where(idxs > 0, cum_weights[idxs_clipped], 0.0)
    
    # P = sum_{k,l} w_k w_l probs[k,l]
    P = np.sum(w[:, None] * w[None, :] * probs)
    
    return float(P)

def _objective_joint(params, n):
    """
    Objective function for joint optimization of support and weights.
    params: [s_1, s_2, ..., s_n, log_w_1, log_w_2, ..., log_w_n]
    Returns -P (to minimize)
    """
    s_raw = params[:n]
    log_w = params[n:]
    
    # Support: use softplus to ensure non-negative
    s = np.log1p(np.exp(s_raw))
    
    # Weights: use softmax to ensure they sum to 1 and are positive
    w = softmax(log_w)
    
    P = _compute_P(s, w)
    return -P

def _refine_support_weights(support, weights, max_iter=2000, ftol=1e-12):
    """
    Refine both support and weights using L-BFGS-B.
    Returns (support, weights, score)
    """
    n = len(support)
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    # Convert to internal parameters
    # Support: invert softplus: s = log(1 + exp(x)) => x = log(expm1(s))
    # For s close to 0, expm1(s) ~ s, so x ~ log(s)
    # For numerical stability, clip s to [1e-10, large]
    s_clipped = np.clip(s, 1e-10, 1e6)
    s_params = np.log(np.expm1(s_clipped))
    
    # Weights: convert to log-softmax space
    # w = softmax(log_w) => log_w = log(w) (up to a constant)
    w_clipped = np.clip(w, 1e-10, 1.0)
    log_w_params = np.log(w_clipped)
    
    x0 = np.concatenate([s_params, log_w_params])
    
    try:
        result = minimize(
            _objective_joint, x0, args=(n,),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-8}
        )
        
        s_opt_raw = result.x[:n]
        log_w_opt = result.x[n:]
        
        s_opt = np.log1p(np.exp(s_opt_raw))
        w_opt = softmax(log_w_opt)
        
        # Ensure validity
        s_opt = np.maximum(s_opt, 0.0)
        w_opt = np.maximum(w_opt, 0.0)
        w_sum = w_opt.sum()
        if w_sum <= 1e-12:
            return support, weights, _compute_P(support, weights)
        w_opt = w_opt / w_sum
        
        score = _compute_P(s_opt, w_opt)
        return s_opt, w_opt, score
        
    except Exception:
        return support, weights, _compute_P(support, weights)

def _optimize_weights_on_grid(grid, w_init=None, max_iter=200, step_size=0.1):
    """
    Optimize weights on a fixed grid using projected gradient ascent on the simplex.
    This treats the measure as a piecewise-constant density on the grid.
    """
    m = len(grid)
    if w_init is None:
        # Initialize with a distribution favoring the condition
        # We want X1+X2+X3 < 2*X4, so X4 large, X1,X2,X3 small
        # This suggests putting weight on both low and high ends, or specifically
        # making X4 likely to be large relative to the sum of three others.
        # A bimodal distribution often works well: mass at 0 and mass at 1.
        # Let's try a mix.
        w = np.ones(m)
        # Boost the right side
        w = w * np.linspace(1, 2, m)
        w = w / w.sum()
    else:
        w = np.array(w_init, dtype=np.float64)
        w = np.maximum(w, 0)
        w_sum = w.sum()
        if w_sum <= 1e-12:
            w = np.ones(m) / m
        else:
            w = w / w_sum
    
    s = grid
    best_w = w.copy()
    best_P = _compute_P(s, w)
    
    for it in range(max_iter):
        # Compute gradient numerically
        # P(w) is quadratic in w? No, P = sum w_k w_l F_kl.
        # F_kl depends on s, which is fixed here.
        # So P is a quadratic form w^T A w where A_kl = F_kl.
        # Wait, F_kl is the probability that X1+X2 < 2s_l - s_k.
        # F_kl = sum_{i,j: s_i+s_j < 2s_l-s_k} w_i w_j.
        # So P = sum_{k,l} w_k w_l sum_{i,j: cond} w_i w_j.
        # This is a 4th order polynomial in w.
        # So we need numerical gradients or analytical.
        # Numerical is safer for implementation speed in this context.
        
        grad = np.zeros(m)
        eps = 1e-5
        for i in range(m):
            w_plus = w.copy()
            w_plus[i] += eps
            w_plus = np.maximum(w_plus, 0)
            w_plus_sum = w_plus.sum()
            if w_plus_sum > 0:
                w_plus = w_plus / w_plus_sum
            else:
                w_plus = np.ones(m) / m
            
            w_minus = w.copy()
            w_minus[i] -= eps
            w_minus = np.maximum(w_minus, 0)
            w_minus_sum = w_minus.sum()
            if w_minus_sum > 0:
                w_minus = w_minus / w_minus_sum
            else:
                w_minus = np.ones(m) / m
            
            f_plus = _compute_P(s, w_plus)
            f_minus = _compute_P(s, w_minus)
            
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        # Project gradient onto simplex tangent space?
        # Or just do gradient step and project.
        w_new = w + step_size * grad
        w_new = np.maximum(w_new, 0)
        w_new_sum = w_new.sum()
        if w_new_sum <= 1e-12:
            w_new = np.ones(m) / m
        else:
            w_new = w_new / w_new_sum
        
        P_new = _compute_P(s, w_new)
        
        if P_new > best_P:
            best_P = P_new
            best_w = w_new.copy()
            w = w_new
            # Decay step size if we made progress? Or keep constant?
            # Let's decay slightly to converge.
            step_size *= 0.98
        else:
            # Reduce step size if no progress
            step_size *= 0.8
            # If step size is too small, break
            if step_size < 1e-6:
                break
    
    return best_w, best_P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    best_support = None
    best_weights = None
    best_score = -1.0
    
    def update_best(s, w, score):
        nonlocal best_support, best_weights, best_score
        if score > best_score:
            best_support = s.copy()
            best_weights = w.copy()
            best_score = score
    
    np.random.seed(42)
    
    # Strategy 1: Grid-based weight optimization (Variational approach)
    # Use a fine grid and optimize weights.
    # Then, refine the support points where mass is concentrated.
    
    grid_sizes = [64, 128, 256]
    
    for g_size in grid_sizes:
        if time.time() > deadline - 120.0:
            break
            
        # Grid range: [0, 10] is a good start.
        # The condition X1+X2+X3 < 2*X4 implies X4 should be larger than average.
        # If support is [0,1], P is often around 0.5?
        # Let's try [0, 1] and [0, 5].
        
        for s_max in [1.0, 5.0, 10.0]:
            if time.time() > deadline - 120.0:
                break
                
            grid = np.linspace(0, s_max, g_size)
            
            # Try a few initial weight distributions
            inits = []
            # 1. Uniform
            inits.append(np.ones(g_size))
            # 2. Right-skewed (favoring larger X4)
            inits.append(np.linspace(1, 3, g_size))
            # 3. Bimodal (mass at 0 and s_max)
            bimodal = np.zeros(g_size)
            bimodal[:g_size//10] = 1.0
            bimodal[-g_size//10:] = 1.0
            inits.append(bimodal)
            # 4. Left-skewed (favoring small X1,X2,X3) - actually same as right-skewed for X4?
            # No, X1,X2,X3 are also drawn from same dist.
            # If dist is concentrated at 0, X1+X2+X3 ~ 0, X4 ~ 0. 0 < 0 is false.
            # If dist is concentrated at 1, 3 < 2 is false.
            # We need spread.
            
            for w_init in inits:
                if time.time() > deadline - 120.0:
                    break
                
                w_opt, P_opt = _optimize_weights_on_grid(grid, w_init, max_iter=100, step_size=0.5)
                
                if P_opt > best_score:
                    # Extract support: keep points with significant weight
                    # This reduces n for the next phase
                    threshold = 1e-4
                    mask = w_opt > threshold
                    s_candidate = grid[mask]
                    w_candidate = w_opt[mask]
                    w_candidate = w_candidate / w_candidate.sum()
                    
                    update_best(s_candidate, w_candidate, P_opt)
                    
                    # Refine support and weights jointly using L-BFGS-B
                    if time.time() < deadline - 60.0:
                        s_refined, w_refined, score_refined = _refine_support_weights(
                            s_candidate, w_candidate, max_iter=1000
                        )
                        update_best(s_refined, w_refined, score_refined)
                        
                        # Further refinement
                        if time.time() < deadline - 30.0:
                            s_refined2, w_refined2, score_refined2 = _refine_support_weights(
                                s_refined, w_refined, max_iter=1000
                            )
                            update_best(s_refined2, w_refined2, score_refined2)
    
    # Strategy 2: Random restarts with joint optimization (from current program)
    # This helps if the grid approach gets stuck in local optima.
    
    for n in [3, 4, 5, 6]:
        if time.time() > deadline - 60.0:
            break
        
        n_restarts = 20
        for r in range(n_restarts):
            if time.time() > deadline - 60.0:
                break
            
            s_init = np.random.uniform(0, 10, n)
            w_init = np.random.dirichlet(np.ones(n) * 0.5)
            
            s_refined, w_refined, score = _refine_support_weights(s_init, w_init, max_iter=500)
            update_best(s_refined, w_refined, score)
            
            if score > best_score - 0.01:
                s_refined2, w_refined2, score2 = _refine_support_weights(s_refined, w_refined, max_iter=1000)
                update_best(s_refined2, w_refined2, score2)
    
    # Final refinement
    if best_support is not None and time.time() < deadline - 30.0:
        s_final, w_final, score_final = _refine_support_weights(
            best_support, best_weights, max_iter=2000, ftol=1e-14
        )
        update_best(s_final, w_final, score_final)
        
    # Fallback
    if best_support is None:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        
    return best_support.astype(np.float64), best_weights.astype(np.float64)
