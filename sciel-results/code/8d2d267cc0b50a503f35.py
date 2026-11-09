# EVOLVE-BLOCK-START
import numpy as np
import time


def _union_area(x, M=20000):
    """
    Compute the union area of 64 triangles.
    
    x: shape (64,) - base positions
    M: number of quadrature points
    Returns: area (scalar)
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    
    # L_i(y) = x_i + a_i * y
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    # Sort L along axis=1
    order = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, order, axis=1)
    
    # Consecutive differences
    diffs = np.diff(L_sorted, axis=1)
    
    # union_len at each y
    union_len = w + np.sum(np.minimum(diffs, w[:, np.newaxis]), axis=1)
    area = np.mean(union_len)
    
    return area


def _soft_min_area(x, M=2000, beta=50.0):
    """
    Compute a smooth surrogate for the union area using soft-min.
    
    Uses soft-min of (diffs, w) via log-sum-exp for differentiability.
    
    x: shape (64,) - base positions
    M: number of quadrature points
    beta: inverse temperature for soft-min (higher = sharper)
    Returns: (area, grad) where grad has shape (64,)
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    
    # L_i(y) = x_i + a_i * y, shape (M, n)
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    # Sort L along axis=1 and track permutation
    order = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, order, axis=1)
    
    # Consecutive differences
    diffs = np.diff(L_sorted, axis=1)  # shape (M, n-1)
    w_col = w[:, np.newaxis]  # shape (M, 1)
    
    # Soft-min of (diffs, w): 
    # soft_min(a, b) = -logsumexp(-beta*a, -beta*b) / beta
    # We want min(diffs, w) smoothly
    # soft_min = -logsumexp(-beta*diffs, -beta*w) / beta
    
    # For numerical stability, use the max trick
    # Let's compute: for each (m,k), soft_min(diffs[m,k], w[m])
    
    # Use a stable soft-min:
    # s = min(diffs, w)
    # soft_min = s - (1/beta) * log(1 + exp(-beta*(diffs - s)) + exp(-beta*(w - s)))
    # Actually simpler: soft_min(a,b) = -(1/beta)*log(exp(-beta*a) + exp(-beta*b))
    # But this can overflow. Use:
    # soft_min(a,b) = min(a,b) - (1/beta)*log(1 + exp(-beta*|a-b|))
    
    # Let's use the formulation:
    # soft_min(a, b) = min(a,b) - (1/beta) * log(1 + exp(-beta * |a - b|))
    
    diff_abs = np.abs(diffs - w_col)  # shape (M, n-1)
    s_min = np.minimum(diffs, w_col)  # shape (M, n-1)
    
    # For numerical stability, clip the exponent
    # exp(-beta * diff_abs) is in (0, 1]
    # log(1 + exp(-beta * diff_abs)) is well-behaved
    # When diff_abs is large, exp(-beta*diff_abs) ~ 0, log(1+0) = 0
    # When diff_abs is small, log(1 + ~1) = log(2)
    
    # Use np.log1p for stability
    exp_term = np.exp(-beta * diff_abs)
    soft_correction = (1.0 / beta) * np.log1p(exp_term)
    
    soft_min_vals = s_min - soft_correction  # shape (M, n-1)
    
    # union_len = w + sum soft_min_vals
    union_len = w + np.sum(soft_min_vals, axis=1)
    area = np.mean(union_len)
    
    # Gradient computation
    # d(soft_min(a,b))/da where a=diffs[m,k], b=w[m]
    # soft_min = s - (1/beta)*log(1+exp(-beta*|a-b|))
    # where s = min(a,b)
    
    # Case 1: a < b (so s = a)
    #   d(soft_min)/da = 1 - (1/beta)*d/da[log(1+exp(-beta*(b-a)))]
    #   = 1 - (1/beta)*[exp(-beta*(b-a)) * beta] / (1 + exp(-beta*(b-a)))
    #   = 1 - exp(-beta*(b-a)) / (1 + exp(-beta*(b-a)))
    #   = 1 / (1 + exp(-beta*(b-a)))
    #   = sigmoid(beta*(a-b))  ... wait let me redo
    #   Actually: d/da log(1+exp(-beta*(b-a))) = exp(-beta*(b-a)) * beta / (1+exp(-beta*(b-a)))
    #   So d(soft_min)/da = 1 - (1/beta)*beta*exp(-beta*(b-a))/(1+exp(-beta*(b-a)))
    #   = 1 - exp(-beta*(b-a))/(1+exp(-beta*(b-a)))
    #   = 1/(1+exp(-beta*(b-a)))
    
    # Case 2: a > b (so s = b)
    #   d(soft_min)/da = 0 - (1/beta)*d/da[log(1+exp(-beta*(a-b)))]
    #   = -(1/beta)*exp(-beta*(a-b))*(-beta)/(1+exp(-beta*(a-b)))
    #   = exp(-beta*(a-b))/(1+exp(-beta*(a-b)))
    #   = 1/(1+exp(beta*(a-b)))
    
    # Unified: d(soft_min(a,b))/da = 1/(1+exp(-beta*(a-b))) when a<=b
    #                              = 1/(1+exp(beta*(a-b))) when a>b
    # This is: sigmoid(beta*(a-b)) for a<=b, and sigmoid(-beta*(a-b)) for a>b
    # Which is just: 1/(1+exp(-beta*sign(a-b)*(a-b))) ... 
    # Actually simpler: it's the sigmoid of -beta*(a-b) when a>b, and sigmoid(beta*(a-b)) when a<=b
    # = sigmoid(-beta*|a-b| + beta*(a-b)) ... 
    # Let's just compute it directly:
    
    # da/dx_j for a = diffs[m,k] = L_sorted[m,k+1] - L_sorted[m,k]
    # = a_{order[m,k+1]} - a_{order[m,k]}  (the y-coefficient part, since x cancels in diff)
    # Wait no: L_i = x_i + a_i*y, so diffs = (x_{order[k+1]} + a_{order[k+1]}*y) - (x_{order[k]} + a_{order[k]}*y)
    # = (x_{order[k+1]} - x_{order[k]}) + (a_{order[k+1]} - a_{order[k]}) * y
    
    # So d(diffs)/dx_j:
    #   if j == order[m,k+1]: +1
    #   if j == order[m,k]: -1
    #   else: 0
    
    # Similarly for b = w[m], db/dx_j = 0 for all j
    
    # So d(soft_min)/dx_j = d(soft_min)/da * d(a)/dx_j
    
    # Let g_plus = d(soft_min)/da (the derivative w.r.t. the first arg 'a'=diffs)
    # Let g_minus = d(soft_min)/db (the derivative w.r.t. the second arg 'b'=w)
    # Since db/dx_j = 0, only g_plus matters
    
    # g_plus[m,k] = 1/(1+exp(-beta*(diffs[m,k] - w[m]))) if diffs[m,k] <= w[m]
    #              = 1/(1+exp(beta*(diffs[m,k] - w[m]))) if diffs[m,k] > w[m]
    
    # Simplify: let d = diffs[m,k] - w[m]
    # g_plus = 1/(1+exp(-beta*d)) if d <= 0
    #         = 1/(1+exp(beta*d)) if d > 0
    #         = 1/(1+exp(-beta*|d|)) ... no
    #         Actually: for d<=0: 1/(1+exp(-beta*d)) = sigmoid(beta*d)
    #                    for d>0: 1/(1+exp(beta*d)) = sigmoid(-beta*d)
    # So g_plus = sigmoid(-beta*|d|) ... no, let's verify:
    # d=-1, beta=1: g_plus = 1/(1+exp(1)) = 1/(1+e) ≈ 0.269
    # sigmoid(-beta*|d|) = sigmoid(-1) = 1/(1+exp(1)) ≈ 0.269 ✓
    # d=1, beta=1: g_plus = 1/(1+exp(1)) ≈ 0.269
    # sigmoid(-beta*|d|) = sigmoid(-1) ≈ 0.269 ✓
    # d=0: g_plus = 1/2, sigmoid(0) = 1/2 ✓
    
    # So g_plus = sigmoid(-beta * |diffs - w|)
    
    # Wait, that's not right. Let me recheck.
    # For d <= 0: g_plus = 1/(1+exp(-beta*d)) 
    #   d=-1: 1/(1+exp(1)) = 0.269
    # For d > 0: g_plus = 1/(1+exp(beta*d))
    #   d=1: 1/(1+exp(1)) = 0.269
    # So yes, g_plus = 1/(1+exp(beta*|d|))
    
    # But wait, that's symmetric. Let me recheck the derivation.
    # Case a < b: d = a - b < 0
    # g_plus = 1/(1+exp(-beta*(b-a))) = 1/(1+exp(-beta*(-d))) = 1/(1+exp(beta*d))
    # Since d<0, beta*d < 0, exp(beta*d) < 1, so this is > 1/2
    # Hmm, that contradicts. Let me redo more carefully.
    
    # soft_min(a,b) = min(a,b) - (1/beta)*log(1 + exp(-beta*|a-b|))
    
    # Let s = min(a,b), d = |a-b|
    # Case a <= b: s = a, d = b-a
    #   f = a - (1/beta)*log(1 + exp(-beta*(b-a)))
    #   df/da = 1 - (1/beta)*[exp(-beta*(b-a)) * beta] / (1 + exp(-beta*(b-a)))
    #   = 1 - exp(-beta*(b-a)) / (1 + exp(-beta*(b-a)))
    #   = 1/(1 + exp(-beta*(b-a)))
    #   Let u = b-a >= 0, so df/da = 1/(1+exp(-beta*u)) = sigmoid(beta*u)
    #   Since u = b-a, and a<=b, this is >= 1/2
    #   In terms of d=a-b=-u: df/da = sigmoid(-beta*d) where d<=0
    #   = 1/(1+exp(beta*d)) ... since d<=0, exp(beta*d)<=1
    #   Hmm, sigmoid(-beta*d) with d=-1, beta=1: sigmoid(1) = 0.731
    #   Let me verify: a=0, b=1, beta=1
    #   f = 0 - log(1+exp(-1)) = -log(1+0.368) = -0.313
    #   f(a+eps) with eps=0.01: a=0.01, b=1, s=0.01, d=0.99
    #   f = 0.01 - log(1+exp(-0.99)) = 0.01 - log(1+0.371) = 0.01 - 0.316 = -0.306
    #   df/da ≈ (−0.306−(−0.313))/0.01 = 0.007/0.01 = 0.7
    #   sigmoid(beta*u) = sigmoid(1) = 0.731 ✓ (close enough with finite diff)
    
    # Case a > b: s = b, d = a-b
    #   f = b - (1/beta)*log(1 + exp(-beta*(a-b)))
    #   df/da = 0 - (1/beta)*[exp(-beta*(a-b)) * (-beta)] / (1 + exp(-beta*(a-b)))
    #   = exp(-beta*(a-b)) / (1 + exp(-beta*(a-b)))
    #   = 1/(1 + exp(beta*(a-b)))
    #   Let v = a-b > 0: df/da = 1/(1+exp(beta*v)) = sigmoid(-beta*v)
    #   In terms of d=a-b=v>0: df/da = sigmoid(-beta*d)
    
    # Unified: df/da = sigmoid(-beta*|a-b|) ... 
    # Case a<=b: sigmoid(-beta*(b-a)) = sigmoid(beta*(a-b)) = sigmoid(-beta*|a-b|) since a-b<0
    # Wait: a<=b means a-b<=0, |a-b|=b-a
    # sigmoid(-beta*|a-b|) = sigmoid(-beta*(b-a)) = sigmoid(beta*(a-b))
    # But we derived df/da = sigmoid(beta*(b-a)) for a<=b
    # These are different! sigmoid(beta*(a-b)) vs sigmoid(beta*(b-a))
    # For a=0,b=1: sigmoid(-1)=0.269 vs sigmoid(1)=0.731
    # Our finite diff gave ~0.7, so df/da = sigmoid(beta*(b-a)) for a<=b
    
    # So: 
    #   if a <= b: df/da = sigmoid(beta*(b-a))
    #   if a > b: df/da = sigmoid(-beta*(a-b))
    #   Unified: df/da = sigmoid(-beta*(a-b)) ... no
    #   if a<=b: sigmoid(beta*(b-a)) = sigmoid(-beta*(a-b))
    #   if a>b: sigmoid(-beta*(a-b))
    #   So YES: df/da = sigmoid(-beta*(a-b)) always!
    
    # Let me verify: a=0, b=1: sigmoid(-1*(0-1)) = sigmoid(1) = 0.731 ✓
    # a=1, b=0: sigmoid(-1*(1-0)) = sigmoid(-1) = 0.269 ✓
    
    # Great! So g_plus[m,k] = sigmoid(-beta * (diffs[m,k] - w[m]))
    
    d_diff = diffs - w_col  # shape (M, n-1)
    g_plus = 1.0 / (1.0 + np.exp(-beta * d_diff))  # sigmoid(-beta*d_diff)
    
    # Now accumulate gradient:
    # grad[order[m,k+1]] += g_plus[m,k]
    # grad[order[m,k]] -= g_plus[m,k]
    
    grad = np.zeros(n, dtype=np.float64)
    
    # Use add.at for accumulation
    # g_plus has shape (M, n-1)
    # order has shape (M, n)
    
    # Flatten
    g_flat = g_plus.flatten()
    order_plus_flat = order[:, 1:].flatten()  # order[m, k+1] for k in 0..n-2
    order_minus_flat = order[:, :-1].flatten()  # order[m, k] for k in 0..n-2
    
    np.add.at(grad, order_plus_flat, g_flat)
    np.add.at(grad, order_minus_flat, -g_flat)
    
    # Scale by 1/M (mean)
    grad /= M
    
    return area, grad


def _union_area_grad(x, M=20000):
    """
    Compute the union area and its gradient w.r.t. x using exact min.
    
    x: shape (64,) - base positions
    M: number of quadrature points
    Returns: (area, grad) where grad has shape (64,)
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    
    # L_i(y) = x_i + a_i * y, shape (M, n)
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    # Sort L along axis=1 and track permutation
    order = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, order, axis=1)
    
    # Consecutive differences
    diffs = np.diff(L_sorted, axis=1)
    
    # mask: (M, n-1) boolean, True where diffs < w (i.e., overlapping)
    mask = diffs < w[:, np.newaxis]
    
    # union_len = w + sum min(diffs, w)
    union_len = w + np.sum(np.minimum(diffs, w[:, np.newaxis]), axis=1)
    area = np.mean(union_len)
    
    # Gradient: where mask is True, contribute +1 to order[m,k], -1 to order[m,k+1]
    grad = np.zeros(n, dtype=np.float64)
    
    rows, cols = np.where(mask)
    
    idx_plus = order[rows, cols]
    idx_minus = order[rows, cols + 1]
    
    np.add.at(grad, idx_plus, 1.0)
    np.add.at(grad, idx_minus, -1.0)
    
    grad /= M
    
    return area, grad


def _keich_init(n=64):
    """
    Keich construction initial guess.
    """
    k = int(np.log2(n))
    j_arr = np.arange(1, k + 1)
    positions = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)
    return positions


def _random_init(n=64, seed=0):
    """
    Random initial guess.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=n)


def _lbfgs_optimize_soft(x0, maxiter=200, M=2000, beta=50.0):
    """
    Run L-BFGS-B optimization using the soft-min surrogate.
    Returns (x_opt, soft_area_opt, exact_area_opt).
    """
    from scipy.optimize import minimize
    
    def objective_soft(x):
        area, grad = _soft_min_area(x, M=M, beta=beta)
        return area, grad
    
    res = minimize(
        objective_soft,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-10}
    )
    
    x_opt = res.x
    soft_area_opt = _soft_min_area(x_opt, M=M, beta=beta)[0]
    exact_area_opt = _union_area(x_opt, M=M)
    
    return x_opt, soft_area_opt, exact_area_opt


def _lbfgs_optimize_exact(x0, maxiter=200, M=2000):
    """
    Run L-BFGS-B optimization using the exact objective.
    Returns (x_opt, area_opt).
    """
    from scipy.optimize import minimize
    
    def objective_exact(x):
        area, grad = _union_area_grad(x, M=M)
        return area, grad
    
    res = minimize(
        objective_exact,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-10}
    )
    
    x_opt = res.x
    area_opt = _union_area(x_opt, M=M)
    
    return x_opt, area_opt


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses basin hopping with soft-min surrogate for smooth optimization,
    followed by exact polishing. Multiple diverse initial points and
    adaptive perturbation help escape local minima.
    
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    
    start_time = time.time()
    deadline = start_time + 625.0  # 625 second limit (5s safety margin)
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return deadline - time.time()
    
    best_x = _keich_init(n)
    best_area = _union_area(best_x, M=2000)
    
    # Phase 1: Generate diverse initial points and run soft-min optimization
    inits = []
    
    # Keich construction
    inits.append(_keich_init(n))
    
    # Perturbed Keich with various noise levels
    for seed in range(8):
        rng = np.random.default_rng(seed)
        noise_level = 0.03 + 0.04 * (seed % 3)
        inits.append(_keich_init(n) + rng.normal(0, noise_level, size=n))
    
    # Random initializations
    for seed in range(4):
        inits.append(_random_init(n, seed=200 + seed))
    
    # Sort inits by quick area estimate
    init_areas = [_union_area(x, M=500) for x in inits]
    init_order = np.argsort(init_areas)
    inits = [inits[i] for i in init_order]
    
    # Phase 1: Soft-min optimization from each initial point
    for i, x0 in enumerate(inits):
        if time_left() < 30.0:
            break
        
        try:
            x_opt, soft_area, exact_area = _lbfgs_optimize_soft(x0, maxiter=150, M=2000, beta=50.0)
            
            if exact_area < best_area:
                best_x = x_opt.copy()
                best_area = exact_area
        except Exception:
            continue
    
    # Phase 2: Basin hopping - perturb best solution and re-optimize
    rng = np.random.default_rng(42)
    n_hops = 0
    max_hops = 50
    hop_size = 0.05
    
    for hop in range(max_hops):
        if time_left() < 60.0:
            break
        if elapsed() > 400.0:  # Reserve time for final polish
            break
        
        # Generate perturbation
        perturbation = rng.normal(0, hop_size, size=n)
        x_perturbed = best_x + perturbation
        
        try:
            # Optimize from perturbed point using soft-min
            x_opt, soft_area, exact_area = _lbfgs_optimize_soft(x_perturbed, maxiter=100, M=2000, beta=50.0)
            
            if exact_area < best_area:
                best_x = x_opt.copy()
                best_area = exact_area
                hop_size = max(0.01, hop_size * 0.8)  # Reduce hop size on success
            else:
                hop_size = min(0.2, hop_size * 1.2)  # Increase hop size on failure
        except Exception:
            continue
    
    # Phase 3: Final polish with exact objective at full resolution
    if time_left() > 10.0:
        try:
            x_final, area_final = _lbfgs_optimize_exact(best_x, maxiter=50, M=20000)
            if area_final < best_area:
                best_x = x_final.copy()
                best_area = area_final
        except Exception:
            pass
    
    # Phase 4: Very tight final polish if time allows
    if time_left() > 5.0:
        try:
            x_final, area_final = _lbfgs_optimize_exact(best_x, maxiter=30, M=20000)
            if area_final < best_area:
                best_x = x_final.copy()
                best_area = area_final
        except Exception:
            pass
    
    return best_x


# EVOLVE-BLOCK-END
