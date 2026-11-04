import numpy as np
import time
import math
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0
OFFSET = 10  # 1/2 / DH

_IDX = np.arange(N, dtype=float)

_DEADLINE = 150.0
_SAFETY = 18.0

_start = time.time()

def _elapsed():
    return time.time() - _start

def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()

# --------------------------------------------------------------------------- #
#  Basis (model) definition
# --------------------------------------------------------------------------- #
def _make_basis():
    """Gaussian bump basis.  M centers spread over the grid, 2 widths each."""
    M = 10
    centers = np.linspace(3.0, N - 4.0, M)
    widths = [2.2, 4.5]
    basis = []
    for c in centers:
        for w in widths:
            basis.append(np.exp(-0.5 * ((_IDX - c) / w) ** 2))
    B = np.stack(basis, axis=1)  # shape (N, P)
    return B

B = _make_basis()
P = B.shape[1]  # number of parameters

def _forward(params):
    """h = max(0, B @ params)."""
    h = B @ params
    np.maximum(h, 0.0, out=h)
    return h

def _ratio_from_h(h):
    """R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) from a precomputed nonneg h."""
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 1e-300 or l2sq <= 1e-300:
        return 0.0
    full = np.convolve(h, h, mode='full')
    ac = DH * full[OFFSET:OFFSET + N]
    acmax = float(np.max(ac))
    if acmax <= 1e-300:
        return 0.0
    return l2sq / (l1 * acmax)

def _ratio(params):
    return _ratio_from_h(_forward(params))

def _logR(params, smooth_beta=50.0):
    h = _forward(params)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 1e-300 or l2sq <= 1e-300:
        return -1e30
    full = np.convolve(h, h, mode='full')
    ac = DH * full[OFFSET:OFFSET + N]
    # Soft-max: log-sum-exp for smooth approximation of log(max(ac))
    ac_pos = np.maximum(ac, 1e-300)
    ac_max = np.max(ac_pos)
    if ac_max <= 1e-300:
        return -1e30
    # log-sum-exp: log(sum(exp(beta*(ac - ac_max))))/beta + ac_max
    scaled = beta_safe = 1.0 / smooth_beta
    shifted = ac_pos - ac_max
    log_sum_exp = ac_max + math.log(np.sum(np.exp(shifted / scaled))) * scaled
    if log_sum_exp <= 1e-300:
        return -1e30
    R = l2sq / (l1 * math.exp(log_sum_exp))
    if R <= 1e-300:
        return -1e30
    return math.log(R)

def _logR_and_grad(params, smooth_beta=50.0):
    """Analytic gradient of log R with soft-max smoothing w.r.t. params.

    Uses the chain rule through h = max(0, B a).  The active (positive) set
    determines the Jacobian dh/da = B_active.
    
    For the soft-max: log(acmax_smooth) = logsumexp(ac/beta)/beta
    d log(acmax_smooth) / d ac_i = (1/beta) * exp(ac_i/beta - logsumexp(ac/beta))
                                 = (1/beta) * softmax(ac/beta)_i
    """
    h_raw = B @ params
    pos = h_raw > 0.0
    h = np.where(pos, h_raw, 0.0)

    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 1e-300 or l2sq <= 1e-300:
        return -1e30, np.zeros_like(params)

    full = np.convolve(h, h, mode='full')
    ac = DH * full[OFFSET:OFFSET + N]
    ac_pos = np.maximum(ac, 1e-300)
    ac_max = np.max(ac_pos)
    if ac_max <= 1e-300:
        return -1e30, np.zeros_like(params)

    # Soft-max computation
    inv_beta = 1.0 / smooth_beta
    shifted = ac_pos - ac_max
    exp_shifted = np.exp(shifted * inv_beta)
    sum_exp = np.sum(exp_shifted)
    if sum_exp <= 1e-300:
        return -1e30, np.zeros_like(params)
    log_sum_exp = ac_max + inv_beta * math.log(sum_exp)
    
    R = l2sq / (l1 * math.exp(log_sum_exp))
    if R <= 1e-300:
        return -1e30, np.zeros_like(params)
    logR = math.log(R)

    # d logR / d h_i:
    # logR = log l2sq - log l1 - log(acmax_smooth)
    # d log l2sq / dh_i = 2 h_i / l2sq
    # d log l1 / dh_i = 1 / l1
    # d log(acmax_smooth) / dh_i = (1/beta) * softmax(ac/beta)_i * (d ac_i / dh_i)
    # where softmax_i = exp(ac_i/beta) / sum(exp(ac_j/beta))
    
    softmax_weights = exp_shifted / sum_exp  # shape (N,)
    
    # d ac_i / dh_k = 2 * DH * h_{i-k} if 0 <= i-k < N else 0
    # For each i, d ac_i / dh is a vector. We need:
    # d log(acmax_smooth) / dh_k = sum_i softmax_weights[i] * (1/beta) * d ac_i / dh_k
    # = (1/beta) * sum_i softmax_weights[i] * 2 * DH * h_{i-k} (if valid)
    
    # Build d log(acmax_smooth) / dh as a vector of length N
    dlog_acmax_dh = np.zeros(N)
    # For each k, dlog_acmax_dh[k] = (1/beta) * 2*DH * sum_i softmax_weights[i] * h_{i-k}
    # This is (1/beta) * 2*DH * (softmax_weights * h) convolved appropriately
    # ac_i = DH * sum_j h_j h_{i-j}, so d ac_i / dh_k = 2*DH*h_{i-k} (symmetric)
    # dlog_acmax_dh[k] = (1/beta) * sum_i softmax_weights[i] * 2*DH*h_{i-k}
    
    # Efficient: for each k, sum over i where i-k is valid
    # dlog_acmax_dh[k] = (2*DH/beta) * sum_i softmax_weights[i] * h[max(0, min(N-1, i-k))]
    # But we need to be careful: h_{i-k} is only defined if 0 <= i-k < N
    
    # Use convolution: (softmax_weights * h)[k] = sum_i softmax_weights[i] * h[i-k] 
    # This is cross-correlation. Let's compute directly.
    
    # For k in 0..N-1:
    #   dlog_acmax_dh[k] = (2*DH/beta) * sum_{i: 0<=i-k<N} softmax_weights[i] * h[i-k]
    
    # Vectorized: for each offset d = i-k, i = k+d, so d ranges such that 0<=k+d<N
    # dlog_acmax_dh[k] = (2*DH/beta) * sum_{d: -k <= d <= N-1-k} softmax_weights[k+d] * h[d]
    
    # This is a convolution of softmax_weights with h, evaluated at specific points.
    # Let's just do it with a loop over the N values since N=50 is small.
    for k in range(N):
        s = 0.0
        for i in range(N):
            j = i - k
            if 0 <= j < N:
                s += softmax_weights[i] * h[j]
        dlog_acmax_dh[k] = (2.0 * DH / smooth_beta) * s

    dlogR_dh = (2.0 * h / l2sq) - (1.0 / l1) - dlog_acmax_dh

    # Chain: d logR / d a_p = sum_i dlogR_dh_i * dh_i/da_p = sum_{i in active} dlogR_dh_i * B[i,p]
    dlogR_dh_active = np.where(pos, dlogR_dh, 0.0)
    grad = B.T @ dlogR_dh_active

    # Sanity fallback: if anything non-finite, use numerical gradient.
    if not np.all(np.isfinite(grad)):
        grad = np.zeros_like(params)
        eps = 1e-6
        for p in range(P):
            pp = params.copy(); pp[p] += eps
            pm = params.copy(); pm[p] -= eps
            lp = _logR(pp); lm = _logR(pm)
            grad[p] = (lp - lm) / (2.0 * eps)
    return logR, grad

def _neg_logR(params):
    v = _logR(params)
    return -v

def _neg_logR_jac(params):
    v, g = _logR_and_grad(params)
    return v, -g

# --------------------------------------------------------------------------- #
#  Optimization
# --------------------------------------------------------------------------- #
def _polish(params0, budget):
    """
    Run L-BFGS-B on -logR starting from params0, with a time budget.
    Returns (best_params, best_logR).
    """
    if _time_left() < 2.0:
        return params0.copy(), _logR(params0)
    
    best_params = params0.copy()
    best_val = _logR(params0)
    
    try:
        res = minimize(
            _neg_logR,
            params0,
            jac=_neg_logR_jac,
            method='L-BFGS-B',
            options={
                'maxiter': 500,
                'gtol': 1e-8,
                'ftol': 1e-12,
                'maxls': 50,
                'maxcor': 20,
            }
        )
        if res.success or res.fun < -best_val:
            best_params = res.x.copy()
            best_val = _logR(res.x)
    except Exception:
        pass
    
    return best_params, best_val

# --------------------------------------------------------------------------- #
#  Direct 50-dim local optimization (from current program)
# --------------------------------------------------------------------------- #
def _autoconv(h):
    """Compute autoconvolution of h (full mode)."""
    return np.convolve(h, h, mode='full')

def _ratio_direct(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h**2)
    f = _autoconv(h)
    f_inf = np.max(f)
    if f_inf <= 0:
        return 0.0
    return l2_sq / (l1 * f_inf)

def _local_optimize_direct(h_init, max_iter=300, lr=0.01, deadline=None):
    """Local optimization using projected gradient ascent on the ratio."""
    h = h_init.copy()
    h = np.maximum(h, 0)
    
    best_h = h.copy()
    best_ratio = _ratio_direct(h)
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradient of log(ratio) w.r.t. h
        l1 = np.sum(h)
        if l1 <= 0:
            break
        l2_sq = np.sum(h**2)
        f = _autoconv(h)
        f_inf = np.max(f)
        if f_inf <= 0:
            break
        
        ratio = l2_sq / (l1 * f_inf)
        if ratio > best_ratio:
            best_ratio = ratio
            best_h = h.copy()
        
        # Gradient computation
        f_max_idx = np.where(f >= f_inf * 0.999)[0]
        if len(f_max_idx) == 0:
            f_max_idx = [np.argmax(f)]
        
        grad = np.zeros(N)
        for j in f_max_idx:
            for i in range(N):
                if 0 <= j - i < N:
                    grad[i] += h[j - i]
                if 0 <= i - j < N:
                    grad[i] += h[i - j]
        
        grad_ratio = 2 * h / l2_sq - 1 / l1 - grad / f_inf
        
        # Project gradient ascent step
        h_new = h + lr * grad_ratio
        h_new = np.maximum(h_new, 0)
        
        # Normalize to keep scale reasonable (ratio is scale invariant)
        h_new_norm = np.sum(h_new)
        if h_new_norm > 0:
            h_new = h_new / h_new_norm * l1
        
        h = h_new
    
    return best_h, best_ratio

# --------------------------------------------------------------------------- #
#  Profile generators
# --------------------------------------------------------------------------- #
def _gaussian_profile(sigma, center=0.5):
    """Generate a Gaussian profile."""
    t = np.linspace(0, 1, N)
    return np.exp(-0.5 * ((t - center) / sigma) ** 2)

def _lorentzian_profile(gamma, center=0.5):
    """Generate a Lorentzian profile."""
    t = np.linspace(0, 1, N)
    return 1.0 / (1.0 + ((t - center) / gamma) ** 2)

def _powerlaw_profile(alpha, center=0.5, width=0.5):
    """Generate a power-law profile."""
    t = np.linspace(0, 1, N)
    x = (t - center) / width
    return np.maximum(1.0 - np.abs(x), 0) ** alpha

def _two_peak_profile(sigma1, sigma2, c1, c2):
    """Generate a two-peak profile."""
    t = np.linspace(0, 1, N)
    g1 = np.exp(-0.5 * ((t - c1) / sigma1) ** 2)
    g2 = np.exp(-0.5 * ((t - c2) / sigma2) ** 2)
    return g1 + g2

# --------------------------------------------------------------------------- #
#  Main construct
# --------------------------------------------------------------------------- #
def construct():
    global _start
    _start = time.time()
    deadline = _start + _DEADLINE - _SAFETY
    
    best_h = np.ones(N)
    best_ratio = _ratio_direct(best_h)
    
    # Generate diverse initializations
    initializations = []
    
    # Power laws
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        initializations.append(_powerlaw_profile(alpha))
    
    # Gaussians with different widths and centers
    for sigma in [0.1, 0.2, 0.3, 0.4]:
        for center in [0.3, 0.5, 0.7]:
            initializations.append(_gaussian_profile(sigma, center))
    
    # Lorentzians
    for gamma in [0.1, 0.2, 0.3, 0.5]:
        for center in [0.3, 0.5, 0.7]:
            initializations.append(_lorentzian_profile(gamma, center))
    
    # Two peaks
    for s1 in [0.1, 0.2]:
        for s2 in [0.1, 0.2]:
            for c1 in [0.3]:
                for c2 in [0.7]:
                    initializations.append(_two_peak_profile(s1, s2, c1, c2))
    
    # Random initializations
    for _ in range(10):
        h_rand = np.random.rand(N)
        initializations.append(h_rand)
    
    # Evaluate all initializations
    for h_init in initializations:
        if time.time() > deadline:
            break
        h_init = np.maximum(h_init, 0)
        r = _ratio_direct(h_init)
        if r > best_ratio:
            best_ratio = r
            best_h = h_init.copy()
    
    # Local optimization from best initialization and top few
    scored_inits = []
    for h_init in initializations:
        h_init = np.maximum(h_init, 0)
        r = _ratio_direct(h_init)
        scored_inits.append((r, h_init))
    
    scored_inits.sort(key=lambda x: -x[0])
    
    top_k = min(5, len(scored_inits))
    for i in range(top_k):
        if time.time() > deadline:
            break
        _, h_init = scored_inits[i]
        h_opt, r_opt = _local_optimize_direct(
            h_init, 
            max_iter=200, 
            lr=0.005,
            deadline=deadline
        )
        if r_opt > best_ratio:
            best_ratio = r_opt
            best_h = h_opt
    
    # Model-based optimization: fit parameters to best_h
    if time.time() < deadline - 30:
        # Try to find params that approximate best_h
        # Use least squares to get initial params
        try:
            A, residuals, rank, sv = np.linalg.lstsq(B, best_h, rcond=None)
            params0 = A
            h_model = _forward(params0)
            r_model = _ratio_from_h(h_model)
            
            if r_model > 0:
                # Polish with L-BFGS-B
                params_opt, logR_opt = _polish(params0, budget=20)
                h_opt = _forward(params_opt)
                r_opt = _ratio_from_h(h_opt)
                
                if r_opt > best_ratio:
                    best_ratio = r_opt
                    best_h = h_opt
        except Exception:
            pass
    
    # Try multiple random parameter initializations for model-based
    if time.time() < deadline - 20:
        for _ in range(3):
            if time.time() > deadline:
                break
            params_rand = np.random.randn(P) * 0.1
            params_rand += 0.5
            h_rand_model = _forward(params_rand)
            r_rand = _ratio_from_h(h_rand_model)
            
            if r_rand > 0:
                params_opt, logR_opt = _polish(params_rand, budget=10)
                h_opt = _forward(params_opt)
                r_opt = _ratio_from_h(h_opt)
                
                if r_opt > best_ratio:
                    best_ratio = r_opt
                    best_h = h_opt
    
    # Final polish with smaller learning rate
    if time.time() < deadline - 10:
        h_opt, r_opt = _local_optimize_direct(
            best_h,
            max_iter=300,
            lr=0.001,
            deadline=deadline
        )
        if r_opt > best_ratio:
            best_ratio = r_opt
            best_h = h_opt
    
    # Ensure non-negative
    best_h = np.maximum(best_h, 0)
    
    return list(best_h)
