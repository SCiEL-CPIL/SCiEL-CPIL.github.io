"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_cross_correlation_min(h):
    """Compute min over shifts of the cross-correlation R(s) = ∫ h(x)h(x+s)dx.
    
    We evaluate at s = 2k/n for k = 1, ..., n-1 (shifts within (0,2)).
    R(2k/n) = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
    
    We want to maximize min_k R(2k/n).
    """
    n = len(h)
    min_corr = float('inf')
    for k in range(1, n):
        corr = np.dot(h[:n-k], h[k:]) * (2.0 / n)
        if corr < min_corr:
            min_corr = corr
    return min_corr

def _compute_all_correlations(h):
    """Compute all correlations R(k) for k=1,...,n-1."""
    n = len(h)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    return corrs

def _soft_min_cross_correlation(h, beta=100.0):
    """Compute a smooth approximation of min_k R(2k/n) using log-sum-exp."""
    corrs = _compute_all_correlations(h)
    
    min_val = np.min(corrs)
    shifted = corrs - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - log_sum / beta
    return soft_min, corrs

def _objective_and_grad_soft(x, beta=100.0):
    """Objective and gradient using soft-min approximation."""
    h = x
    n = len(h)
    
    # Compute all correlations
    corrs = _compute_all_correlations(h)
    
    # Clean soft-min formulation: soft_min = -(1/beta) * log(sum_k exp(-beta * corrs_k))
    # Use numerically stable version
    min_val = np.min(corrs)
    shifted = corrs - min_val
    exp_neg_beta = np.exp(-beta * shifted)
    sum_exp_neg = np.sum(exp_neg_beta)
    soft_min = min_val - np.log(sum_exp_neg) / beta
    obj = 1.0 - soft_min
    
    # d(soft_min)/d(corrs_k) = exp(-beta*(corrs_k - min_val)) / sum_j exp(-beta*(corrs_j - min_val))
    weights = exp_neg_beta / sum_exp_neg
    
    # d(soft_min)/d(h) = sum_k weights_k * d(corrs_k)/d(h)
    grad = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1] * (2.0 / n)
        if w > 1e-15:
            grad[:n-k] += w * h[k:]
            grad[k:] += w * h[:n-k]
    
    # d(obj)/d(h) = -d(soft_min)/d(h) = -grad
    grad_obj = -grad
    
    return obj, grad_obj

def _project_to_simplex(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum}."""
    n = len(v)
    
    # First project onto {x >= 0, sum = target_sum}
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = np.nonzero(u * np.arange(1, n + 1) > (cssv - target_sum))[0]
    if len(rho_candidates) == 0:
        rho = n - 1
    else:
        rho = rho_candidates[-1]
    theta = (cssv[rho] - target_sum) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    
    # Clip to [0, 1]
    w = np.clip(w, 0, 1)
    
    # Adjust to ensure sum = target_sum
    for _ in range(10):  # Iterate a few times
        diff = target_sum - np.sum(w)
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            mask = w < 1.0
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        else:
            mask = w > 0.0
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        w = np.clip(w, 0, 1)
    
    return w

def _spectral_optimization(h_init, target_sum, n, time_budget):
    """Optimize in frequency domain: shape zero-mean component to maximize min autocorrelation.
    
    Identity: I(s) = 0.25 - R_u(s) where u = h - 0.5, so maximizing min R_u minimizes max I.
    """
    best_h = h_init.copy()
    best_min_corr = _compute_cross_correlation_min(best_h)
    
    # Work in Fourier domain
    # h = 0.5 + u, where u has zero mean
    # u_hat = FFT(u)
    # We want to maximize min_k R_u(k) = (2/n) * sum u[i]*u[i+k]
    
    # In Fourier domain, R_u(k) relates to |u_hat|^2 weighted by cos(2*pi*k*m/n)
    # For step functions on [0,2] with n steps, the correlation at shift k is:
    # R_u(k) = (2/n) * sum_{i=0}^{n-k-1} u[i]*u[i+k]
    # This is not exactly a circular correlation, but close for large n
    
    # Strategy: optimize the magnitude spectrum |u_hat|^2 to maximize min autocorrelation
    # Use gradient descent on the spectrum magnitudes
    
    def _u_from_h(h):
        return h - 0.5
    
    def _h_from_u(u):
        return np.clip(0.5 + u, 0, 1)
    
    def _project_h(h, target_sum):
        h = np.clip(h, 0, 1)
        diff = target_sum - np.sum(h)
        if abs(diff) > 1e-12:
            mask = (h > 1e-10) & (h < 1 - 1e-10)
            if np.sum(mask) > 0:
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0, 1)
        return h
    
    # Start from initial h
    h = h_init.copy()
    u = _u_from_h(h)
    
    # Get initial spectrum
    u_hat = np.fft.rfft(u)
    # Keep only real-valued u, so u_hat satisfies conjugate symmetry
    
    # Optimize the phase and magnitude of u_hat
    # Actually, for real u, we can parameterize by u_hat[0:m+1] with appropriate symmetry
    # Simpler: just do gradient descent on u directly, but use spectral insights for initialization
    
    # Better approach: use the fact that we want flat autocorrelation
    # The optimal u should have a specific spectral shape
    
    # Let's do iterative refinement:
    # 1. Compute current autocorrelation
    # 2. Find the minimum
    # 3. Adjust u to increase the minimum component
    
    for iteration in range(500):
        if _time_remaining() < 6.0:
            break
        
        u = _u_from_h(h)
        corrs = _compute_all_correlations(h)
        min_idx = np.argmin(corrs)
        min_corr = corrs[min_idx]
        
        if min_corr > best_min_corr:
            best_min_corr = min_corr
            best_h = h.copy()
        
        # Compute gradient of min_corr w.r.t. u
        # d(corrs[k])/d(u[i]) = (2/n) * (u[i+k] if i < n-k else 0) + (2/n) * (u[i-k] if i >= k else 0)
        k_star = min_idx + 1
        grad_u = np.zeros(n)
        for j in range(n - k_star):
            grad_u[j] += (2.0 / n) * u[j + k_star]
            grad_u[j + k_star] += (2.0 / n) * u[j]
        
        # Spectral shaping: modify u in frequency domain to boost low-frequency components
        # that contribute to the minimum correlation
        u_hat = np.fft.rfft(u)
        freqs = np.fft.rfftfreq(n, d=1.0/n)
        
        # Boost frequencies that align with the problematic shift
        # The shift k_star corresponds to frequency 1/k_star cycles per unit
        target_freq = k_star / n
        
        # Apply spectral filter to enhance anti-correlation at the target shift
        # This is a heuristic: reduce power at frequencies that cause high correlation at shift k_star
        
        # Actually, let's just do projected gradient descent on u
        step_size = 0.02
        u_new = u + step_size * grad_u
        
        # Project back: ensure h = 0.5 + u is in [0,1] and sums to target_sum
        h_new = _h_from_u(u_new)
        h_new = _project_h(h_new, target_sum)
        
        new_min_corr = _compute_cross_correlation_min(h_new)
        if new_min_corr > min_corr - 1e-14:
            h = h_new
        else:
            step_size *= 0.8
            if step_size < 1e-6:
                break
    
    return best_h, best_min_corr

def construct():
    """Construct step heights minimizing maximum overlap."""
    n = N
    target_sum = n / 2.0
    
    best_x = np.full(n, 0.5)
    best_obj = 1.0 - _compute_cross_correlation_min(best_x)
    
    # Generate diverse initializations
    np.random.seed(42)
    
    restarts = []
    # Uniform
    restarts.append(np.full(n, 0.5))
    # Block structures
    for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
        k = int(n * frac)
        restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, 2.0*target_sum/(n-k))]))
        restarts.append(np.concatenate([np.full(k, 2.0*target_sum/k), np.full(n-k, 0.0)]))
    # Alternating patterns
    restarts.append(np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)]))
    restarts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)]))
    # Random
    for _ in range(15):
        restarts.append(np.random.uniform(0, 1, n))
    # Smooth gradients
    restarts.append(np.linspace(0, 1, n))
    restarts.append(np.linspace(1, 0, n))
    # Sinusoidal
    x = np.linspace(0, 2*np.pi, n)
    restarts.append(0.5 + 0.3*np.sin(x))
    restarts.append(0.5 + 0.3*np.cos(x))
    restarts.append(0.5 + 0.4*np.sin(2*x))
    restarts.append(0.5 + 0.4*np.cos(2*x))
    restarts.append(0.5 + 0.25*np.sin(3*x))
    # Triangular wave
    tri = np.array([abs(2*(i/n - 0.5)) for i in range(n)])
    restarts.append(0.5 + 0.3*(tri - tri.mean()) / (tri.std() + 1e-10))
    # Square wave variants
    for period in [4, 6, 8]:
        sq = np.array([1.0 if (i // (period//2)) % 2 == 0 else 0.0 for i in range(n)])
        restarts.append(0.5 + 0.3*(sq - 0.5))
    
    # Phase 1: Soft-min optimization with increasing beta per restart
    beta_schedule = [10.0, 30.0, 100.0, 300.0]
    
    for x0 in restarts:
        if _time_remaining() < 8.0:
            break
        
        x = _project_to_simplex(x0, target_sum)
        
        for beta in beta_schedule:
            if _time_remaining() < 4.0:
                break
            try:
                res = minimize(
                    _objective_and_grad_soft,
                    x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                res_x = _project_to_simplex(res.x, target_sum)
                res_obj = 1.0 - _compute_cross_correlation_min(res_x)
                if res_obj < best_obj:
                    best_obj = res_obj
                    best_x = res_x
                x = res_x
            except Exception:
                pass
    
    # Phase 1.5: Spectral optimization on best solution
    if _time_remaining() > 20.0:
        try:
            h_spec, min_corr_spec = _spectral_optimization(best_x, target_sum, n, 20.0)
            obj_spec = 1.0 - min_corr_spec
            if obj_spec < best_obj:
                best_obj = obj_spec
                best_x = h_spec
        except Exception:
            pass
    
    # Phase 2: High-beta polish on best solution
    if _time_remaining() > 12.0:
        for beta in [500.0, 1000.0, 2000.0]:
            if _time_remaining() < 4.0:
                break
            try:
                res = minimize(
                    _objective_and_grad_soft,
                    best_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n,
                    options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                res_x = _project_to_simplex(res.x, target_sum)
                res_obj = 1.0 - _compute_cross_correlation_min(res_x)
                if res_obj < best_obj:
                    best_obj = res_obj
                    best_x = res_x
            except Exception:
                pass
    
    # Phase 3: Local search with subgradient on the active constraint
    if _time_remaining() > 8.0:
        step_size = 0.05
        no_improve_count = 0
        for _ in range(200):
            if _time_remaining() < 4.0:
                break
            h = best_x
            n = len(h)
            
            min_corr = float('inf')
            k_star = 1
            for k in range(1, n):
                corr = np.dot(h[:n-k], h[k:]) * (2.0 / n)
                if corr < min_corr:
                    min_corr = corr
                    k_star = k
            
            grad = np.zeros(n)
            for j in range(n - k_star):
                grad[j] += (2.0 / n) * h[j + k_star]
                grad[j + k_star] += (2.0 / n) * h[j]
            
            h_new = h + step_size * grad
            h_new = _project_to_simplex(h_new, target_sum)
            
            new_obj = 1.0 - _compute_cross_correlation_min(h_new)
            if new_obj < best_obj - 1e-14:
                best_obj = new_obj
                best_x = h_new
                no_improve_count = 0
            else:
                no_improve_count += 1
                step_size *= 0.7
                if no_improve_count > 20:
                    break
    
    # Phase 4: Final SLSQP polish with exact objective
    if _time_remaining() > 5.0:
        def _exact_obj(x):
            return 1.0 - _compute_cross_correlation_min(x)
        
        def _exact_obj_grad(x):
            h = x
            n = len(h)
            min_corr = float('inf')
            k_star = 1
            for k in range(1, n):
                corr = np.dot(h[:n-k], h[k:]) * (2.0 / n)
                if corr < min_corr:
                    min_corr = corr
                    k_star = k
            grad = np.zeros(n)
            for j in range(n - k_star):
                grad[j] += (2.0 / n) * h[j + k_star]
                grad[j + k_star] += (2.0 / n) * h[j]
            return -grad  # d(1 - min_corr)/d(h) = -d(min_corr)/d(h)
        
        try:
            res = minimize(
                _exact_obj,
                best_x,
                method='SLSQP',
                jac=_exact_obj_grad,
                bounds=[(0, 1)] * n,
                options={'maxiter': 300, 'ftol': 1e-16}
            )
            res_x = _project_to_simplex(res.x, target_sum)
            res_obj = 1.0 - _compute_cross_correlation_min(res_x)
            if res_obj < best_obj:
                best_obj = res_obj
                best_x = res_x
        except Exception:
            pass
    
    # Ensure exact sum
    best_x = np.clip(best_x, 0, 1)
    diff = target_sum - np.sum(best_x)
    if abs(diff) > 1e-12:
        mask = (best_x > 0) & (best_x < 1)
        if np.sum(mask) > 0:
            best_x[mask] += diff / np.sum(mask)
            best_x = np.clip(best_x, 0, 1)
    
    return best_x
# EVOLVE-BLOCK-END
