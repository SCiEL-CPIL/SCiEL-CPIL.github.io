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
    """Compute a smooth approximation of min_k R(2k/n) using log-sum-exp.
    
    soft_min = -log(sum(exp(-beta * R_k))) / beta
    
    This is differentiable and approximates the hard min as beta -> inf.
    """
    corrs = _compute_all_correlations(h)
    
    # Use log-sum-exp for numerical stability
    min_val = np.min(corrs)
    shifted = corrs - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - log_sum / beta
    return soft_min, corrs

def _objective_soft(x, beta=100.0):
    """Objective: minimize 1 - soft_min_corr, approximating max overlap."""
    h = x
    soft_min, _ = _soft_min_cross_correlation(h, beta)
    obj = 1.0 - soft_min
    return obj

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

def _fourier_to_h(coeffs, n, target_sum):
    """Convert Fourier coefficients (DC removed) to h values.
    coeffs: array of shape (n,) where coeffs[0] is DC (ignored), coeffs[1:n] are complex.
    Returns h in [0,1] with sum = n/2.
    """
    h = np.full(n, 0.5)
    for k in range(1, n):
        if k >= len(coeffs) or np.abs(coeffs[k]) < 1e-15:
            continue
        c = coeffs[k]
        h += np.real(c * np.exp(2j * np.pi * k * np.arange(n) / n))
    h = np.clip(h, 0, 1)
    h = _project_to_simplex(h, target_sum)
    return h

def _h_to_fourier(h, n):
    """Convert h to Fourier coefficients (DC removed)."""
    h_centered = h - np.mean(h)
    coeffs = np.zeros(n, dtype=complex)
    for k in range(1, n):
        coeffs[k] = np.sum(h_centered * np.exp(-2j * np.pi * k * np.arange(n) / n)) / n
    return coeffs

def _spectral_objective(coeffs, n, target_sum, beta):
    """Objective in Fourier coefficient space: 1 - soft_min_corr(h)."""
    h = _fourier_to_h(coeffs, n, target_sum)
    soft_min, _ = _soft_min_cross_correlation(h, beta)
    return 1.0 - soft_min

def _spectral_objective_and_grad(coeffs, n, target_sum, beta, n_harm):
    """Objective and gradient in Fourier coefficient space using finite differences for gradient."""
    h = _fourier_to_h(coeffs, n, target_sum)
    soft_min, _ = _soft_min_cross_correlation(h, beta)
    obj = 1.0 - soft_min
    
    # Compute gradient via finite differences (complex step is not applicable due to projection)
    grad = np.zeros(len(coeffs), dtype=complex)
    eps = 1e-7
    for k in range(1, n_harm + 1):
        # Real part
        coeffs_plus = coeffs.copy()
        coeffs_plus[k] += eps
        h_plus = _fourier_to_h(coeffs_plus, n, target_sum)
        soft_min_plus, _ = _soft_min_cross_correlation(h_plus, beta)
        obj_plus = 1.0 - soft_min_plus
        
        coeffs_minus = coeffs.copy()
        coeffs_minus[k] -= eps
        h_minus = _fourier_to_h(coeffs_minus, n, target_sum)
        soft_min_minus, _ = _soft_min_cross_correlation(h_minus, beta)
        obj_minus = 1.0 - soft_min_minus
        
        grad[k] = (obj_plus - obj_minus) / (2 * eps)
    
    return obj, grad

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
    
    # Phase 2: Spectral optimization in Fourier space
    if _time_remaining() > 20.0:
        # Try different numbers of harmonics
        for n_harm in [2, 3, 4, 5, 6, 8]:
            if _time_remaining() < 10.0:
                break
            
            # Initialize coefficients from best_x
            coeffs = _h_to_fourier(best_x, n)
            
            # Also try random initializations
            for trial in range(5):
                if _time_remaining() < 5.0:
                    break
                
                if trial == 0:
                    # Use coefficients from best_x
                    coeffs = _h_to_fourier(best_x, n)
                else:
                    # Random coefficients
                    coeffs = np.zeros(n, dtype=complex)
                    for k in range(1, n_harm + 1):
                        amp = np.random.uniform(0.1, 0.4)
                        phase = np.random.uniform(0, 2 * np.pi)
                        coeffs[k] = amp * np.exp(1j * phase)
                
                # Optimize in Fourier space
                for beta in [10.0, 30.0, 100.0]:
                    if _time_remaining() < 3.0:
                        break
                    try:
                        res = minimize(
                            _spectral_objective_and_grad,
                            coeffs,
                            args=(n, target_sum, beta, n_harm),
                            method='L-BFGS-B',
                            jac=True,
                            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                        )
                        res_h = _fourier_to_h(res.x, n, target_sum)
                        res_obj = 1.0 - _compute_cross_correlation_min(res_h)
                        if res_obj < best_obj:
                            best_obj = res_obj
                            best_x = res_h
                            coeffs = _h_to_fourier(res_h, n)
                    except Exception:
                        pass
    
    # Phase 3: High-beta polish on best solution
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
    
    # Phase 4: Local search with subgradient on the active constraint
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
    
    # Phase 5: Final SLSQP polish with exact objective
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
