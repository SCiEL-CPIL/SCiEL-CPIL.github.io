import numpy as np
from scipy.optimize import minimize, newton
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 205

# Precompute Hermite polynomial coefficients for H_0, H_4, H_8, H_12
# Using physicists' convention: H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}
# H_0 = 1
# H_4 = 16x^4 - 48x^2 + 12
# H_8 = 256x^8 - 3072x^6 + 13440x^4 - 13440x^2 + 1680
# H_12 = 4096x^12 - 98304x^10 + 1290240x^8 - 6451200x^6 + 12902400x^4 - 6451200x^2 + 665280

# Coefficients for x^(2k) in each Hermite polynomial
_H_COEFFS = np.array([
    [1.0, 0, 0, 0, 0, 0, 0],           # H_0
    [12.0, -48.0, 16.0, 0, 0, 0, 0],   # H_4
    [1680.0, -13440.0, 13440.0, -3072.0, 256.0, 0, 0],  # H_8
    [665280.0, -6451200.0, 12902400.0, -6451200.0, 1290240.0, -98304.0, 4096.0]  # H_12
])

# H_n(0) values for the constraint g(0) = 0
_H_AT_ZERO = np.array([1.0, 12.0, 1680.0, 665280.0])

# Precompute I_n = ∫ x^(2n) e^{-x^2} dx = (2n)!/(4^n n!) * sqrt(pi) for n=0..14
def _compute_I_table():
    I = np.zeros(15)
    for n in range(15):
        if n == 0:
            I[n] = np.sqrt(np.pi)
        else:
            # (2n)! / (4^n * n!) * sqrt(pi)
            log_num = sum(math.log(k) for k in range(1, 2*n+1))
            log_den = n * math.log(4) + sum(math.log(k) for k in range(1, n+1))
            I[n] = math.exp(log_num - log_den) * np.sqrt(np.pi)
    return I

_I_TABLE = _compute_I_table()

# Matrix M where M[k, i, j] = I_{k+i+j} for k=0,1; i,j=0..6
# This allows vectorized computation of moments
_M_SPATIAL = np.zeros((2, 7, 7))
_M_FREQ = np.zeros((2, 7, 7))
for k in range(2):
    for i in range(7):
        for j in range(7):
            idx = k + i + j
            if idx < 15:
                _M_SPATIAL[k, i, j] = _I_TABLE[idx]
                _M_FREQ[k, i, j] = 2.0 * np.pi * _I_TABLE[idx]

# Signs for frequency domain: q_k = p_k * (-1)^k
_SIGNS = np.array([(-1.0)**k for k in range(7)])

def _compute_bound_and_grad(params):
    """
    Compute the uncertainty bound sqrt(<x^2><xi^2>) and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 is determined by g(0)=0: a12 = -(a0 + 12*a4 + 1680*a8) / 665280
    """
    a0, a4, a8 = params[0], params[1], params[2]
    
    # Compute a12 from constraint g(0) = 0
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 665280.0
    coeffs = np.array([a0, a4, a8, a12])
    
    # Compute polynomial coefficients p[k] for x^(2k)
    p = _H_COEFFS @ coeffs  # shape (7,)
    
    # Compute spatial moments m_k = ∫ x^(2k) |P(x)|^2 e^{-x^2} dx
    # m_k = sum_i sum_j p_i p_j I_{k+i+j}
    # Using matrix form: m_k = p^T M_k p
    m = np.array([p @ _M_SPATIAL[k] @ p for k in range(2)])
    m0, m2 = m[0], m[1]
    
    # Compute frequency moments
    # Q(xi) = P(-i*xi), so q_k = p_k * (-1)^k
    q = p * _SIGNS
    n = np.array([q @ _M_FREQ[k] @ q for k in range(2)])
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Compute gradient
    # da12/da_c = -H_c(0) / H_12(0)
    da12_vec = -_H_AT_ZERO[:3] / _H_AT_ZERO[3]
    
    grad = np.zeros(3)
    for c_idx in range(3):
        # dp/da_c = H_COEFFS[c_idx] + da12/da_c * H_COEFFS[3]
        dp = _H_COEFFS[c_idx] + da12_vec[c_idx] * _H_COEFFS[3]
        
        # dm_k/da_c = 2 * dp^T M_k p
        dm = 2.0 * np.array([dp @ _M_SPATIAL[k] @ p for k in range(2)])
        
        # dq/da_c = dp * signs
        dq = dp * _SIGNS
        
        # dn_k/da_c = 2 * dq^T M_FREQ_k q
        dn = 2.0 * np.array([dq @ _M_FREQ[k] @ q for k in range(2)])
        
        # d(vf)/da_c = (dm2*m0 - m2*dm0) / m0^2
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        # d(vF)/da_c = (dn2*n0 - n2*dn0) / n0^2
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        
        grad[c_idx] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad

def _compute_log_obj_and_grad(params):
    """
    Compute log(vf) + log(vF) and its gradient (for interior-point/barrier approach).
    This is equivalent to minimizing bound^2/2 in log space, with natural positivity handling.
    """
    a0, a4, a8 = params[0], params[1], params[2]
    
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 665280.0
    coeffs = np.array([a0, a4, a8, a12])
    
    p = _H_COEFFS @ coeffs
    m = np.array([p @ _M_SPATIAL[k] @ p for k in range(2)])
    m0, m2 = m[0], m[1]
    
    q = p * _SIGNS
    n = np.array([q @ _M_FREQ[k] @ q for k in range(2)])
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    if m2 <= 0 or n2 <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    log_obj = math.log(vf) + math.log(vF)
    
    da12_vec = -_H_AT_ZERO[:3] / _H_AT_ZERO[3]
    
    grad = np.zeros(3)
    for c_idx in range(3):
        dp = _H_COEFFS[c_idx] + da12_vec[c_idx] * _H_COEFFS[3]
        dm = 2.0 * np.array([dp @ _M_SPATIAL[k] @ p for k in range(2)])
        dq = dp * _SIGNS
        dn = 2.0 * np.array([dq @ _M_FREQ[k] @ q for k in range(2)])
        
        # d(log(vf))/da_c = dm2/m2 - dm0/m0
        dlog_vf = dm[1]/m2 - dm[0]/m0
        # d(log(vF))/da_c = dn2/n2 - dn0/n0
        dlog_vF = dn[1]/n2 - dn[0]/n0
        
        grad[c_idx] = dlog_vf + dlog_vF
    
    return log_obj, grad

def _objective(params):
    val, _ = _compute_bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _compute_bound_and_grad(params)
    return val, grad

def _log_objective(params):
    val, _ = _compute_log_obj_and_grad(params)
    return val

def _log_objective_grad(params):
    val, grad = _compute_log_obj_and_grad(params)
    return val, grad

def _newton_polish(params, max_iter=100, tol=1e-15):
    """
    High-precision Newton's method using the Hessian of the log-objective.
    The Hessian is computed via finite differences of the gradient (fast enough for 3D).
    """
    x = params.copy().astype(float)
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        f, g = _compute_log_obj_and_grad(x)
        if not np.isfinite(f):
            break
        if np.max(np.abs(g)) < tol:
            break
        
        # Compute Hessian via central differences
        h = 1e-7
        H = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                ei = np.zeros(3); ei[i] = h
                ej = np.zeros(3); ej[j] = h
                _, g_plus = _compute_log_obj_and_grad(x + ei + ej)
                _, g_minus1 = _compute_log_obj_and_grad(x - ei + ej)
                _, g_minus2 = _compute_log_obj_and_grad(x + ei - ej)
                _, g_minus3 = _compute_log_obj_and_grad(x - ei - ej)
                H[i, j] = (g_plus[i] - g_minus1[i] - g_minus2[i] + g_minus3[i]) / (4.0 * h * h)
        
        # Solve H * dx = -g
        try:
            dx = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            break
        
        # Line search
        alpha = 1.0
        f_new = f
        for _ in range(50):
            x_trial = x + alpha * dx
            f_trial, _ = _compute_log_obj_and_grad(x_trial)
            if np.isfinite(f_trial) and f_trial < f_new:
                x = x_trial
                f_new = f_trial
                break
            alpha *= 0.5
        else:
            break
        
        if np.max(np.abs(dx)) < tol:
            break
    
    return x

def construct():
    """
    Global grid search + log-barrier interior-point optimization + Newton polish.
    Returns three coefficients (a0, a4, a8) for H0, H4, H8.
    """
    best_val = np.inf
    best_params = None
    
    def _try_optimize(start, method='L-BFGS-B', maxiter=5000, ftol=1e-16, gtol=1e-14, use_log=False):
        """Helper to try optimization from a starting point."""
        nonlocal best_val, best_params
        try:
            if use_log:
                obj_func = _log_objective_grad
            else:
                obj_func = _objective_grad
            
            if method == 'L-BFGS-B':
                res = minimize(obj_func, start, method='L-BFGS-B',
                              jac=True, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
            elif method == 'Nelder-Mead':
                obj_only = _log_objective if use_log else _objective
                res = minimize(obj_only, start, method='Nelder-Mead',
                              options={'maxiter': maxiter, 'xatol': 1e-16, 'fatol': 1e-16})
            else:
                return
            
            # Evaluate the actual bound (not log) for comparison
            actual_val, _ = _compute_bound_and_grad(res.x)
            if np.isfinite(actual_val) and actual_val < best_val:
                best_val = actual_val
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 1: Dense grid search
    a0_range = np.linspace(-3.0, 1.0, 25)
    a4_range = np.linspace(-0.3, 0.3, 25)
    a8_range = np.linspace(-0.005, 0.005, 15)
    
    grid_starts = []
    for a0 in a0_range:
        for a4 in a4_range:
            for a8 in a8_range:
                grid_starts.append(np.array([a0, a4, a8]))
    
    known_good = np.array([-1.13, 0.04, 0.000308642])
    for da0 in np.linspace(-0.5, 0.5, 11):
        for da4 in np.linspace(-0.05, 0.05, 11):
            for da8 in np.linspace(-0.0005, 0.0005, 5):
                grid_starts.append(known_good + np.array([da0, da4, da8]))
    
    rng = np.random.default_rng(12345)
    for scale in [0.5, 1.0, 2.0, 5.0]:
        for _ in range(10):
            grid_starts.append(rng.normal(0, scale, 3))
    
    # Evaluate grid points
    grid_results = []
    for start in grid_starts:
        if time.time() > _DEADLINE - 80:
            break
        try:
            val, _ = _compute_bound_and_grad(start)
            if np.isfinite(val):
                grid_results.append((val, start))
        except Exception:
            pass
    
    grid_results.sort(key=lambda x: x[0])
    
    # Phase 2: Optimize from top candidates using both standard and log-barrier approaches
    top_candidates = [start for _, start in grid_results[:50]]
    if known_good not in top_candidates:
        top_candidates.append(known_good)
    
    for start in top_candidates:
        if time.time() > _DEADLINE - 40:
            break
        # Standard L-BFGS-B
        _try_optimize(start, method='L-BFGS-B', maxiter=5000, ftol=1e-16, gtol=1e-14, use_log=False)
        # Log-barrier L-BFGS-B
        if time.time() < _DEADLINE - 35:
            _try_optimize(start, method='L-BFGS-B', maxiter=5000, ftol=1e-16, gtol=1e-14, use_log=True)
    
    # Phase 3: Newton polish on the best point
    if best_params is not None and time.time() < _DEADLINE - 15:
        polished = _newton_polish(best_params, max_iter=100, tol=1e-15)
        actual_val, _ = _compute_bound_and_grad(polished)
        if np.isfinite(actual_val) and actual_val < best_val:
            best_val = actual_val
            best_params = polished.copy()
    
    # Phase 4: Final high-precision L-BFGS-B with log objective
    if best_params is not None and time.time() < _DEADLINE - 10:
        _try_optimize(best_params, method='L-BFGS-B', maxiter=10000, ftol=1e-18, gtol=1e-16, use_log=True)
        
        # Another Newton polish
        if time.time() < _DEADLINE - 5:
            polished = _newton_polish(best_params, max_iter=50, tol=1e-16)
            actual_val, _ = _compute_bound_and_grad(polished)
            if np.isfinite(actual_val) and actual_val < best_val:
                best_val = actual_val
                best_params = polished.copy()
    
    # Phase 5: Fine grid around best point if time remains
    if best_params is not None and time.time() < _DEADLINE - 5:
        bp = best_params
        fine_starts = []
        for da0 in np.linspace(-0.1, 0.1, 7):
            for da4 in np.linspace(-0.01, 0.01, 7):
                for da8 in np.linspace(-0.0001, 0.0001, 5):
                    fine_starts.append(bp + np.array([da0, da4, da8]))
        
        for start in fine_starts:
            if time.time() > _DEADLINE - 3:
                break
            _try_optimize(start, method='L-BFGS-B', maxiter=2000, ftol=1e-16, gtol=1e-14, use_log=True)
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
        
    return best_params
