"""Improved program: log-objective + analytic gradients + BFGS + Nelder-Mead + polish."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_hermite
import time

_DEADLINE = 195  # safety margin before 210s kill

def _hermite_h(n, x):
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    elif n == 2:
        return 4 * x**2 - 2
    elif n == 3:
        return 8 * x**3 - 12 * x
    elif n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    elif n == 7:
        # H_7(x) = 128x^7 - 4480x^5 + 40320x^3 - 67200x
        return 128*x**7 - 4480*x**5 + 40320*x**3 - 67200*x
    elif n == 8:
        return 256*x**8 - 4480*x**6 + 26880*x**4 - 67200*x**2 + 40320
    elif n == 11:
        # H_11(x) = 2048x^11 - 147456x^9 + 3133440x^7 - 30198720x^5 + 122305920x^3 - 145152000x
        return (2048*x**11 - 147456*x**9 + 3133440*x**7 
                - 30198720*x**5 + 122305920*x**3 - 145152000*x)
    elif n == 12:
        return (4096*x**12 - 147456*x**10 + 2580480*x**8 
                - 20321280*x**6 + 81729600*x**4 
                - 172972800*x**2 + 103219200)
    else:
        raise ValueError(f"Hermite H_{n} not implemented")

_quad_data = None

def _get_quad_data():
    global _quad_data
    if _quad_data is None:
        n_quad = 300
        x_quad, w_quad = roots_hermite(n_quad)
        
        H0 = _hermite_h(0, x_quad)
        H4 = _hermite_h(4, x_quad)
        H8 = _hermite_h(8, x_quad)
        H12 = _hermite_h(12, x_quad)
        
        # Derivatives: H_n'(x) = 2n H_{n-1}(x)
        dH0 = np.zeros_like(x_quad)
        dH4 = 8 * _hermite_h(3, x_quad)
        dH8 = 16 * _hermite_h(7, x_quad)
        dH12 = 24 * _hermite_h(11, x_quad)
        
        x2 = x_quad**2
        x2w = x2 * w_quad
        
        # Precompute P_i and Q_i basis vectors (independent of coefficients)
        d_d0 = -H0_0 / H12_0
        d_d4 = -H4_0 / H12_0
        d_d8 = -H8_0 / H12_0
        
        Q_base = dH12 - x_quad * H12
        P0 = H0 + d_d0 * H12
        P4 = H4 + d_d4 * H12
        P8 = H8 + d_d8 * H12
        Q0 = dH0 + d_d0 * Q_base
        Q4 = dH4 + d_d4 * Q_base
        Q8 = dH8 + d_d8 * Q_base
        
        _quad_data = (x_quad, w_quad, H0, H4, H8, H12, dH0, dH4, dH8, dH12, x2, x2w,
                      P0, P4, P8, Q0, Q4, Q8)
    return _quad_data

H0_0 = 1.0
H4_0 = 12.0
H8_0 = 40320.0
H12_0 = 103219200.0

def _compute_bound_and_grad(coeffs):
    """
    Compute log(var_x) + log(var_k) and its gradient.
    Returns (log_val, grad) or (inf, None).
    """
    a0, a4, a8 = coeffs
    
    x_quad, w_quad, H0, H4, H8, H12, dH0, dH4, dH8, dH12, x2, x2w, \
        P0, P4, P8, Q0, Q4, Q8 = _get_quad_data()
    
    # P = a0*P0 + a4*P4 + a8*P8
    P = a0*P0 + a4*P4 + a8*P8
    P2 = P * P
    
    # Q = a0*Q0 + a4*Q4 + a8*Q8
    Q = a0*Q0 + a4*Q4 + a8*Q8
    Q2 = Q * Q
    
    norm2 = np.sum(w_quad * P2)
    if norm2 <= 1e-15:
        return float('inf'), None
    
    Sx = np.sum(x2w * P2)
    if Sx <= 1e-15:
        return float('inf'), None
    
    Sd = np.sum(w_quad * Q2)
    if Sd <= 1e-15:
        return float('inf'), None
    
    var_x = Sx / norm2
    var_k = Sd / norm2
    
    log_val = np.log(var_x) + np.log(var_k)
    
    # Gradients using d(log(Sx/S0))/dc = dSx/Sx - dS0/S0
    dS0_0 = 2.0 * np.sum(w_quad * P * P0)
    dS0_4 = 2.0 * np.sum(w_quad * P * P4)
    dS0_8 = 2.0 * np.sum(w_quad * P * P8)
    
    dSx_0 = 2.0 * np.sum(x2w * P * P0)
    dSx_4 = 2.0 * np.sum(x2w * P * P4)
    dSx_8 = 2.0 * np.sum(x2w * P * P8)
    
    dSd_0 = 2.0 * np.sum(w_quad * Q * Q0)
    dSd_4 = 2.0 * np.sum(w_quad * Q * Q4)
    dSd_8 = 2.0 * np.sum(w_quad * Q * Q8)
    
    gx0 = dSx_0 / Sx - dS0_0 / norm2
    gx4 = dSx_4 / Sx - dS0_4 / norm2
    gx8 = dSx_8 / Sx - dS0_8 / norm2
    
    gk0 = dSd_0 / Sd - dS0_0 / norm2
    gk4 = dSd_4 / Sd - dS0_4 / norm2
    gk8 = dSd_8 / Sd - dS0_8 / norm2
    
    grad = np.array([gx0 + gk0, gx4 + gk4, gx8 + gk8])
    
    return log_val, grad

def _newton_solve(start, max_iter=50, tol=1e-12):
    """Newton's method on the gradient system to find exact local minimum."""
    x = start.copy()
    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            break
        f, g = _compute_bound_and_grad(x)
        if f == float('inf'):
            break
        if np.linalg.norm(g) < tol:
            break
        # Compute Hessian via finite differences of gradient
        eps = 1e-6
        H = np.zeros((3, 3))
        for j in range(3):
            xp = x.copy()
            xp[j] += eps
            _, gp = _compute_bound_and_grad(xp)
            xm = x.copy()
            xm[j] -= eps
            _, gm = _compute_bound_and_grad(xm)
            H[:, j] = (gp - gm) / (2 * eps)
        # Damped Newton step
        try:
            step = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            break
        # Line search for damping
        alpha = 1.0
        for _ in range(20):
            x_new = x + alpha * step
            f_new, _ = _compute_bound_and_grad(x_new)
            if f_new != float('inf') and f_new < f:
                x = x_new
                break
            alpha *= 0.5
        else:
            break
    return x

def _compute_bound(coeffs):
    """Original bound (product) for compatibility."""
    if time.time() > _DEADLINE:
        return float('inf')
    val, _ = _compute_bound_and_grad(coeffs)
    if val == float('inf'):
        return float('inf')
    return np.exp(val)

def construct():
    start_time = time.time()
    
    seed = np.array([-113/100, 1/25, 1.0/3240])
    
    best_result = seed.copy()
    best_log_val = _compute_bound_and_grad(seed)[0]
    
    if best_log_val == float('inf'):
        best_log_val = 999.0
    
    def log_objective(params):
        if time.time() > _DEADLINE:
            return float('inf')
        val, _ = _compute_bound_and_grad(params)
        return val
    
    def log_objective_grad(params):
        if time.time() > _DEADLINE:
            return float('inf'), np.zeros(3)
        val, grad = _compute_bound_and_grad(params)
        if val == float('inf'):
            return 999.0, np.zeros(3)
        return val, grad
    
    def _try_update(cand, val):
        nonlocal best_result, best_log_val
        if val < best_log_val:
            best_log_val = val
            best_result = cand.copy()
    
    # Phase 1: BFGS from seed with analytic gradients
    try:
        result = minimize(log_objective_grad, seed, method='L-BFGS-B',
                         jac=True, options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13})
        _try_update(result.x, result.fun)
    except Exception:
        pass
    
    # Phase 1b: Newton polish from BFGS result
    if time.time() < _DEADLINE - 30:
        try:
            newton_res = _newton_solve(best_result, max_iter=30, tol=1e-13)
            nv, _ = _compute_bound_and_grad(newton_res)
            _try_update(newton_res, nv)
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead from seed (robust, no gradient)
    try:
        result = minimize(log_objective, seed, method='Nelder-Mead',
                         options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
        _try_update(result.x, result.fun)
    except Exception:
        pass
    
    # Phase 3: BFGS from best so far
    if time.time() < _DEADLINE - 40:
        try:
            result = minimize(log_objective_grad, best_result, method='L-BFGS-B',
                             jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-13})
            _try_update(result.x, result.fun)
        except Exception:
            pass
    
    # Phase 4: Random restarts with diverse initializations
    np.random.seed(42)
    n_restarts = 25
    if time.time() < _DEADLINE - 30:
        for i in range(n_restarts):
            if time.time() > _DEADLINE - 30:
                break
            # Generate diverse starting points
            if i < 5:
                # Small perturbations of best
                random_start = best_result + np.random.normal(0, 0.05, 3)
            elif i < 10:
                # Medium perturbations
                random_start = best_result + np.random.normal(0, 0.2, 3)
            elif i < 15:
                # Large random in box
                random_start = np.random.uniform(-2.0, 2.0, 3)
            elif i < 20:
                # Scaled random
                scale = 0.5 + (i - 15) * 0.3
                random_start = np.random.uniform(-scale, scale, 3)
            else:
                # Very large scale to explore far regions
                random_start = np.random.uniform(-5.0, 5.0, 3)
            
            try:
                result = minimize(log_objective_grad, random_start, method='L-BFGS-B',
                                 jac=True, options={'maxiter': 2000, 'ftol': 1e-13, 'gtol': 1e-11})
                _try_update(result.x, result.fun)
            except Exception:
                pass
            
            # Nelder-Mead from random start (only for some)
            if time.time() < _DEADLINE - 30 and i % 2 == 0:
                try:
                    result = minimize(log_objective, random_start, method='Nelder-Mead',
                                     options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-13, 'adaptive': True})
                    _try_update(result.x, result.fun)
                except Exception:
                    pass
    
    # Phase 5: Final high-precision polish
    if time.time() < _DEADLINE - 15:
        try:
            result = minimize(log_objective_grad, best_result, method='L-BFGS-B',
                             jac=True, options={'maxiter': 15000, 'ftol': 1e-16, 'gtol': 1e-15})
            _try_update(result.x, result.fun)
        except Exception:
            pass
        
        # Newton polish
        if time.time() < _DEADLINE - 10:
            try:
                newton_res = _newton_solve(best_result, max_iter=50, tol=1e-14)
                nv, _ = _compute_bound_and_grad(newton_res)
                _try_update(newton_res, nv)
            except Exception:
                pass
        
        # One more Nelder-Mead polish with very tight tolerance
        if time.time() < _DEADLINE - 5:
            try:
                result = minimize(log_objective, best_result, method='Nelder-Mead',
                                 options={'maxiter': 30000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True})
                _try_update(result.x, result.fun)
            except Exception:
                pass
    
    return np.array(best_result)

# EVOLVE-BLOCK-END
