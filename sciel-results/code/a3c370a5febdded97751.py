import numpy as np
from scipy.optimize import minimize, least_squares
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 195

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

# Precompute the constraint Jacobian for a12
# a12 = -(a0 + 12*a4 + 1680*a8) / 665280
_CONSTRAINT_NUM = np.array([1.0, 12.0, 1680.0])
_CONSTRAINT_DEN = 665280.0

def _compute_bound_and_grad(params):
    """
    Compute the uncertainty bound sqrt(<x^2><xi^2>) and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 is determined by g(0)=0: a12 = -(a0 + 12*a4 + 1680*a8) / 665280
    """
    a0, a4, a8 = params[0], params[1], params[2]
    
    # Compute a12 from constraint g(0) = 0
    a12 = -np.dot(_CONSTRAINT_NUM, params) / _CONSTRAINT_DEN
    coeffs = np.array([a0, a4, a8, a12])
    
    # Compute polynomial coefficients p[k] for x^(2k)
    p = _H_COEFFS @ coeffs  # shape (7,)
    
    # Compute spatial moments m_k = ∫ x^(2k) |P(x)|^2 e^{-x^2} dx
    # m_k = sum_i sum_j p_i p_j I_{k+i+j}
    # Using matrix form: m_k = p^T M_k p
    m0 = p @ _M_SPATIAL[0] @ p
    m2 = p @ _M_SPATIAL[1] @ p
    
    # Compute frequency moments
    # Q(xi) = P(-i*xi), so q_k = p_k * (-1)^k
    q = p * _SIGNS
    n0 = q @ _M_FREQ[0] @ q
    n2 = q @ _M_FREQ[1] @ q
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Precompute common terms for gradient
    # da12/da_c = -CONSTRAINT_NUM[c] / CONSTRAINT_DEN
    da12_vec = -_CONSTRAINT_NUM / _CONSTRAINT_DEN
    
    # Precompute dp for each parameter (shared across moments)
    dp_list = []
    for c_idx in range(3):
        dp = _H_COEFFS[c_idx] + da12_vec[c_idx] * _H_COEFFS[3]
        dp_list.append(dp)
    
    grad = np.zeros(3)
    for c_idx in range(3):
        dp = dp_list[c_idx]
        
        # dm_k/da_c = 2 * dp^T M_k p
        dm0 = 2.0 * dp @ _M_SPATIAL[0] @ p
        dm2 = 2.0 * dp @ _M_SPATIAL[1] @ p
        
        # dq/da_c = dp * signs
        dq = dp * _SIGNS
        
        # dn_k/da_c = 2 * dq^T M_FREQ_k q
        dn0 = 2.0 * dq @ _M_FREQ[0] @ q
        dn2 = 2.0 * dq @ _M_FREQ[1] @ q
        
        # d(vf)/da_c = (dm2*m0 - m2*dm0) / m0^2
        dvm2 = (dm2*m0 - m2*dm0) / (m0*m0)
        # d(vF)/da_c = (dn2*n0 - n2*dn0) / n0^2
        dvn2 = (dn2*n0 - n2*dn0) / (n0*n0)
        
        grad[c_idx] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad

def _objective(params):
    val, _ = _compute_bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _compute_bound_and_grad(params)
    return val, grad

def _least_squares_residuals(params):
    """
    Residuals for least_squares formulation.
    We minimize ||[bound - target, params - params0]||^2
    where target is a fixed value (the current best bound).
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -np.dot(_CONSTRAINT_NUM, params[:3]) / _CONSTRAINT_DEN
    coeffs = np.array([a0, a4, a8, a12])
    
    p = _H_COEFFS @ coeffs
    m0 = p @ _M_SPATIAL[0] @ p
    m2 = p @ _M_SPATIAL[1] @ p
    q = p * _SIGNS
    n0 = q @ _M_FREQ[0] @ q
    n2 = q @ _M_FREQ[1] @ q
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Use a fixed target that will be updated
    target = _LS_TARGET[0]
    weight = _LS_WEIGHT[0]
    
    return np.array([weight * (bound - target), params[0], params[1], params[2]])

_LS_TARGET = [1e18]
_LS_WEIGHT = [1.0]

def _least_squares_jacobian(params):
    """
    Jacobian for least_squares formulation.
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -np.dot(_CONSTRAINT_NUM, params[:3]) / _CONSTRAINT_DEN
    coeffs = np.array([a0, a4, a8, a12])
    
    p = _H_COEFFS @ coeffs
    m0 = p @ _M_SPATIAL[0] @ p
    m2 = p @ _M_SPATIAL[1] @ p
    q = p * _SIGNS
    n0 = q @ _M_FREQ[0] @ q
    n2 = q @ _M_FREQ[1] @ q
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return np.eye(4)
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return np.eye(4)
    
    bound = np.sqrt(vf * vF)
    weight = _LS_WEIGHT[0]
    
    # Compute gradient of bound
    da12_vec = -_CONSTRAINT_NUM / _CONSTRAINT_DEN
    dp_list = []
    for c_idx in range(3):
        dp = _H_COEFFS[c_idx] + da12_vec[c_idx] * _H_COEFFS[3]
        dp_list.append(dp)
    
    grad_bound = np.zeros(3)
    for c_idx in range(3):
        dp = dp_list[c_idx]
        dm0 = 2.0 * dp @ _M_SPATIAL[0] @ p
        dm2 = 2.0 * dp @ _M_SPATIAL[1] @ p
        dq = dp * _SIGNS
        dn0 = 2.0 * dq @ _M_FREQ[0] @ q
        dn2 = 2.0 * dq @ _M_FREQ[1] @ q
        dvm2 = (dm2*m0 - m2*dm0) / (m0*m0)
        dvn2 = (dn2*n0 - n2*dn0) / (n0*n0)
        grad_bound[c_idx] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    # Jacobian: first row is weight * grad_bound, then identity for params
    J = np.zeros((4, 3))
    J[0, :] = weight * grad_bound
    J[1, 0] = 1.0
    J[2, 1] = 1.0
    J[3, 2] = 1.0
    
    return J

def construct():
    """
    Multi-start optimization to minimize the uncertainty bound.
    Returns three coefficients (a0, a4, a8) for H0, H4, H8.
    """
    best_val = np.inf
    best_params = None
    
    # Generate diverse starting points
    starts = []
    
    # Known good region from previous solutions
    starts.append(np.array([-1.13, 0.04, 0.000308642]))
    starts.append(np.array([-1.0, 0.05, 0.0005]))
    starts.append(np.array([-0.5, 0.02, 0.0002]))
    starts.append(np.array([-2.0, 0.1, 0.001]))
    starts.append(np.array([0.0, 0.0, 0.0]))
    starts.append(np.array([1.0, -0.1, -0.001]))
    starts.append(np.array([-1.5, 0.08, 0.0008]))
    starts.append(np.array([-1.2, 0.045, 0.00035]))
    starts.append(np.array([-0.8, 0.03, 0.00025]))
    starts.append(np.array([-1.3, 0.05, 0.0004]))
    
    # Add more structured starts around the best known point
    base = np.array([-1.13, 0.04, 0.000308642])
    for da0 in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]:
        for da4 in [-0.01, -0.005, 0.0, 0.005, 0.01]:
            starts.append(base + np.array([da0, da4, 0.0]))
    
    # Add random starts with various scales
    rng = np.random.default_rng(12345)
    for scale in [0.1, 0.5, 1.0, 2.0]:
        for _ in range(3):
            starts.append(rng.normal(0, scale, 3))
    
    # Add some structured starts
    for a0 in np.linspace(-3, 3, 7):
        for a4 in np.linspace(-0.2, 0.2, 3):
            starts.append(np.array([a0, a4, 0.0003]))
    
    # Phase 1: L-BFGS-B optimization from each start
    for start in starts:
        if time.time() > _DEADLINE - 60:
            break
        
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Use least_squares with the best parameters found
    if best_params is not None and time.time() < _DEADLINE - 40:
        # Update the target for least_squares
        _LS_TARGET[0] = best_val
        _LS_WEIGHT[0] = 1.0
        
        try:
            res_ls = least_squares(_least_squares_residuals, best_params, 
                                   jac=_least_squares_jacobian,
                                   method='trf', max_nfev=10000, xtol=1e-15, ftol=1e-15, gtol=1e-15)
            # Evaluate the actual bound at the least_squares result
            ls_val, _ = _compute_bound_and_grad(res_ls.x)
            if ls_val < best_val and np.isfinite(ls_val):
                best_val = ls_val
                best_params = res_ls.x.copy()
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead polish
    if best_params is not None and time.time() < _DEADLINE - 20:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Another L-BFGS-B polish with tighter tolerances
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Final least_squares pass with updated target
    if best_params is not None and time.time() < _DEADLINE - 5:
        _LS_TARGET[0] = best_val
        _LS_WEIGHT[0] = 10.0  # Increase weight to focus on matching the target
        
        try:
            res_ls = least_squares(_least_squares_residuals, best_params, 
                                   jac=_least_squares_jacobian,
                                   method='trf', max_nfev=10000, xtol=1e-15, ftol=1e-15, gtol=1e-15)
            ls_val, _ = _compute_bound_and_grad(res_ls.x)
            if ls_val < best_val and np.isfinite(ls_val):
                best_val = ls_val
                best_params = res_ls.x.copy()
        except Exception:
            pass
    
    # Phase 6: Final high-precision L-BFGS-B
    if best_params is not None and time.time() < _DEADLINE - 2:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 20000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
        
    return best_params
