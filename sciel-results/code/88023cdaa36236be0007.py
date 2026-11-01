import numpy as np
from scipy.optimize import minimize
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 195

# Probabilists' Hermite polynomials: He_n(x) = (-1)^n e^{x^2/2} d^n/dx^n e^{-x^2/2}
# He_0 = 1
# He_4 = x^4 - 6x^2 + 3
# He_8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# He_12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945

# Coefficients for x^(2k) in each Hermite polynomial, k=0..6
_HE_COEFFS = np.array([
    [1.0, 0, 0, 0, 0, 0, 0],           # He_0
    [3.0, -6.0, 1.0, 0, 0, 0, 0],      # He_4
    [105.0, -420.0, 210.0, -28.0, 1.0, 0, 0],  # He_8
    [945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0]  # He_12
])

# He_n(0) values for the constraint g(0) = 0
_HE_AT_ZERO = np.array([1.0, 3.0, 105.0, 945.0])

# Precompute I_n = ∫ x^(2n) e^{-x^2} dx = (2n-1)!! * sqrt(pi) / 2^n for n=0..14
def _compute_I_table():
    I = np.zeros(15)
    for n in range(15):
        if n == 0:
            I[n] = np.sqrt(np.pi)
        else:
            log_num = sum(math.log(k) for k in range(1, 2*n+1))
            log_den = n * math.log(4) + sum(math.log(k) for k in range(1, n+1))
            I[n] = math.exp(log_num - log_den) * np.sqrt(np.pi)
    return I

_I_TABLE = _compute_I_table()

# Matrix M where M[k, i, j] = I_{k+i+j} for k=0,1; i,j=0..6
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
    g(x) = (a0*He0 + a4*He4 + a8*He8 + a12*He12) * exp(-x^2/2)
    a12 is determined by g(0)=0: a12 = -(a0 + 3*a4 + 105*a8) / 945
    """
    a0, a4, a8 = params[0], params[1], params[2]
    
    # Compute a12 from constraint g(0) = 0
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    coeffs = np.array([a0, a4, a8, a12])
    
    # Compute polynomial coefficients p[k] for x^(2k)
    p = _HE_COEFFS @ coeffs  # shape (7,)
    
    # Compute spatial moments m_k = ∫ x^(2k) |P(x)|^2 e^{-x^2} dx
    m0 = p @ _M_SPATIAL[0] @ p
    m2 = p @ _M_SPATIAL[1] @ p
    
    # Compute frequency moments
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
    
    # Compute gradient
    da12_vec = -_HE_AT_ZERO[:3] / _HE_AT_ZERO[3]
    
    grad = np.zeros(3)
    for c_idx in range(3):
        dp = _HE_COEFFS[c_idx] + da12_vec[c_idx] * _HE_COEFFS[3]
        
        dm0 = 2.0 * (dp @ _M_SPATIAL[0] @ p)
        dm2 = 2.0 * (dp @ _M_SPATIAL[1] @ p)
        
        dq = dp * _SIGNS
        
        dn0 = 2.0 * (dq @ _M_FREQ[0] @ q)
        dn2 = 2.0 * (dq @ _M_FREQ[1] @ q)
        
        dvm2 = (dm2*m0 - m2*dm0) / (m0*m0)
        dvn2 = (dn2*n0 - n2*dn0) / (n0*n0)
        
        grad[c_idx] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad

def _objective(params):
    val, _ = _compute_bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _compute_bound_and_grad(params)
    return val, grad

def construct():
    """
    Multi-start optimization to minimize the uncertainty bound.
    Returns three coefficients (a0, a4, a8) for He0, He4, He8.
    """
    best_val = np.inf
    best_params = None
    candidates = []
    
    # Generate diverse starting points
    starts = []
    
    # Known good region from previous solutions (adjusted for probabilists' convention)
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
    
    # Add random starts with various scales
    rng = np.random.default_rng(12345)
    for scale in [0.1, 0.5, 1.0, 2.0]:
        for _ in range(5):
            starts.append(rng.normal(0, scale, 3))
    
    # Add some structured starts
    for a0 in np.linspace(-3, 3, 7):
        for a4 in np.linspace(-0.2, 0.2, 5):
            starts.append(np.array([a0, a4, 0.0003]))
    
    # Optimize from each start
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
            
            # Keep top 3 candidates for polishing
            if np.isfinite(res.fun):
                candidates.append((res.fun, res.x.copy()))
        except Exception:
            pass
    
    # Sort candidates by value and keep top 3
    if candidates:
        candidates.sort(key=lambda x: x[0])
        top_candidates = [c[1] for c in candidates[:3]]
    else:
        top_candidates = [best_params] if best_params is not None else []
    
    # Polish top candidates with Nelder-Mead
    for cand in top_candidates:
        if time.time() > _DEADLINE - 10:
            break
        
        try:
            res = minimize(_objective, cand, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-18, 'fatol': 1e-18, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish on the absolute best
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 100000, 'xatol': 1e-20, 'fatol': 1e-20, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
        
    return best_params
