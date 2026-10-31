"""Combined: analytic moments + scale-invariance-aware multi-start optimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.integrate import quad

_START = time.time()
_DEADLINE = 210.0
_MARGIN = 15.0

def _elapsed():
    return time.time() - _START

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)!/(4^n * n!) * sqrt(pi)
def _compute_I(n_max=14):
    I = np.zeros(n_max)
    log_pi = np.log(np.pi)
    for n in range(n_max):
        log_val = 0.5 * log_pi
        for m in range(1, 2*n+1):
            log_val += np.log(m)
        log_val -= n * np.log(4.0)
        for m in range(1, n+1):
            log_val -= np.log(m)
        I[n] = np.exp(log_val)
    return I

_I = _compute_I(14)

# Physicist's Hermite polynomial coefficients (in powers x^0, x^2, ..., x^{2k})
# H_0 = 1
# H_4 = x^4 - 6x^2 + 3
# H_8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H_12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945

_H0_c = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_H4_c = np.array([3.0, -6.0, 1.0, 0.0, 0.0, 0.0, 0.0])
_H8_c = np.array([105.0, -420.0, 210.0, -28.0, 1.0, 0.0, 0.0])
_H12_c = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])

_H_all = np.array([_H0_c, _H4_c, _H8_c, _H12_c])  # shape (4,7)

def _get_p(a0, a4, a8):
    """Return polynomial coefficients p[0..6] for P(x) = sum p_k x^{2k}."""
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H_all  # (4,) @ (4,7) -> (7,)
    return p, a12

def _moments_from_p(p):
    """Compute m0, m2, n0, n2 from polynomial coefficients p."""
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    # m_k = sum_{i,j} p_i p_j I[k+i+j] for k=0,1 (i.e., m0, m2)
    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j] for k=0,1
    
    m0 = 0.0
    m2 = 0.0
    n0 = 0.0
    n2 = 0.0
    
    for i in range(7):
        pi = p[i]
        qi = q[i]
        if pi == 0.0 and qi == 0.0:
            continue
        for j in range(7):
            idx0 = i + j
            idx2 = idx0 + 1
            if idx0 < 14:
                m0 += pi * p[j] * _I[idx0]
                n0 += qi * q[j] * _I[idx0]
            if idx2 < 14:
                m2 += pi * p[j] * _I[idx2]
                n2 += qi * q[j] * _I[idx2]
    
    n0 *= 2.0 * np.pi
    n2 *= 2.0 * np.pi
    
    return m0, m2, n0, n2

def _bound(params):
    """Compute uncertainty bound. Returns (bound, m0, m2, n0, n2, p)."""
    a0, a4, a8 = params[0], params[1], params[2]
    p, a12 = _get_p(a0, a4, a8)
    m0, m2, n0, n2 = _moments_from_p(p)
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, m0, m2, n0, n2, p
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0 or not np.isfinite(vf) or not np.isfinite(vF):
        return 1e18, m0, m2, n0, n2, p
    
    bound = np.sqrt(vf * vF)
    return bound, m0, m2, n0, n2, p

def _bound_grad(params):
    """Compute bound and analytic gradient."""
    a0, a4, a8 = params
    p, a12 = _get_p(a0, a4, a8)
    m0, m2, n0, n2 = _moments_from_p(p)
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0 or not np.isfinite(vf) or not np.isfinite(vF):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # da12/da_c
    da12_da = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # dp/da_c = H_c + da12_da[c] * H12
    H_full = [_H0_c, _H4_c, _H8_c]
    
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    grad = np.zeros(3)
    for c in range(3):
        dp = H_full[c] + da12_da[c] * _H12_c
        dq = dp * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        
        # dm_k/da_c = 2 * sum_i dp_i * sum_j p_j I[k+i+j]
        dm0 = 0.0
        dm2 = 0.0
        for i in range(7):
            dpi = dp[i]
            if dpi == 0.0:
                continue
            s0 = 0.0
            s2 = 0.0
            for j in range(7):
                idx0 = i + j
                idx2 = idx0 + 1
                if idx0 < 14:
                    s0 += p[j] * _I[idx0]
                if idx2 < 14:
                    s2 += p[j] * _I[idx2]
            dm0 += 2.0 * dpi * s0
            dm2 += 2.0 * dpi * s2
        
        # dn_k/da_c = 2*pi * 2 * sum_i dq_i * sum_j q_j I[k+i+j]
        dn0 = 0.0
        dn2 = 0.0
        for i in range(7):
            dqi = dq[i]
            if dqi == 0.0:
                continue
            s0 = 0.0
            s2 = 0.0
            for j in range(7):
                idx0 = i + j
                idx2 = idx0 + 1
                if idx0 < 14:
                    s0 += q[j] * _I[idx0]
                if idx2 < 14:
                    s2 += q[j] * _I[idx2]
            dn0 += 4.0 * np.pi * dqi * s0
            dn2 += 4.0 * np.pi * dqi * s2
        
        dvm2 = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn2 = (dn2 * n0 - n2 * dn0) / (n0 * n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad

def _objective(params):
    val, _ = _bound_grad(params)
    return val

def _objective_grad(params):
    return _bound_grad(params)

def _nm_objective(params):
    """Nelder-Mead objective (no gradient)."""
    val, _ = _bound(params)
    return val

def construct():
    """Return the three Hermite coefficients (degrees 0, 4, 8)."""
    # Seed starting point
    seed = np.array([-113.0/100, 1.0/25, 1.0/3240])
    
    try:
        val0, _ = _bound(seed)
        best_val = val0
    except Exception:
        return seed
    
    best_coeffs = seed.copy()
    
    # Generate diverse starting points exploiting scale invariance
    rng = np.random.default_rng(42)
    
    starts = [seed.copy()]
    
    # Scale variations (exploiting homogeneity)
    for scale in [0.3, 0.5, 0.7, 1.0, 1.3, 1.7, 2.5, 4.0]:
        starts.append(seed * scale)
    
    # Random perturbations at multiple scales
    for sigma in [0.05, 0.1, 0.2, 0.5, 1.0]:
        for _ in range(5):
            starts.append(seed + rng.normal(0, sigma, size=3))
    
    # Structured diverse points
    structured = [
        [0.0, 0.0, 0.0],
        [-0.5, 0.02, 0.00015],
        [-1.0, 0.04, 0.0003],
        [-1.5, 0.06, 0.0005],
        [-2.0, 0.08, 0.0007],
        [-3.0, 0.12, 0.001],
        [0.5, -0.02, -0.0001],
        [1.0, -0.04, -0.0003],
        [-1.13, 0.04, 0.00031],
        [-1.2, 0.035, 0.00028],
        [-1.05, 0.045, 0.00033],
    ]
    for s in structured:
        starts.append(np.array(s))
    
    # Log-scale random exploration
    for _ in range(10):
        s0 = rng.uniform(0.1, 5.0)
        s4 = rng.uniform(0.005, 0.1)
        s8 = rng.uniform(1e-5, 0.001)
        sign0 = rng.choice([-1, 1])
        starts.append(np.array([sign0*s0, s4, s8]))
    
    # L-BFGS-B from each start
    for x0 in starts:
        if _elapsed() > _DEADLINE - _MARGIN:
            break
        try:
            res = minimize(
                _objective_grad, x0,
                jac=True, method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            continue
    
    # Nelder-Mead polish from best (gradient-free, robust)
    if _elapsed() < _DEADLINE - _MARGIN:
        try:
            res_nm = minimize(
                _nm_objective, best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if res_nm.fun < best_val and np.isfinite(res_nm.fun):
                best_val = res_nm.fun
                best_coeffs = res_nm.x.copy()
        except Exception:
            pass
    
    # Final L-BFGS-B high-precision polish
    if _elapsed() < _DEADLINE - _MARGIN:
        try:
            res = minimize(
                _objective_grad, best_coeffs,
                jac=True, method='L-BFGS-B',
                options={'maxiter': 2000, 'maxfun': 5000, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
        
        # One more NM polish
        if _elapsed() < _DEADLINE - 5:
            try:
                res_nm2 = minimize(
                    _nm_objective, best_coeffs,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
                )
                if res_nm2.fun < best_val and np.isfinite(res_nm2.fun):
                    best_val = res_nm2.fun
                    best_coeffs = res_nm2.x.copy()
            except Exception:
                pass
    
    # Safety check
    if not np.all(np.isfinite(best_coeffs)):
        return seed
    return best_coeffs
# EVOLVE-BLOCK-END
