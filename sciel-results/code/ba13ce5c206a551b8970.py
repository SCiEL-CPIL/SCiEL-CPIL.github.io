"""Combined: analytic moment computation for uncertainty bound + multi-start L-BFGS-B."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 180.0
_MARGIN = 10.0

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n * n!) * sqrt(pi)
# Need up to n=12 (since P has degree 12, P^2 has degree 24, so max power 24 = 2*12)
def _compute_I(n_max=13):
    """Compute I_n for n=0..n_max-1."""
    I = np.zeros(n_max)
    log_pi = np.log(np.pi)
    for n in range(n_max):
        # (2n)! / (4^n * n!) * sqrt(pi)
        log_val = 0.0
        # log((2n)!)
        for m in range(1, 2*n+1):
            log_val += np.log(m)
        # - n*log(4)
        log_val -= n * np.log(4.0)
        # - log(n!)
        for m in range(1, n+1):
            log_val -= np.log(m)
        # + 0.5*log(pi)
        log_val += 0.5 * log_pi
        I[n] = np.exp(log_val)
    return I

_I = _compute_I(13)  # I[0]..I[12]

# Physicist's Hermite polynomials: H_n(x)
# H_0 = 1
# H_4 = x^4 - 6x^2 + 3
# H_8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H_12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945

# Coefficients of H_n as polynomial in x: H_n = sum_{k=0}^{n//2} c_k x^{2k}
# H_0: [1]
# H_4: [3, -6, 1]  (constant, x^2, x^4)
# H_8: [105, -420, 210, -28, 1]  (constant, x^2, x^4, x^6, x^8)
# H_12: [945, -7560, 13860, -6930, 1155, -66, 1]  (constant, x^2, ..., x^12)

_H0_coeffs = np.array([1.0])  # degree 0
_H4_coeffs = np.array([3.0, -6.0, 1.0])  # degree 4
_H8_coeffs = np.array([105.0, -420.0, 210.0, -28.0, 1.0])  # degree 8
_H12_coeffs = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])  # degree 12

# Values at x=0
_H0_0 = 1.0
_H4_0 = 3.0
_H8_0 = 105.0
_H12_0 = 945.0

# Constraint: g(0) = 0 => a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
# a12 = -(a0 + 3*a4 + 105*a8) / 945

def _get_polynomial_coeffs(a0, a4, a8):
    """Return coefficients p_k of P(x) = sum p_k x^{2k} for g(x) = P(x) e^{-x^2/2}."""
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    # Each H has coefficients in powers x^0, x^2, x^4, ...
    # H0: 1 coeff
    # H4: 3 coeffs (x^0, x^2, x^4)
    # H8: 5 coeffs (x^0, x^2, x^4, x^6, x^8)
    # H12: 7 coeffs (x^0, x^2, x^4, x^6, x^8, x^10, x^12)
    
    p = np.zeros(7)  # p[0]..p[6] for x^0, x^2, x^4, x^6, x^8, x^10, x^12
    
    # Add a0 * H0
    p[0] += a0 * _H0_coeffs[0]
    
    # Add a4 * H4
    for k in range(len(_H4_coeffs)):
        p[k] += a4 * _H4_coeffs[k]
    
    # Add a8 * H8
    for k in range(len(_H8_coeffs)):
        p[k] += a8 * _H8_coeffs[k]
    
    # Add a12 * H12
    for k in range(len(_H12_coeffs)):
        p[k] += a12 * _H12_coeffs[k]
    
    return p, a12


def _compute_moments(p, q):
    """
    Compute moments m_k = sum_{i,j} p_i p_j I[k+i+j] for k=0,1,2
    and n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j] for k=0,1,2
    where q_k = p_k * (-1)^k (for Fourier transform)
    """
    m = np.zeros(3)
    n = np.zeros(3)
    
    # Precompute for efficiency
    # m_k = sum_{i=0}^{6} sum_{j=0}^{6} p_i p_j I[k+i+j]
    for k in range(3):
        mk = 0.0
        nk = 0.0
        for i in range(7):
            pi = p[i]
            qi = q[i]
            if pi == 0.0 and qi == 0.0:
                continue
            for j in range(7):
                idx = k + i + j
                if idx < len(_I):
                    mk += pi * p[j] * _I[idx]
                    nk += qi * q[j] * _I[idx]
        m[k] = mk
        n[k] = 2.0 * np.pi * nk
    
    return m, n


def _bound_and_grad(params):
    """
    Compute uncertainty bound and its analytic gradient.
    g(x) = P(x) e^{-x^2/2} where P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    a12 = -(a0 + 3a4 + 105a8)/945
    
    A(f)^2 = <x^2> = m2/m0 where m_k = int x^k |g|^2 dx
    A(F)^2 = <xi^2> = n2/n0 where n_k = int xi^k |G|^2 dxi
    bound = sqrt(A(f)^2 * A(F)^2) = sqrt((m2/m0) * (n2/n0))
    """
    a0, a4, a8 = params[0], params[1], params[2]
    
    p, a12 = _get_polynomial_coeffs(a0, a4, a8)
    
    # q_k = p_k * (-1)^k for Fourier transform
    q = np.zeros(7)
    for k in range(7):
        q[k] = p[k] * ((-1.0) ** k)
    
    m, n = _compute_moments(p, q)
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1.0, 1e18, 1.0, 1e18, 1.0])
    
    vf = m2 / m0  # A(f)^2
    vF = n2 / n0  # A(F)^2
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1.0, 1e18, 1.0, 1e18, 1.0])
    
    bound = np.sqrt(vf * vF)
    
    # Gradient: d(bound)/d(a_c) = (1/(2*bound)) * (vF * d(vf)/d(a_c) + vf * d(vF)/d(a_c))
    # d(vf)/d(a_c) = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    # d(vF)/d(a_c) = (dn2/da_c * n0 - n2 * dn0/da_c) / n0^2
    
    # da12/da0 = -1/945, da12/da4 = -3/945, da12/da8 = -105/945
    da12_da = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # dp_k/da_c for each coefficient c=0,1,2 (a0, a4, a8)
    # p = a0*H0 + a4*H4 + a8*H8 + a12*H12
    # dp/da_c = H_c + da12_da[c] * H12
    
    # H0 coeffs: [1, 0, 0, 0, 0, 0, 0]
    # H4 coeffs: [3, -6, 1, 0, 0, 0, 0]
    # H8 coeffs: [105, -420, 210, -28, 1, 0, 0]
    # H12 coeffs: [945, -7560, 13860, -6930, 1155, -66, 1]
    
    H0_full = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    H4_full = np.array([3.0, -6.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    H8_full = np.array([105.0, -420.0, 210.0, -28.0, 1.0, 0.0, 0.0])
    H12_full = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
    
    H_full = [H0_full, H4_full, H8_full]
    
    grad = np.zeros(3)
    
    for c in range(3):
        # dp/da_c
        dp = H_full[c] + da12_da[c] * H12_full
        
        # dm_k/da_c = 2 * sum_i dp_i * (sum_j p_j I[k+i+j])
        dm = np.zeros(3)
        for k in range(3):
            s = 0.0
            for i in range(7):
                dpi = dp[i]
                if dpi == 0.0:
                    continue
                inner = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < len(_I):
                        inner += p[j] * _I[idx]
                s += dpi * inner
            dm[k] = 2.0 * s
        
        # dq/da_c
        dq = np.zeros(7)
        for k in range(7):
            dq[k] = dp[k] * ((-1.0) ** k)
        
        # dn_k/da_c = 2*pi * 2 * sum_i dq_i * (sum_j q_j I[k+i+j])
        dn = np.zeros(3)
        for k in range(3):
            s = 0.0
            for i in range(7):
                dqi = dq[i]
                if dqi == 0.0:
                    continue
                inner = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < len(_I):
                        inner += q[j] * _I[idx]
                s += dqi * inner
            dn[k] = 4.0 * np.pi * s
        
        dvm2 = (dm[1] * m0 - m2 * dm[0]) / (m0 * m0)
        dvn2 = (dn[1] * n0 - n2 * dn[0]) / (n0 * n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    """Objective function for minimization."""
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    """Return (value, gradient) for L-BFGS-B."""
    return _bound_and_grad(params)


def construct():
    """Return the three Hermite coefficients (degrees 0,4,8)."""
    # Prior state of the art as a starting point (from current program)
    # Note: current program uses physicist's Hermite, so seed values are in that basis
    best_coeffs = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    
    try:
        val0, _ = _bound_and_grad(best_coeffs)
        best_val = val0
    except Exception:
        return best_coeffs
    
    # Multi-start L-BFGS-B
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = [best_coeffs.copy()]
    
    # Random perturbations
    for _ in range(15):
        starts.append(best_coeffs + rng.normal(0, 0.1, size=3))
    
    # Structured starts
    for scale in [0.5, 0.7, 1.0, 1.3, 2.0]:
        starts.append(best_coeffs * scale)
    
    # Additional structured points
    starts.append(np.array([0.0, 0.0, 0.0]))
    starts.append(np.array([-1.0, 0.04, 0.0003]))
    starts.append(np.array([-1.5, 0.06, 0.0005]))
    starts.append(np.array([-0.5, 0.02, 0.00015]))
    starts.append(np.array([-2.0, 0.08, 0.0007]))
    
    for x0 in starts:
        if time.time() > _DEADLINE - _MARGIN:
            break
        try:
            res = minimize(
                _objective_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 600, 'ftol': 1e-14, 'gtol': 1e-12},
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            continue
    
    # Final high-precision polish
    if time.time() < _DEADLINE - _MARGIN:
        try:
            res = minimize(
                _objective_grad,
                best_coeffs,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'maxfun': 2000, 'ftol': 1e-16, 'gtol': 1e-14},
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Safety: ensure finite
    if not np.all(np.isfinite(best_coeffs)):
        return np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    return best_coeffs
# EVOLVE-BLOCK-END
