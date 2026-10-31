"""Minimize the C4 uncertainty bound via correct moment computation + L-BFGS-B + polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, fsolve

_DEADLINE = time.time() + 195.0  # safety margin under 210s wall-clock

# Precompute I[n] = integral x^(2n) e^(-x^2) dx = (2n-1)!! * sqrt(pi) / 2^n
# = (2n)! / (4^n * n!) * sqrt(pi)
def _precompute_I(N=14):
    I = np.zeros(N)
    sp = np.sqrt(np.pi)
    for n in range(N):
        if n == 0:
            I[n] = sp
        else:
            I[n] = I[n-1] * (2*n - 1) / (2.0 * n)
    return I

_I = _precompute_I(14)

# Hermite polynomial coefficients (physicists': H_n(x))
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
# Coefficients for x^(2k): index k
_H0_coeffs = np.array([1.0, 0, 0, 0, 0, 0, 0])       # p0=1
_H4_coeffs = np.array([3.0, -6.0, 1.0, 0, 0, 0, 0])   # p0=3, p1=-6, p2=1
_H8_coeffs = np.array([105.0, -420.0, 210.0, -28.0, 1.0, 0, 0])
_H12_coeffs = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])

def _get_p(a):
    """Get polynomial coefficients p_k for P(x) = sum p_k x^(2k)."""
    a0, a4, a8 = a[0], a[1], a[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    p = np.zeros(7)
    p[0] = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12  # should be ~0
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    return p

def _compute_moments(p):
    """Compute m_k = integral x^(2k) P^2 e^(-x^2) dx for k=0,1,2,3."""
    m = np.zeros(4)  # m0, m2, m4, m6
    for i in range(7):
        pi = p[i]
        if abs(pi) < 1e-15:
            continue
        for j in range(7):
            pij = pi * p[j]
            if abs(pij) < 1e-15:
                continue
            for k in range(4):
                idx = k + i + j
                if idx < 14:
                    m[k] += pij * _I[idx]
    return m

def _compute_fourier_moments(p):
    """Compute Fourier-side moments. G(xi) = sqrt(2*pi) P(-i*xi) e^(-xi^2/2).
    P(-i*xi) = sum p_k (-1)^k xi^(2k), so Q_k = p_k * (-1)^k.
    n_k = 2*pi * integral xi^(2k) Q^2 e^(-xi^2) dxi"""
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    n = np.zeros(3)  # n0, n2, n4
    for i in range(7):
        qi = q[i]
        if abs(qi) < 1e-15:
            continue
        for j in range(7):
            qij = qi * q[j]
            if abs(qij) < 1e-15:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    n[k] += 2.0 * np.pi * qij * _I[idx]
    return n

def _bound(a):
    """Compute the C4 uncertainty bound."""
    p = _get_p(a)
    m = _compute_moments(p)
    m0, m2, m4, m6 = m[0], m[1], m[2], m[3]
    if m0 <= 1e-30 or not np.isfinite(m0):
        return 1e300
    mu2 = m2 / m0
    mu4 = m4 / m0
    mu6 = m6 / m0
    den = mu4 - mu2 * mu2
    if den <= 1e-30:
        return 1e300
    num = np.sqrt(1.0 + mu6) - 1.0
    if num < 0.0:
        return 1e300
    return 4.0 * num / den

def _bound_and_grad(a):
    """Compute bound and its gradient with respect to a = (a0, a4, a8)."""
    p = _get_p(a)
    m = _compute_moments(p)
    m0, m2, m4, m6 = m[0], m[1], m[2], m[3]
    if m0 <= 1e-30 or not np.isfinite(m0):
        return 1e300, np.array([1e300, 1e300, 1e300])
    
    mu2 = m2 / m0
    mu4 = m4 / m0
    mu6 = m6 / m0
    den = mu4 - mu2 * mu2
    if den <= 1e-30:
        return 1e300, np.array([1e300, 1e300, 1e300])
    num = np.sqrt(1.0 + mu6) - 1.0
    if num < 0.0:
        return 1e300, np.array([1e300, 1e300, 1e300])
    
    bound = 4.0 * num / den
    
    # Gradient computation
    # dbound/da_c = 4 * [ (0.5/sqrt(1+mu6)) * dmu6/da_c * den - num * dden/da_c ] / den^2
    # dmu_k/da_c = (dm_k/da_c * m0 - m_k * dm0/da_c) / m0^2
    # dm_k/da_c = 2 * sum_i dp_i/da_c * sum_j p_j * I[k+i+j]
    
    # da12/da_c for c=0,1,2: [-1/945, -3/945, -105/945]
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    grad = np.zeros(3)
    dsq = 0.5 / np.sqrt(1.0 + mu6)
    
    for c in range(3):
        # dp_k/da_c
        dp = np.zeros(7)
        dp[0] = np.array([1.0, 3.0, 105.0])[c] + 945.0 * da12[c]
        dp[1] = np.array([0.0, -6.0, -420.0])[c] - 7560.0 * da12[c]
        dp[2] = np.array([0.0, 1.0, 210.0])[c] + 13860.0 * da12[c]
        dp[3] = np.array([0.0, 0.0, -28.0])[c] - 6930.0 * da12[c]
        dp[4] = np.array([0.0, 0.0, 1.0])[c] + 1155.0 * da12[c]
        dp[5] = np.array([0.0, 0.0, 0.0])[c] - 66.0 * da12[c]
        dp[6] = np.array([0.0, 0.0, 0.0])[c] + 1.0 * da12[c]
        
        # dm_k/da_c for k=0,1,2,3
        dm = np.zeros(4)
        for i in range(7):
            dpi = dp[i]
            if abs(dpi) < 1e-15:
                continue
            for k in range(4):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 14:
                        s += p[j] * _I[idx]
                dm[k] += 2.0 * dpi * s
        
        # dmu_k/da_c
        dmu = np.zeros(4)
        for k in range(4):
            dmu[k] = (dm[k] * m0 - m[k] * dm[0]) / (m0 * m0)
        
        dden = dmu[2] - 2.0 * mu2 * dmu[1]
        grad[c] = 4.0 * (dsq * dmu[3] * den - num * dden) / (den * den)
    
    return bound, grad

def _neg_bound(a):
    val, _ = _bound_and_grad(a)
    return val

def _neg_bound_jac(a):
    val, grad = _bound_and_grad(a)
    return val, grad

def _refine(a0):
    """High-precision polish using fsolve on the gradient."""
    def gfun(a):
        _, g = _bound_and_grad(a)
        return g
    
    try:
        sol, info, ier, msg = fsolve(gfun, a0, full_output=True, xtol=1e-14, maxfev=500)
        if ier == 1 and np.all(np.isfinite(sol)):
            val_sol = _bound(sol)
            val_orig = _bound(a0)
            if val_sol < val_orig:
                return sol
    except Exception:
        pass
    return a0

def construct():
    global _DEADLINE
    best_a = np.array([-1.13, 0.04, 1.0/3240.0])
    best_val = _bound(best_a)
    
    # Multi-start optimization
    starts = []
    # Seed point
    starts.append(best_a.copy())
    
    # Systematic grid
    for s0 in [-3.0, -2.5, -2.0, -1.5, -1.13, -1.0, -0.8, -0.5, -0.3]:
        for s4 in [0.0, 0.01, 0.04, 0.1, 0.2, 0.5]:
            for s8 in [0.0, 0.0003, 0.001, 0.01, 0.05]:
                starts.append(np.array([s0, s4, s8]))
    
    # Random perturbations around seed
    rng = np.random.RandomState(42)
    for _ in range(30):
        p = best_a * np.exp(rng.normal(0, 1.0, 3))
        starts.append(p)
    
    for a0 in starts:
        if time.time() > _DEADLINE - 15.0:
            break
        try:
            res = minimize(_neg_bound_jac, a0, method='L-BFGS-B',
                          jac=True, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-13})
            if np.isfinite(res.fun):
                val = _bound(res.x)
                if val < best_val:
                    best_val = val
                    best_a = res.x.copy()
        except Exception:
            pass
    
    # High-precision polish
    if time.time() < _DEADLINE - 5.0:
        try:
            a_pol = _refine(best_a)
            val_pol = _bound(a_pol)
            if val_pol < best_val:
                best_val = val_pol
                best_a = a_pol
        except Exception:
            pass
    
    return np.array([best_a[0], best_a[1], best_a[2]])
# EVOLVE-BLOCK-END
