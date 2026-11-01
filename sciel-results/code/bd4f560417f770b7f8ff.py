"""Exact analytic uncertainty bound with gradient + multi-start optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_T0 = time.time()
_DEADLINE = _T0 + 195.0  # 15s safety margin before 210s kill

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)!/(4^n n!) * sqrt(pi), n=0..14
_I = np.zeros(15)
for _n in range(15):
    _num = 1.0
    for _i in range(1, 2 * _n + 1):
        _num *= _i
    _den = 4.0 ** _n
    for _i in range(1, _n + 1):
        _den *= _i
    _I[_n] = (_num / _den) * np.sqrt(np.pi)

_SP = np.sqrt(np.pi)


def _poly_coeffs(a0, a4, a8):
    """Return coefficients p[0..6] of P(x)=sum p_k x^{2k} for the Hermite combo.
    Uses monic Hermite: H_n(x) = n! * sum_{k=0}^{n//2} (-1)^k binom(n,k) x^{n-2k}/(n-2k)!
    But we use the standard physicist's Hermite polynomials directly.
    H0(x) = 1
    H4(x) = 16x^4 - 48x^2 + 12  =>  as even poly: 12 + (-48)x^2 + 16x^4
    H8(x) = 256x^8 - 4480x^6 + 30240x^4 - 67200x^2 + 16800
    H12(x) = 4096x^12 - 129024x^10 + 1814400x^8 - 14417280x^6 + 55776000x^4 - 91452000x^2 + 47900160
    
    We express P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12 as sum_{k=0}^{6} p_k x^{2k}
    where a12 = -(a0 + 12*a4 + 16800*a8)/47900160 to enforce P(0)=0.
    """
    a12 = -(a0 + 12.0 * a4 + 16800.0 * a8) / 47900160.0
    
    p = np.zeros(7)
    # H0: 1
    p[0] += a0
    # H4: 12 - 48x^2 + 16x^4
    p[0] += a4 * 12.0
    p[1] += a4 * (-48.0)
    p[2] += a4 * 16.0
    # H8: 16800 - 67200x^2 + 30240x^4 - 4480x^6 + 256x^8
    p[0] += a8 * 16800.0
    p[1] += a8 * (-67200.0)
    p[2] += a8 * 30240.0
    p[3] += a8 * (-4480.0)
    p[4] += a8 * 256.0
    # H12: 47900160 - 91452000x^2 + 55776000x^4 - 14417280x^6 + 1814400x^8 - 129024x^10 + 4096x^12
    p[0] += a12 * 47900160.0
    p[1] += a12 * (-91452000.0)
    p[2] += a12 * 55776000.0
    p[3] += a12 * (-14417280.0)
    p[4] += a12 * 1814400.0
    p[5] += a12 * (-129024.0)
    p[6] += a12 * 4096.0
    
    return p, a12


def _moments(p, q):
    """Compute m_k = sum_{i,j} p_i p_j I[k+i+j] for k=0,1,2 and similarly for q.
    p has 7 elements (indices 0..6), q has 7 elements.
    k+i+j can go up to 2+6+6=14, so we need I[0..14].
    """
    m = np.zeros(3)
    for k in range(3):
        s = 0.0
        for i in range(7):
            pi = p[i]
            if pi == 0.0:
                continue
            for j in range(7):
                pj = p[j]
                if pj == 0.0:
                    continue
                idx = k + i + j
                if idx < 15:
                    s += pi * pj * _I[idx]
        m[k] = s
    n = np.zeros(3)
    for k in range(3):
        s = 0.0
        for i in range(7):
            qi = q[i]
            if qi == 0.0:
                continue
            for j in range(7):
                qj = q[j]
                if qj == 0.0:
                    continue
                idx = k + i + j
                if idx < 15:
                    s += qi * qj * _I[idx]
        n[k] = s
    return m, n


def _bound_and_grad(params):
    """Compute uncertainty product sqrt(m2/m0 * n2/n0) and its gradient.
    m_k from |g|^2 = P(x)^2 e^{-x^2}, n_k from |G|^2 proportional to Q(xi)^2 e^{-xi^2}
    where Q_k = (-1)^k p_k (from Fourier transform of x^{2k} e^{-x^2/2}).
    """
    a0, a4, a8 = params[0], params[1], params[2]
    p, a12 = _poly_coeffs(a0, a4, a8)
    
    # q_k = (-1)^k p_k
    q = np.zeros(7)
    for k in range(7):
        q[k] = p[k] * ((-1.0) ** k)
    
    m, n = _moments(p, q)
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0 or not np.isfinite(vf) or not np.isfinite(vF):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Gradient: d(bound)/d(a_c) = (1/(2*bound)) * (vF * d(vf)/d(a_c) + vf * d(vF)/d(a_c))
    # d(vf)/d(a_c) = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    # dm_k/da_c = 2 * sum_i dp_i/da_c * sum_j p_j I[k+i+j]
    
    # da12/da0 = -1/47900160, da12/da4 = -12/47900160, da12/da8 = -16800/47900160
    da12_da = np.array([-1.0, -12.0, -16800.0]) / 47900160.0
    
    grad = np.zeros(3)
    for c in range(3):
        # dp_k/da_c = (dp_k/da_c_direct) + (da12/da_c) * (dp_k/da12)
        # Direct contributions:
        # p[0] = a0 + 12a4 + 16800a8 + 47900160*a12
        # p[1] = -48a4 - 67200a8 - 91452000*a12
        # p[2] = 16a4 + 30240a8 + 55776000*a12
        # p[3] = -4480a8 - 14417280*a12
        # p[4] = 256a8 + 1814400*a12
        # p[5] = -129024*a12
        # p[6] = 4096*a12
        
        dp = np.zeros(7)
        # Direct part
        if c == 0:  # a0
            dp[0] = 1.0
        elif c == 1:  # a4
            dp[0] = 12.0
            dp[1] = -48.0
            dp[2] = 16.0
        elif c == 2:  # a8
            dp[0] = 16800.0
            dp[1] = -67200.0
            dp[2] = 30240.0
            dp[3] = -4480.0
            dp[4] = 256.0
        
        # a12 contribution
        d12 = da12_da[c]
        dp[0] += d12 * 47900160.0
        dp[1] += d12 * (-91452000.0)
        dp[2] += d12 * 55776000.0
        dp[3] += d12 * (-14417280.0)
        dp[4] += d12 * 1814400.0
        dp[5] += d12 * (-129024.0)
        dp[6] += d12 * 4096.0
        
        # dm_k/da_c = 2 * sum_i dp_i * (sum_j p_j I[k+i+j])
        dm = np.zeros(3)
        for i in range(7):
            dpi = dp[i]
            if dpi == 0.0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 15:
                        s += p[j] * _I[idx]
                dm[k] += 2.0 * dpi * s
        
        # dq_k/da_c = (-1)^k dp_k
        dq = np.zeros(7)
        for k in range(7):
            dq[k] = dp[k] * ((-1.0) ** k)
        
        dn = np.zeros(3)
        for i in range(7):
            dqi = dq[i]
            if dqi == 0.0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 15:
                        s += q[j] * _I[idx]
                dn[k] += 2.0 * dqi * s
        
        dvm2 = (dm[1] * m0 - m2 * dm[0]) / (m0 * m0)
        dvn2 = (dn[1] * n0 - n2 * dn[0]) / (n0 * n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    """Find (c0, c4, c8) minimizing the uncertainty product. Returns np.array([c0, c4, c8])."""
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])  # seed
    
    def _time_left():
        return _DEADLINE - time.time()
    
    # Phase 1: Differential Evolution for global exploration
    if _time_left() > 60:
        try:
            bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.01, 0.01)]
            de_result = differential_evolution(
                _objective, bounds,
                seed=42, maxiter=300, tol=1e-12,
                popsize=15, mutation=(0.3, 1.0), recombination=0.7,
                polish=False,
                callback=lambda xk, convergence: (convergence < 1e-14 or time.time() > _DEADLINE - 50)
            )
            if de_result.fun < best_val and np.isfinite(de_result.fun):
                best_val = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Multi-start L-BFGS-B with analytic gradient
    starts = [
        best_params.copy(),
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-2.0, 0.1, 0.001]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.2, 0.06, 0.0006]),
    ]
    
    rng = np.random.default_rng(7)
    for _ in range(15):
        starts.append(rng.normal(0, 0.5, 3))
    
    for start in starts:
        if _time_left() < 15:
            break
        try:
            res = minimize(
                _objective_grad, start, method='L-BFGS-B',
                jac=True, options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final Nelder-Mead polish
    if _time_left() > 10 and best_val < np.inf:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: One more L-BFGS-B from polished point
    if _time_left() > 5 and best_val < np.inf:
        try:
            res = minimize(
                _objective_grad, best_params, method='L-BFGS-B',
                jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if not np.isfinite(best_val):
        return np.array([-1.13, 0.04, 0.000308642])
    
    return best_params

# EVOLVE-BLOCK-END
