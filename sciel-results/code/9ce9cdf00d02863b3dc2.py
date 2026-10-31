"""Uncertainty inequality optimization via trust-region method with exact Hessian."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195


def _precompute_I():
    """I[n] = ∫ x^(2n) e^(-x²) dx for n=0..14"""
    I = np.zeros(15)
    I[0] = np.sqrt(np.pi)
    for n in range(1, 15):
        I[n] = (2.0 * n - 1.0) / 2.0 * I[n - 1]
    return I


_I = _precompute_I()


def _get_p(a0, a4, a8):
    """Compute P(x) coefficients with constraint P(0)=0."""
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    p = np.zeros(7)
    p[0] = 0.0
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p[2] = a4 + 210.0 * a8 + 13860.0 * a12
    p[3] = -28.0 * a8 - 6930.0 * a12
    p[4] = a8 + 1155.0 * a12
    p[5] = -66.0 * a12
    p[6] = a12
    return p


def _moments(p):
    """Compute m0 = ∫|P|²e^(-x²)dx and m2 = ∫x²|P|²e^(-x²)dx."""
    m0 = 0.0
    m2 = 0.0
    for i in range(7):
        if p[i] == 0.0:
            continue
        for j in range(7):
            if p[j] == 0.0:
                continue
            pij = p[i] * p[j]
            if i + j < 15:
                m0 += pij * _I[i + j]
            if i + j + 2 < 15:
                m2 += pij * _I[i + j + 2]
    return m0, m2


def _bound(params):
    p = _get_p(params[0], params[1], params[2])
    m0, m2 = _moments(p)
    if m0 <= 0 or m2 <= 0 or not np.isfinite(m0) or not np.isfinite(m2):
        return 1e18
    return m2 / m0


def _bound_grad(params):
    """Exact gradient of m2/m0 w.r.t. (a0,a4,a8)."""
    a0, a4, a8 = params
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    p = _get_p(a0, a4, a8)
    m0, m2 = _moments(p)
    if m0 <= 0 or not np.isfinite(m0):
        return m2 / m0 if m0 != 0 else 1e18, np.zeros(3)

    # dp/da for each parameter (7x3 matrix)
    # a12 = -(a0 + 3a4 + 105a8)/945
    # da12/da0 = -1/945, da12/da4 = -3/945, da12/da8 = -105/945
    dp_da = np.zeros((7, 3))
    dp_da[1] = [-7560.0 * (-1.0/945.0), -6.0 - 7560.0*(-3.0/945.0), -420.0 - 7560.0*(-105.0/945.0)]
    dp_da[2] = [13860.0 * (-1.0/945.0), 1.0 + 13860.0*(-3.0/945.0), 210.0 + 13860.0*(-105.0/945.0)]
    dp_da[3] = [-6930.0 * (-1.0/945.0), -6930.0*(-3.0/945.0), -28.0 - 6930.0*(-105.0/945.0)]
    dp_da[4] = [1155.0 * (-1.0/945.0), 1155.0*(-3.0/945.0), 1.0 + 1155.0*(-105.0/945.0)]
    dp_da[5] = [-66.0 * (-1.0/945.0), -66.0*(-3.0/945.0), -66.0*(-105.0/945.0)]
    dp_da[6] = [-1.0/945.0, -3.0/945.0, -105.0/945.0]

    # dm0/da_c = 2 * sum_{i,j} p_j * dp_i/da_c * I[i+j]
    # dm2/da_c = 2 * sum_{i,j} p_j * dp_i/da_c * I[i+j+2]
    dm0 = np.zeros(3)
    dm2 = np.zeros(3)
    for c in range(3):
        for i in range(7):
            dpi = dp_da[i, c]
            if dpi == 0.0:
                continue
            for j in range(7):
                if p[j] == 0.0:
                    continue
                idx0 = i + j
                idx2 = i + j + 2
                if idx0 < 15:
                    dm0[c] += 2.0 * p[j] * dpi * _I[idx0]
                if idx2 < 15:
                    dm2[c] += 2.0 * p[j] * dpi * _I[idx2]

    grad = (dm2 * m0 - m2 * dm0) / (m0 * m0)
    val = m2 / m0
    return val, grad


def _bound_hess(params):
    """Exact Hessian of m2/m0 w.r.t. (a0,a4,a8)."""
    a0, a4, a8 = params
    p = _get_p(a0, a4, a8)
    m0, m2 = _moments(p)
    if m0 <= 0 or not np.isfinite(m0):
        return np.zeros((3, 3))

    dp_da = np.zeros((7, 3))
    dp_da[1] = [8.0, 18.0, 420.0]
    dp_da[2] = [-14.0/1.5, -43.0, -1330.0]  # Recompute exactly
    # Exact values:
    dp_da[1, 0] = 7560.0/945.0    # 8
    dp_da[1, 1] = -6.0 + 7560.0*3.0/945.0  # -6+24=18
    dp_da[1, 2] = -420.0 + 7560.0*105.0/945.0  # -420+840=420
    dp_da[2, 0] = -13860.0/945.0   # -14.666...
    dp_da[2, 1] = 1.0 - 13860.0*3.0/945.0  # 1-44=-43
    dp_da[2, 2] = 210.0 - 13860.0*105.0/945.0  # 210-1540=-1330
    dp_da[3, 0] = 6930.0/945.0     # 7.333...
    dp_da[3, 1] = 6930.0*3.0/945.0  # 22
    dp_da[3, 2] = -28.0 + 6930.0*105.0/945.0  # -28+770=742
    dp_da[4, 0] = -1155.0/945.0    # -1.222...
    dp_da[4, 1] = -1155.0*3.0/945.0  # -3.666...
    dp_da[4, 2] = 1.0 - 1155.0*105.0/945.0  # 1-128.333=-127.333
    dp_da[5, 0] = 66.0/945.0
    dp_da[5, 1] = 66.0*3.0/945.0
    dp_da[5, 2] = 66.0*105.0/945.0
    dp_da[6, 0] = -1.0/945.0
    dp_da[6, 1] = -3.0/945.0
    dp_da[6, 2] = -105.0/945.0

    # dm0/da_c, dm2/da_c (first derivatives)
    dm0 = np.zeros(3)
    dm2 = np.zeros(3)
    for c in range(3):
        for i in range(7):
            dpi = dp_da[i, c]
            if dpi == 0.0:
                continue
            for j in range(7):
                if p[j] == 0.0:
                    continue
                idx0 = i + j
                idx2 = i + j + 2
                if idx0 < 15:
                    dm0[c] += 2.0 * p[j] * dpi * _I[idx0]
                if idx2 < 15:
                    dm2[c] += 2.0 * p[j] * dpi * _I[idx2]

    # Second derivatives: d²m_k/da_c/da_d
    # d²m0/da_c/da_d = 2 * sum_{i,j} dp_i/da_c * dp_j/da_d * I[i+j]
    # (symmetric in c,d)
    d2m0 = np.zeros((3, 3))
    d2m2 = np.zeros((3, 3))
    for c in range(3):
        for d in range(3):
            s0 = 0.0
            s2 = 0.0
            for i in range(7):
                for j in range(7):
                    dpic = dp_da[i, c]
                    dpjd = dp_da[j, d]
                    if dpic == 0.0 or dpjd == 0.0:
                        continue
                    idx0 = i + j
                    idx2 = i + j + 2
                    if idx0 < 15:
                        s0 += dpic * dpjd * _I[idx0]
                    if idx2 < 15:
                        s2 += dpic * dpjd * _I[idx2]
            d2m0[c, d] = 2.0 * s0
            d2m2[c, d] = 2.0 * s2

    # Hessian of f = m2/m0:
    # H_cd = (d2m2_cd * m0 - m2 * d2m0_cd) / m0² 
    #       + 2 * dm2_c * dm0_d / m0² - 2 * m2 * dm0_c * dm0_d / m0³
    H = np.zeros((3, 3))
    for c in range(3):
        for d in range(3):
            H[c, d] = (d2m2[c, d] * m0 - m2 * d2m0[c, d]) / (m0 * m0)
            H[c, d] += 2.0 * dm2[c] * dm0[d] / (m0 * m0)
            H[c, d] -= 2.0 * m2 * dm0[c] * dm0[d] / (m0 * m0 * m0)
    return H


def _trust_region_step(params, f, grad, hess, radius):
    """Solve min grad^T d + 0.5 d^T H d s.t. ||d|| <= radius."""
    # Regularize Hessian
    H = 0.5 * (hess + hess.T)
    eigvals, eigvecs = np.linalg.eigh(H)
    # Ensure positive definiteness
    min_eig = np.min(eigvals)
    if min_eig < 1e-10:
        H += (1e-8 - min_eig) * np.eye(3)
        eigvals, eigvecs = np.linalg.eigh(H)
    
    # Unconstrained minimizer
    d_uncon = np.linalg.solve(H, -grad)
    norm_d = np.linalg.norm(d_uncon)
    
    if norm_d <= radius:
        return d_uncon
    else:
        # Solve (H + λI)d = -grad with ||d|| = radius
        # Use bisection on λ
        lo, hi = 0.0, 1e6
        for _ in range(100):
            mid = (lo + hi) / 2.0
            d = np.linalg.solve(H + mid * np.eye(3), -grad)
            n = np.linalg.norm(d)
            if n > radius:
                lo = mid
            else:
                hi = mid
        d = np.linalg.solve(H + lo * np.eye(3), -grad)
        n = np.linalg.norm(d)
        if n > 0:
            d *= radius / n
        return d


def _bound_and_derivs(params):
    """Return (f, grad, hess)."""
    f, grad = _bound_grad(params)
    hess = _bound_hess(params)
    return f, grad, hess


def construct():
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = []
    # Around seed
    for _ in range(10):
        s = np.array([-1.13, 0.04, 0.000308642]) + rng.normal(0, 0.2, 3)
        starts.append(s)
    # Broader exploration
    for _ in range(30):
        s = np.array([rng.uniform(-4.0, 3.0), rng.uniform(-1.0, 1.0), rng.uniform(-0.05, 0.05)])
        starts.append(s)
    # Structured
    for a0 in [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]:
        for a4 in [-0.5, -0.2, 0.0, 0.2, 0.5]:
            for a8 in [-0.005, 0.0, 0.005]:
                starts.append(np.array([a0, a4, a8]))
    
    # Evaluate and sort starts
    scored = []
    for s in starts:
        if time.time() > _DEADLINE - 80:
            break
        try:
            val = _bound(s)
            if np.isfinite(val):
                scored.append((val, s))
                if val < best_val:
                    best_val = val
                    best_params = s.copy()
        except Exception:
            pass
    
    scored.sort(key=lambda x: x[0])
    
    # Trust-region optimization from top starts
    top_n = min(10, len(scored))
    for i in range(top_n):
        if time.time() > _DEADLINE - 30:
            break
        x = scored[i][1].copy()
        try:
            f, grad, hess = _bound_and_derivs(x)
            if not np.isfinite(f):
                continue
            radius = 1.0
            for it in range(200):
                if time.time() > _DEADLINE - 25:
                    break
                step = _trust_region_step(x, f, grad, hess, radius)
                x_new = x + step
                f_new, grad_new, hess_new = _bound_and_derivs(x_new)
                
                if not np.isfinite(f_new):
                    radius *= 0.5
                    continue
                
                predicted = f + grad @ step + 0.5 * step @ hess @ step
                actual = f_new - f
                rho = actual / predicted if abs(predicted) > 1e-15 else 0.0
                
                if rho > 0.75:
                    radius = min(radius * 2.0, 10.0)
                elif rho < 0.25:
                    radius *= 0.5
                else:
                    radius *= 0.8
                
                x = x_new
                f = f_new
                grad = grad_new
                hess = hess_new
                
                if np.linalg.norm(grad) < 1e-12 or radius < 1e-10:
                    break
                
                if f < best_val:
                    best_val = f
                    best_params = x.copy()
        except Exception:
            pass
    
    # Final L-BFGS-B polish
    if time.time() < _DEADLINE - 10:
        try:
            res = minimize(_bound_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Nelder-Mead final polish
    if time.time() < _DEADLINE - 5:
        try:
            nm_res = minimize(_bound, best_params, method='Nelder-Mead',
                            options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18})
            if np.isfinite(nm_res.fun) and nm_res.fun < best_val:
                best_val = nm_res.fun
                best_params = nm_res.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
