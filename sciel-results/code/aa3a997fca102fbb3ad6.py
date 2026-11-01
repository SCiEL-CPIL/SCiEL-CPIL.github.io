"""Direct analytic approach: build Hermite-Gaussian test function to minimize uncertainty product."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 195  # safety margin before 210s kill


def _hermite_coeffs():
    """
    Precompute the polynomial coefficients of H_0, H_4, H_8, H_12.
    H_n are physicists' Hermite polynomials.
    H_0 = 1
    H_4 = x^4 - 6x^2 + 3
    H_8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    H_12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    
    We represent P(x) = sum_{k=0}^{6} p_k * x^{2k}
    H_0: p = [1, 0, 0, 0, 0, 0, 0]
    H_4: p = [3, -6, 1, 0, 0, 0, 0]
    H_8: p = [105, -420, 210, -28, 1, 0, 0]
    H_12: p = [945, -7560, 13860, -6930, 1155, -66, 1]
    """
    H = np.zeros((4, 7))
    H[0, 0] = 1.0
    H[1, 0] = 3.0; H[1, 1] = -6.0; H[1, 2] = 1.0
    H[2, 0] = 105.0; H[2, 1] = -420.0; H[2, 2] = 210.0; H[2, 3] = -28.0; H[2, 4] = 1.0
    H[3, 0] = 945.0; H[3, 1] = -7560.0; H[3, 2] = 13860.0; H[3, 3] = -6930.0
    H[3, 4] = 1155.0; H[3, 5] = -66.0; H[3, 6] = 1.0
    return H


def _compute_moments():
    """
    Precompute I[n] = integral x^{2n} e^{-x^2} dx = (2n-1)!! * sqrt(pi) / 2^n
    for n = 0, 1, ..., 12 (need up to x^{24} since max degree is 12, |P|^2 has degree 24)
    """
    I = np.zeros(13)
    sp = np.sqrt(np.pi)
    for n in range(13):
        if n == 0:
            I[n] = sp
        else:
            I[n] = I[n-1] * (2.0*n - 1.0) / (4.0 * n)
    return I


_H = _hermite_coeffs()
_I = _compute_moments()
_SIGN = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])


def _bound(params):
    """
    Compute A(f)*A(F) for g(x) = P(x) * exp(-x^2/2)
    where P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12 with a12 = -(a0+3a4+105a8)/945
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H
    
    m0 = 0.0
    m2 = 0.0
    for i in range(7):
        pi = p[i]
        if pi == 0.0:
            continue
        for j in range(7):
            pij = pi * p[j]
            m0 += pij * _I[i+j]
            m2 += pij * _I[1+i+j]
    
    q = p * _SIGN
    
    n0 = 0.0
    n2 = 0.0
    for i in range(7):
        qi = q[i]
        if qi == 0.0:
            continue
        for j in range(7):
            qij = qi * q[j]
            n0 += 2.0 * np.pi * qij * _I[i+j]
            n2 += 2.0 * np.pi * qij * _I[1+i+j]
    
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18
    
    return np.sqrt(vf * vF)


def _bound_grad(params):
    """Compute bound and its gradient analytically."""
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H
    
    m0 = 0.0
    m2 = 0.0
    for i in range(7):
        pi = p[i]
        if pi == 0.0:
            continue
        for j in range(7):
            pij = pi * p[j]
            m0 += pij * _I[i+j]
            m2 += pij * _I[1+i+j]
    
    q = p * _SIGN
    
    n0 = 0.0
    n2 = 0.0
    for i in range(7):
        qi = q[i]
        if qi == 0.0:
            continue
        for j in range(7):
            qij = qi * q[j]
            n0 += 2.0 * np.pi * qij * _I[i+j]
            n2 += 2.0 * np.pi * qij * _I[1+i+j]
    
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    S0 = np.zeros(7)
    S2 = np.zeros(7)
    T0 = np.zeros(7)
    T2 = np.zeros(7)
    for i in range(7):
        for j in range(7):
            S0[i] += p[j] * _I[i+j]
            S2[i] += p[j] * _I[1+i+j]
            T0[i] += q[j] * _I[i+j]
            T2[i] += q[j] * _I[1+i+j]
    
    grad = np.zeros(3)
    for c in range(3):
        dp = _H[c, :] + da12[c] * _H[3, :]
        
        dm0 = 2.0 * np.dot(dp, S0)
        dm2 = 2.0 * np.dot(dp, S2)
        
        dq = dp * _SIGN
        
        dn0 = 4.0 * np.pi * np.dot(dq, T0)
        dn2 = 4.0 * np.pi * np.dot(dq, T2)
        
        dvm = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn = (dn2 * n0 - n2 * dn0) / (n0 * n0)
        
        grad[c] = (vF * dvm + vf * dvn) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    return _bound(params)


def _objective_grad(params):
    return _bound_grad(params)


def _newton_polish(x0, max_iter=80, tol=1e-15):
    """Newton's method with numerical Hessian and line search."""
    x = x0.copy()
    f, g = _objective_grad(x)
    if not np.isfinite(f):
        return x
    for _ in range(max_iter):
        if time.time() > _DEADLINE - 3:
            break
        if np.max(np.abs(g)) < tol:
            break
        h = 1e-7
        H = np.zeros((3, 3))
        for i in range(3):
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            _, gp = _objective_grad(xp)
            _, gm = _objective_grad(xm)
            H[i, :] = (gp - gm) / (2.0 * h)
        # Add regularization for numerical stability
        H += 1e-12 * np.eye(3)
        try:
            dx = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            dx = -0.01 * g
        step = 1.0
        for _ in range(30):
            x_try = x + step * dx
            f_try, g_try = _objective_grad(x_try)
            if np.isfinite(f_try) and f_try < f:
                x = x_try
                f = f_try
                g = g_try
                break
            step *= 0.5
        if step < 1e-12:
            break
    return x


def _bfgs_polish(x0, max_iter=20000, tol=1e-15):
    """BFGS polish with very tight tolerances."""
    from scipy.optimize import minimize
    res = minimize(_objective_grad, x0, method='BFGS',
                  jac=True, options={'maxiter': max_iter, 'gtol': tol, 'disp': False})
    if res.fun < _bound(x0) and np.isfinite(res.fun):
        return res.x
    return x0


def _nelder_polish(x0, max_iter=100000, tol=1e-20):
    """Nelder-Mead polish for final refinement."""
    from scipy.optimize import minimize
    res = minimize(_objective, x0, method='Nelder-Mead',
                  options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True})
    if res.fun < _bound(x0) and np.isfinite(res.fun):
        return res.x
    return x0


def construct():
    """
    Direct approach: use the analytic structure to find optimal coefficients.
    
    Strategy:
    1. Multi-start L-BFGS-B with exact gradient
    2. Grid refinement around best
    3. SHGO for global search
    4. BFGS polish
    5. Nelder-Mead polish
    6. Newton polish for highest precision
    """
    from scipy.optimize import minimize, shgo
    
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    # Starting points: seed + structured exploration
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),   # seed
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.5, 0.03, 0.0002]),
        np.array([-0.8, 0.06, 0.0008]),
        np.array([-2.0, 0.02, 0.0001]),
        np.array([-1.2, 0.035, 0.0003]),
        np.array([-1.1, 0.045, 0.00035]),
        np.array([-0.5, 0.08, 0.001]),
        np.array([-3.0, 0.01, 0.00005]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -0.05, 0.001]),
        np.array([-2.5, 0.015, 0.00015]),
        np.array([-1.15, 0.042, 0.00032]),
        np.array([-1.05, 0.038, 0.00028]),
        np.array([-1.3, 0.035, 0.00035]),
        np.array([-0.9, 0.055, 0.0006]),
        np.array([-1.7, 0.025, 0.00025]),
        np.array([-0.7, 0.07, 0.0009]),
        np.array([-2.2, 0.018, 0.00018]),
        np.array([-1.4, 0.032, 0.00032]),
        np.array([-0.6, 0.085, 0.0012]),
        np.array([-2.8, 0.012, 0.00012]),
        np.array([-1.18, 0.041, 0.00031]),
        np.array([-1.08, 0.039, 0.00029]),
        # Additional diverse points
        np.array([-1.12, 0.038, 0.00029]),
        np.array([-1.14, 0.042, 0.00031]),
        np.array([-1.06, 0.045, 0.00036]),
        np.array([-1.22, 0.036, 0.00034]),
        np.array([-0.95, 0.052, 0.00052]),
    ]
    
    # Add random starts with better distribution
    rng = np.random.default_rng(42)
    for _ in range(50):
        a0 = -np.exp(rng.uniform(0.4, 2.2))
        a4 = np.exp(rng.uniform(-3.5, 0.0)) * 0.12
        a8 = np.exp(rng.uniform(-5.5, -0.5)) * 0.0012
        starts.append(np.array([a0, a4, a8]))
    
    # Phase 1: Multi-start L-BFGS-B
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Coarse grid refinement around best found
    if time.time() < _DEADLINE - 35:
        center = best_params.copy()
        scales = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
        for scale in scales:
            if time.time() > _DEADLINE - 15:
                break
            for axis in range(3):
                for direction in [-1, 1]:
                    if time.time() > _DEADLINE - 10:
                        break
                    trial = center.copy()
                    trial[axis] += direction * scale
                    try:
                        val = _bound(trial)
                        if val < best_val:
                            best_val = val
                            best_params = trial.copy()
                            res = minimize(_objective_grad, trial, method='L-BFGS-B',
                                          jac=True, options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16})
                            if res.fun < best_val and np.isfinite(res.fun):
                                best_val = res.fun
                                best_params = res.x.copy()
                    except Exception:
                        pass
    
    # Phase 3: SHGO for global search if time permits
    if time.time() < _DEADLINE - 40:
        try:
            bounds = [(-10, 10), (-1, 1), (-0.01, 0.01)]
            res_shgo = shgo(_objective, bounds, options={'maxiter': 150, 'maxfev': 8000})
            if res_shgo.fun < best_val and np.isfinite(res_shgo.fun):
                best_val = res_shgo.fun
                best_params = res_shgo.x.copy()
                res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                              jac=True, options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: BFGS polish
    if time.time() < _DEADLINE - 12:
        try:
            best_params = _bfgs_polish(best_params, max_iter=30000, tol=1e-16)
            val = _bound(best_params)
            if val < best_val and np.isfinite(val):
                best_val = val
        except Exception:
            pass
    
    # Phase 5: Nelder-Mead polish
    if time.time() < _DEADLINE - 8:
        try:
            best_params = _nelder_polish(best_params, max_iter=200000, tol=1e-20)
            val = _bound(best_params)
            if val < best_val and np.isfinite(val):
                best_val = val
        except Exception:
            pass
    
    # Phase 6: Newton polish for highest precision
    if time.time() < _DEADLINE - 3:
        try:
            best_params = _newton_polish(best_params, max_iter=100, tol=1e-16)
            final_val = _bound(best_params)
            if final_val < best_val and np.isfinite(final_val):
                best_val = final_val
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
