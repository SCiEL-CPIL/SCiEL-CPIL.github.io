"""
Optimized Hermite-Gaussian test function for uncertainty inequality.
Uses analytic gradient and robust multi-start optimization with high-precision polish.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 195  # Safety margin before 210s kill

def _hermite_coeffs():
    """
    Precompute the polynomial coefficients of H_0, H_4, H_8, H_12.
    H_n are physicists' Hermite polynomials.
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
    Precompute I[n] = integral x^{2n} e^{-x^2} dx
    I[n] = (2n-1)!! * sqrt(pi) / 2^n
    Need up to n=12 (for x^24 in |P|^2 where P has degree 12)
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
    
    # Vectorized computation using outer products
    P = np.outer(p, p)
    I_mat = _I[np.arange(7)[:, None] + np.arange(7)[None, :]]
    I_mat2 = _I[1 + np.arange(7)[:, None] + np.arange(7)[None, :]]
    
    m0 = np.sum(P * I_mat)
    m2 = np.sum(P * I_mat2)
    
    q = p * _SIGN
    Q = np.outer(q, q)
    
    n0 = 2.0 * np.pi * np.sum(Q * I_mat)
    n2 = 2.0 * np.pi * np.sum(Q * I_mat2)
    
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18
    
    return np.sqrt(vf * vF)

def _bound_grad(params):
    """Compute bound and its gradient analytically. Vectorized version."""
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H
    
    # Vectorized computation
    P = np.outer(p, p)
    I_mat = _I[np.arange(7)[:, None] + np.arange(7)[None, :]]
    I_mat2 = _I[1 + np.arange(7)[:, None] + np.arange(7)[None, :]]
    
    m0 = np.sum(P * I_mat)
    m2 = np.sum(P * I_mat2)
    
    q = p * _SIGN
    Q = np.outer(q, q)
    
    n0 = 2.0 * np.pi * np.sum(Q * I_mat)
    n2 = 2.0 * np.pi * np.sum(Q * I_mat2)
    
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # Vectorized S0, S2, T0, T2
    S0 = p @ I_mat
    S2 = p @ I_mat2
    T0 = q @ I_mat
    T2 = q @ I_mat2
    
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

def _newton_polish(x0, max_iter=100, tol=1e-16):
    """Newton's method with numerical Hessian and line search."""
    x = x0.copy()
    f, g = _objective_grad(x)
    if not np.isfinite(f):
        return x
    for _ in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        if np.max(np.abs(g)) < tol:
            break
        h = 1e-8
        H = np.zeros((3, 3))
        for i in range(3):
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            _, gp = _objective_grad(xp)
            _, gm = _objective_grad(xm)
            H[i, :] = (gp - gm) / (2.0 * h)
        # Add regularization for numerical stability
        H += 1e-14 * np.eye(3)
        try:
            dx = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            dx = -0.01 * g
        step = 1.0
        for _ in range(40):
            x_try = x + step * dx
            f_try, g_try = _objective_grad(x_try)
            if np.isfinite(f_try) and f_try < f:
                x = x_try
                f = f_try
                g = g_try
                break
            step *= 0.5
        if step < 1e-14:
            break
    return x

def _bfgs_polish(x0, max_iter=10000, tol=1e-15):
    """BFGS polish with tight tolerances."""
    res = minimize(_objective_grad, x0, method='BFGS',
                  jac=True, options={'maxiter': max_iter, 'gtol': tol, 'disp': False})
    if res.fun < _bound(x0) and np.isfinite(res.fun):
        return res.x
    return x0

def construct():
    """
    Robust multi-start optimization with global search and high-precision polish.
    """
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    # Evaluate seed
    seed_val = _bound(best_params)
    best_val = seed_val
    
    # Phase 1: Multi-start L-BFGS-B
    # Generate diverse starting points
    starts = []
    # 1. Seed and immediate neighborhood
    rng = np.random.default_rng(42)
    for _ in range(50):
        starts.append(best_params + rng.normal(0, 0.1, 3))
    
    # 2. Structured grid around seed
    base = best_params.copy()
    for d0 in [-0.5, -0.2, -0.1, 0.1, 0.2, 0.5]:
        for d4 in [-0.02, -0.01, 0.01, 0.02]:
            for d8 in [-0.0001, 0.0001]:
                starts.append(base + np.array([d0, d4, d8]))
                
    # 3. Wider random search
    for _ in range(50):
        a0 = rng.uniform(-3.0, 3.0)
        a4 = rng.uniform(-0.5, 0.5)
        a8 = rng.uniform(-0.01, 0.01)
        starts.append(np.array([a0, a4, a8]))
        
    # 4. Specific heuristic points
    starts.append(np.array([-1.0, 0.05, 0.0005]))
    starts.append(np.array([-1.5, 0.03, 0.0002]))
    starts.append(np.array([-0.8, 0.06, 0.0008]))
    starts.append(np.array([-2.0, 0.02, 0.0001]))
    starts.append(np.array([0.0, 0.0, 0.0]))
    starts.append(np.array([1.0, -0.05, 0.0005]))
    starts.append(np.array([-3.0, 0.1, 0.001]))
    starts.append(np.array([3.0, -0.1, 0.001]))

    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Global Search with Differential Evolution if time permits
    if time.time() < _DEADLINE - 60:
        try:
            bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.05, 0.05)]
            # Use a smaller population and maxiter to fit in time
            res_de = differential_evolution(_objective, bounds, 
                                            maxiter=50, popsize=15, 
                                            tol=1e-8, seed=42, 
                                            polish=False, init='sobol')
            if res_de.fun < best_val and np.isfinite(res_de.fun):
                best_val = res_de.fun
                best_params = res_de.x.copy()
                # Polish the DE result
                res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                              jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
        except Exception:
            pass

    # Phase 3: High-precision Polish
    if time.time() < _DEADLINE - 10:
        try:
            best_params = _bfgs_polish(best_params, max_iter=5000, tol=1e-16)
            val = _bound(best_params)
            if val < best_val and np.isfinite(val):
                best_val = val
        except Exception:
            pass

    if time.time() < _DEADLINE - 5:
        try:
            best_params = _newton_polish(best_params, max_iter=100, tol=1e-18)
            val = _bound(best_params)
            if val < best_val and np.isfinite(val):
                best_val = val
        except Exception:
            pass

    # Final check: ensure we return the best found
    final_val = _bound(best_params)
    if final_val < best_val:
        best_val = final_val
        
    return best_params
# EVOLVE-BLOCK-END
