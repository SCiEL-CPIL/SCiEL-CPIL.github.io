import time
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 610.0
_K = 10
_NTH = 70
_NPHI = 130


def _build_harmonics():
    """Return Y (nphi*nth, 2k+1), weights (nphi*nth,) for quadrature."""
    k = _K
    mu, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI

    Y = np.zeros((_NTH, _NPHI, 2 * k + 1), dtype=np.float64)
    for idx, m in enumerate(range(-k, k + 1)):
        am = abs(m)
        N = np.sqrt((2 * k + 1) / (4.0 * np.pi)
                    * float(factorial(k - am, exact=True))
                    / float(factorial(k + am, exact=True)))
        P = lpmv(am, k, mu)
        if m == 0:
            Y[:, :, idx] = N * P[:, None]
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * N * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * N * P[:, None] * np.sin(am * phi)[None, :]

    Y_flat = Y.reshape(_NTH * _NPHI, 2 * k + 1)
    w_flat = np.outer(w, np.full(_NPHI, dphi)).ravel()

    return Y_flat, w_flat


def _ratio_val_and_grad(c, Y, w):
    """Compute R^4 = S4/S2^2 and its gradient."""
    f = Y @ c
    f2 = f * f
    f3 = f2 * f
    
    S2 = np.dot(w, f2)
    S4 = np.dot(w, f2 * f2)
    
    if S2 < 1e-15:
        return 1e12, np.zeros_like(c)
    
    val = S4 / (S2 * S2)
    
    Yt_wf3 = Y.T @ (w * f3)
    Yt_wf = Y.T @ (w * f)
    
    dS4 = 4.0 * Yt_wf3
    dS2 = 2.0 * Yt_wf
    
    grad = dS4 / (S2 * S2) - 2.0 * S4 * dS2 / (S2 * S2 * S2)
    
    return val, grad


def _hessian_vec_prod(c, Y, w, v):
    """
    Compute Hessian-vector product for R^4 = S4/S2^2.
    
    S2 = w^T (Yc)^2, S4 = w^T (Yc)^4
    R4 = S4 / S2^2
    
    dS2/dc = 2 Y^T w (Yc)
    dS4/dc = 4 Y^T w (Yc)^3
    
    d2S2/dc2 = 2 Y^T diag(w) Y
    d2S4/dc2 = 4 Y^T diag(w * 3f^2) Y  where f = Yc
    
    H = (d2S4 * S2^2 - S4 * 2 * dS2 * dS2^T * 2) / S2^4 
      + S4 * (dS2 * dS2^T * 4) / S2^5  ... let me derive more carefully.
      
    Let g = S4, h = S2^2. Then R4 = g/h.
    dR4 = (dg*h - g*dh)/h^2
    d2R4 = (d2g*h + g*d2h - 2*dg*dh - g*dh^2/h)/h^2
         = (d2g*h - g*d2h + 2*g*dh^2/h - 2*dg*dh)/h^2
    
    Where:
    dg = dS4, dh = 2*S2*dS2
    d2g = 4*Y^T diag(w*3*f^2) Y
    d2h = 2*(dS2*dS2^T + S2*d2S2)
    """
    f = Y @ c
    f2 = f * f
    
    S2 = np.dot(w, f2)
    S4 = np.dot(w, f2 * f2)
    
    if S2 < 1e-15:
        return np.zeros_like(v)
    
    h = S2 * S2
    
    # dS2 = 2 * Y^T (w * f)
    dS2 = 2.0 * (Y.T @ (w * f))
    
    # dS4 = 4 * Y^T (w * f^3)
    dS4 = 4.0 * (Y.T @ (w * f * f2))
    
    # d2S2 * v = 2 * Y^T (w * (Y @ v))
    Yv = Y @ v
    d2S2_v = 2.0 * (Y.T @ (w * Yv))
    
    # d2S4 * v = 4 * Y^T (w * (3*f^2 * (Y @ v)))
    d2S4_v = 4.0 * (Y.T @ (w * (3.0 * f2 * Yv)))
    
    # dh = 2*S2*dS2
    dh = 2.0 * S2 * dS2
    
    # d2h * v = 2*(dS2*dS2^T v + S2*d2S2_v)
    d2h_v = 2.0 * (np.dot(dS2, v) * dS2 + S2 * d2S2_v)
    
    # H*v = (d2g*h - g*d2h + 2*g*dh^2/h - 2*dg*dh)/h^2 applied to v
    # = (d2S4_v * h - S4 * d2h_v + 2*S4*np.dot(dh,v)*dh - 2*dS4*np.dot(dh,v))/h^2
    
    dot_dh_v = np.dot(dh, v)
    
    Hv = (d2S4_v * h - S4 * d2h_v + 2.0 * S4 * dot_dh_v * dh - 2.0 * dS4 * dot_dh_v) / (h * h)
    
    return Hv


def _projected_newton_sphere(x0, Y, w, max_iter=1000, tol=1e-16, time_limit=None):
    """
    Projected Newton method on the unit sphere for minimizing R^4.
    Uses exact Hessian-vector products for the quartic objective.
    """
    n = len(x0)
    x = x0 / np.linalg.norm(x0)
    f_val, g = _ratio_val_and_grad(x, Y, w)
    
    def project_tangent(v, x):
        return v - np.dot(v, x) * x
    
    g_tan = project_tangent(g, x)
    
    # Damped Newton with line search
    step_size = 1.0
    for it in range(max_iter):
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Compute Hessian-vector product for the Newton direction
        # Solve H d = -g_tan approximately using conjugate gradient
        # Since H may not be positive definite, use damped Newton
        
        # First, try a simple gradient descent step if Newton fails
        d = -g_tan
        d = project_tangent(d, x)
        
        dn = np.linalg.norm(d)
        if dn < tol:
            break
        
        # Normalize step direction
        d = d / dn
        
        # Line search: find alpha such that f(x + alpha*d) < f(x)
        # and x + alpha*d is on the sphere
        alpha = step_size
        f_best = f_val
        x_best = x
        g_best = g
        
        for ls_iter in range(50):
            # Take step and project to sphere
            x_trial = x + alpha * d
            nrm = np.linalg.norm(x_trial)
            if nrm < 1e-15:
                break
            x_trial = x_trial / nrm
            
            f_trial, g_trial = _ratio_val_and_grad(x_trial, Y, w)
            
            # Armijo condition
            if f_trial < f_val - 1e-4 * alpha * np.dot(g_tan, d):
                f_best = f_trial
                x_best = x_trial
                g_best = g_trial
                break
            
            alpha *= 0.5
            if alpha < 1e-15:
                break
        
        if alpha < 1e-15 or f_best >= f_val:
            # Backtracking failed, try smaller step or gradient direction
            d = -g_tan
            d = project_tangent(d, x)
            dn = np.linalg.norm(d)
            if dn < tol:
                break
            d = d / dn
            
            # Try very small step
            alpha = 1e-4
            x_trial = x + alpha * d
            nrm = np.linalg.norm(x_trial)
            if nrm > 1e-15:
                x_trial = x_trial / nrm
                f_trial, g_trial = _ratio_val_and_grad(x_trial, Y, w)
                if f_trial < f_val:
                    x = x_trial
                    f_val = f_trial
                    g = g_trial
                    g_tan = project_tangent(g, x)
                    step_size = alpha
                    continue
        
        x = x_best
        f_val = f_best
        g = g_best
        g_tan = project_tangent(g, x)
        
        # Adjust step size based on line search
        if alpha > 0.5:
            step_size = min(2.0 * step_size, 2.0)
        else:
            step_size = alpha
        
        # Convergence check
        if np.linalg.norm(g_tan) < tol:
            break
    
    return x, f_val


def rudin_problem10() -> np.ndarray:
    Y, w = _build_harmonics()
    k = _K
    n = 2 * k + 1
    
    best_c = np.zeros(n)
    best_val = np.inf

    # 1. Diverse Initial Guesses
    inits = []
    v0 = np.zeros(n); v0[k] = 1.0; inits.append(v0)
    v10 = np.zeros(n); v10[2*k] = 1.0; inits.append(v10)
    v10s = np.zeros(n); v10s[k-10] = 1.0; inits.append(v10s)
    v_u = np.ones(n); inits.append(v_u / np.linalg.norm(v_u))
    
    rng = np.random.default_rng(123)
    for _ in range(15):
        v = rng.standard_normal(n)
        inits.append(v / np.linalg.norm(v))
    
    # 2. Projected Newton Optimization from Each Init
    time_limit_tr = _DEADLINE - 200.0
    for x0 in inits:
        if time.time() > time_limit_tr:
            break
        try:
            c_opt, val_opt = _projected_newton_sphere(x0, Y, w, max_iter=1000, tol=1e-16, time_limit=time_limit_tr)
            if val_opt < best_val and np.isfinite(val_opt):
                best_val = val_opt
                best_c = c_opt.copy()
        except Exception:
            pass
    
    # 3. Global Optimization: Differential Evolution (if time remains)
    def de_obj(x):
        norm = np.linalg.norm(x)
        if norm < 1e-12:
            return 1e10
        x_norm = x / norm
        val, _ = _ratio_val_and_grad(x_norm, Y, w)
        return val

    bounds = [(-1.0, 1.0)] * n
    
    try:
        if time.time() < _DEADLINE - 100.0:
            res_de = differential_evolution(
                de_obj, bounds, seed=42, maxiter=2000, popsize=25, 
                tol=1e-10, mutation=(0.5, 1.0), recombination=0.9,
                polish=True
            )
            c_de = res_de.x
            norm = np.linalg.norm(c_de)
            if norm > 1e-12:
                c_de_norm = c_de / norm
                val_de, _ = _ratio_val_and_grad(c_de_norm, Y, w)
                if val_de < best_val and np.isfinite(val_de):
                    best_val = val_de
                    best_c = c_de_norm.copy()
    except Exception:
        pass
    
    # 4. Final High-Precision Polish
    if time.time() < _DEADLINE - 10.0:
        def fun_final(c):
            val, grad = _ratio_val_and_grad(c, Y, w)
            return val, grad
        try:
            res_final = minimize(fun_final, best_c, method="L-BFGS-B", jac=True,
                                 options={"maxiter": 5000, "maxls": 60, "ftol": 1e-18, "gtol": 1e-16})
            c_final = res_final.x
            val_final, _ = _ratio_val_and_grad(c_final, Y, w)
            if val_final < best_val and np.isfinite(val_final):
                best_val = val_final
                best_c = c_final.copy()
        except Exception:
            pass
        
        # Additional projected Newton polish
        try:
            c_polish, val_polish = _projected_newton_sphere(best_c, Y, w, max_iter=2000, tol=1e-18, time_limit=_DEADLINE - 5.0)
            if val_polish < best_val and np.isfinite(val_polish):
                best_val = val_polish
                best_c = c_polish.copy()
        except Exception:
            pass
        
        # Final L-BFGS-B polish with tighter tolerances
        try:
            def fun_tight(c):
                val, grad = _ratio_val_and_grad(c, Y, w)
                return val, grad
            res_tight = minimize(fun_tight, best_c, method="L-BFGS-B", jac=True,
                                 options={"maxiter": 10000, "maxls": 100, "ftol": 1e-20, "gtol": 1e-18})
            c_tight = res_tight.x
            val_tight, _ = _ratio_val_and_grad(c_tight, Y, w)
            if val_tight < best_val and np.isfinite(val_tight):
                best_val = val_tight
                best_c = c_tight.copy()
        except Exception:
            pass

    # Normalize and validate
    out = best_c.astype(np.float64)
    nrm = np.linalg.norm(out)
    if nrm < 1e-12:
        out = np.ones(n, dtype=np.float64)
    else:
        out = out / nrm

    if out.shape != (21,):
        out = np.ones(21, dtype=np.float64)
    if np.any(np.isnan(out)) or np.any(np.isinf(out)):
        out = np.ones(21, dtype=np.float64)
    
    max_abs = np.max(np.abs(out))
    if max_abs < 1e-3:
        out = out / max_abs if max_abs > 0 else np.ones(21)
    if max_abs > 1e5:
        out = out / max_abs

    return out
