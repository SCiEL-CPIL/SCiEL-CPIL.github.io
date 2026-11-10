import time
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import differential_evolution

_DEADLINE = time.time() + 620.0
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
    """
    Compute R^4 = S4/S2^2 and its gradient projected onto the tangent space of the unit sphere.
    Returns (val, grad_tangent) where val = R^4 and grad_tangent is the projected gradient.
    """
    f = Y @ c
    f2 = f * f
    f3 = f2 * f
    f4 = f2 * f2

    S2 = np.dot(w, f2)
    S4 = np.dot(w, f4)

    if S2 < 1e-15:
        return 1e12, np.zeros_like(c)

    val = S4 / (S2 * S2)

    # Gradient of R^4 w.r.t. c:
    # dS4/dc = 4 * Y^T (w * f^3)
    # dS2/dc = 2 * Y^T (w * f)
    # d(R^4)/dc = (4*S2*dS4/dc - 2*S4*dS2/dc) / S2^4
    Yt_wf3 = Y.T @ (w * f3)
    Yt_wf = Y.T @ (w * f)

    grad_full = 4.0 * (S2 * Yt_wf3 - S4 * Yt_wf) / (S2 * S2 * S2)

    # Project onto tangent space: grad_t = grad_full - (grad_full . c) * c
    dot = np.dot(grad_full, c)
    grad_t = grad_full - dot * c

    return val, grad_t


def _hessian_vec_prod(c, Y, w, v):
    """
    Compute Hessian-vector product of R^4 = S4/S2^2 projected to tangent space.
    Uses the second variation formula for the quartic objective.
    """
    f = Y @ c
    f2 = f * f
    f3 = f2 * f
    f4 = f2 * f2

    S2 = np.dot(w, f2)
    S4 = np.dot(w, f4)

    if S2 < 1e-15:
        return np.zeros_like(v)

    # Second derivatives:
    # d2S4/dc^2 = 4 * Y^T diag(w * 3f2) Y
    # d2S2/dc^2 = 2 * Y^T diag(w) Y
    # d2(R^4)/dc^2 = [4*S2*d2S4 - 2*S4*d2S2 - 8*(S2*dS4 - S4*dS2)^T(c) ... ] / S2^4
    # More precisely, using quotient rule for hessian:
    # Let g = S4, h = S2^2, so R^4 = g/h
    # d2(g/h) = (h*d2g - 2*dg^T * dh + g*d2h) / h^2 - 2*(dg^T/h - g*dh^T/h^2) * dh^T / h^2
    # Simplified: H = (S2^2 * d2S4 - 4*S2*dS4*dS2^T + S4*2*dS2*dS2^T) / S2^4
    #           - 2*(S2*dS4 - S4*dS2)/S2^4 * (S2*dS4 - S4*dS2)^T
    
    # Compute dS4 and dS2 as column vectors
    dS4 = 4.0 * Y.T @ (w * f3)
    dS2 = 2.0 * Y.T @ (w * f)
    
    # d2S4 * v = 4 * Y^T (w * 3*f2 * (Y @ v))
    Yv = Y @ v
    d2S4_v = 4.0 * Y.T @ (w * 3.0 * f2 * Yv)
    
    # d2S2 * v = 2 * Y^T (w * (Y @ v))
    d2S2_v = 2.0 * Y.T @ (w * Yv)
    
    # Numerator of first part
    num1 = S2 * S2 * d2S4_v - 4.0 * S2 * (np.dot(dS4, v) * dS2) + S4 * 2.0 * (np.dot(dS2, v) * dS2)
    
    # Second part: -2 * a * a^T where a = (S2*dS4 - S4*dS2)/S2^2
    a = (S2 * dS4 - S4 * dS2) / (S2 * S2)
    num2 = -2.0 * np.dot(a, v) * a
    
    H_v = (num1 + num2) / (S2 * S2)
    
    # Project to tangent space
    H_v -= np.dot(H_v, c) * c
    H_v -= np.dot(H_v, v) * c  # Additional projection for symmetry
    
    return H_v


def _trust_region_sphere(c0, Y, w, max_iter=200, tol=1e-14):
    """
    Projected Newton method on the unit sphere using exact Hessian-vector products.
    Minimizes R^4 = ||f||_4^4 / ||f||_2^4.
    """
    c = c0.copy()
    n = len(c)
    norm = np.linalg.norm(c)
    if norm < 1e-12:
        return c0, np.inf
    c = c / norm

    best_val, best_c = np.inf, c.copy()
    val, grad = _ratio_val_and_grad(c, Y, w)
    if not np.isfinite(val):
        return c0, np.inf
    best_val = val
    best_c = c.copy()

    delta = 1.0  # trust region radius

    for it in range(max_iter):
        if time.time() > _DEADLINE - 30.0:
            break

        val, grad = _ratio_val_and_grad(c, Y, w)
        if not np.isfinite(val):
            break

        if val < best_val:
            best_val = val
            best_c = c.copy()

        gnorm = np.linalg.norm(grad)
        if gnorm < tol:
            break

        # Solve Newton step using CG with Hessian-vector products
        # Solve H * s = -grad where H is the projected Hessian
        s = np.zeros(n)
        r = -grad.copy()
        p = r.copy()
        rr = np.dot(r, r)
        if rr < 1e-30:
            break

        cg_iter = 0
        for cg_it in range(min(n, 80)):
            Hp = _hessian_vec_prod(c, Y, w, p)
            pHp = np.dot(p, Hp)
            if pHp <= 1e-30:
                break
            alpha = rr / pHp
            s_new = s + alpha * p
            if np.linalg.norm(s_new) > delta:
                # Scale to trust region boundary
                s_new = s_new * (delta / np.linalg.norm(s_new))
                break
            r = r - alpha * Hp
            rr_new = np.dot(r, r)
            if np.sqrt(rr_new) < tol * 10:
                s = s_new
                break
            beta = rr_new / rr
            p = r + beta * p
            rr = rr_new
            s = s_new
            cg_iter += 1

        # Ensure s is in tangent space
        s -= np.dot(s, c) * c

        # If CG failed or s is too small, use steepest descent direction
        if np.linalg.norm(s) < 1e-12:
            s = -grad / (gnorm + 1e-30) * min(delta, gnorm / (gnorm + 1e-30))

        # Evaluate at trial point
        c_trial = c + s
        nrm = np.linalg.norm(c_trial)
        if nrm < 1e-12:
            delta *= 0.5
            continue
        c_trial = c_trial / nrm

        val_trial, _ = _ratio_val_and_grad(c_trial, Y, w)
        if not np.isfinite(val_trial):
            delta *= 0.5
            continue

        # Accept or reject
        actual_reduction = val - val_trial
        # Use quadratic model: predicted = grad^T s + 0.5 s^T H s
        # Approximate s^T H s using the CG solution
        Hs = _hessian_vec_prod(c, Y, w, s)
        predicted_reduction = np.dot(grad, s) + 0.5 * np.dot(s, Hs)

        if actual_reduction > 0.25 * predicted_reduction:
            # Accept step
            c = c_trial
            delta = min(2.0 * delta, 5.0)
        else:
            # Reject step
            delta *= 0.5

    return best_c, best_val


def _objective_ratio(c, Y, w):
    """Compute R = ||f||_4 / ||f||_2"""
    f = Y @ c
    f2 = f * f
    S2 = np.dot(w, f2)
    if S2 < 1e-15:
        return np.inf
    S4 = np.dot(w, f2 * f2)
    return (S4 ** 0.25) / np.sqrt(S2)


def _unconstrained_objective(x, Y, w):
    """
    Objective for global optimizers working in [-1, 1] bounds.
    Maps x to unit sphere.
    """
    c = x
    norm = np.linalg.norm(c)
    if norm < 1e-12:
        return 1e10
    c_norm = c / norm
    f = Y @ c_norm
    f2 = f * f
    S2 = np.dot(w, f2)
    if S2 < 1e-15:
        return 1e10
    S4 = np.dot(w, f2 * f2)
    return S4 / (S2 * S2)


def rudin_problem10() -> np.ndarray:
    Y, w = _build_harmonics()
    k = _K
    n = 2 * k + 1

    best_c = np.zeros(n)
    best_val = np.inf

    # 1. Generate diverse initial guesses
    inits = []
    # m=0 only
    v0 = np.zeros(n); v0[k] = 1.0; inits.append(v0)
    # m=10 only (cos)
    v10 = np.zeros(n); v10[2*k] = 1.0; inits.append(v10)
    # m=10 only (sin)
    v10s = np.zeros(n); v10s[k-10] = 1.0; inits.append(v10s)
    # m=9 only
    v9 = np.zeros(n); v9[2*k-1] = 1.0; inits.append(v9)
    v9s = np.zeros(n); v9s[k-9] = 1.0; inits.append(v9s)
    # m=8 only
    v8 = np.zeros(n); v8[2*k-2] = 1.0; inits.append(v8)
    v8s = np.zeros(n); v8s[k-8] = 1.0; inits.append(v8s)
    # Uniform
    v_u = np.ones(n); inits.append(v_u / np.linalg.norm(v_u))
    # Random
    rng = np.random.default_rng(123)
    for _ in range(60):
        v = rng.standard_normal(n)
        inits.append(v / np.linalg.norm(v))

    # 2. Local Optimization from Inits using projected Newton method
    for x0 in inits:
        if time.time() > _DEADLINE - 150.0:
            break
        try:
            c_opt, val_opt = _trust_region_sphere(x0, Y, w, max_iter=400, tol=1e-15)
            val_r = _objective_ratio(c_opt, Y, w)
            if val_r < best_val and np.isfinite(val_r):
                best_val = val_r
                best_c = c_opt.copy()
        except Exception:
            pass

    # 3. Global Optimization: Differential Evolution
    bounds = [(-1.0, 1.0)] * n
    de_obj = lambda x: _unconstrained_objective(x, Y, w)

    try:
        if time.time() < _DEADLINE - 180.0:
            res_de = differential_evolution(
                de_obj, bounds, seed=42, maxiter=1500, popsize=25,
                tol=1e-10, mutation=(0.5, 1.0), recombination=0.9,
                polish=False
            )
            c_de = res_de.x
            norm = np.linalg.norm(c_de)
            if norm > 1e-12:
                c_de_norm = c_de / norm
                # Polish with projected Newton method
                c_polished, val_polished = _trust_region_sphere(c_de_norm, Y, w, max_iter=400, tol=1e-15)
                val_de = _objective_ratio(c_polished, Y, w)
                if val_de < best_val and np.isfinite(val_de):
                    best_val = val_de
                    best_c = c_polished.copy()
    except Exception:
        pass

    # 4. Final high-precision polish with tighter tolerance
    if time.time() < _DEADLINE - 20.0:
        try:
            c_final, val_final = _trust_region_sphere(best_c, Y, w, max_iter=1000, tol=1e-17)
            val_final_r = _objective_ratio(c_final, Y, w)
            if val_final_r < best_val and np.isfinite(val_final_r):
                best_val = val_final_r
                best_c = c_final.copy()
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
