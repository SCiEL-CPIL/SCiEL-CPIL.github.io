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


def _objective_ratio_grad(c, Y, w):
    """Compute R^4 and its gradient."""
    f = Y @ c
    f2 = f * f
    f3 = f2 * f
    
    S2 = np.dot(w, f2)
    S4 = np.dot(w, f2 * f2)
    
    if S2 < 1e-15:
        return 1e12, np.zeros_like(c)
    
    # Minimize R^4 = S4 / S2^2
    val = S4 / (S2 * S2)
    
    # Gradient of S4: 4 * Y^T (w * f^3)
    # Gradient of S2: 2 * Y^T (w * f)
    # d(S4/S2^2) = (S2^2 * dS4 - S4 * 2*S2*dS2) / S2^4
    #            = dS4/S2^2 - 2*S4*dS2/S2^3
    
    Yt_wf3 = Y.T @ (w * f3)
    Yt_wf = Y.T @ (w * f)
    
    dS4 = 4.0 * Yt_wf3
    dS2 = 2.0 * Yt_wf
    
    grad = dS4 / (S2 * S2) - 2.0 * S4 * dS2 / (S2 * S2 * S2)
    
    return val, grad

def rudin_problem10() -> np.ndarray:
    Y, w = _build_harmonics()
    k = _K
    n = 2 * k + 1
    
    best_c = np.zeros(n)
    best_val = np.inf

    # 1. Symmetric Initial Guesses
    inits = []
    # m=0 only
    v0 = np.zeros(n); v0[k] = 1.0; inits.append(v0)
    # m=10 only (cos)
    v10 = np.zeros(n); v10[2*k] = 1.0; inits.append(v10)
    # m=10 only (sin)
    v10s = np.zeros(n); v10s[k-10] = 1.0; inits.append(v10s)
    # Uniform
    v_u = np.ones(n); inits.append(v_u / np.linalg.norm(v_u))
    # Random
    rng = np.random.default_rng(123)
    for _ in range(10):
        v = rng.standard_normal(n)
        inits.append(v / np.linalg.norm(v))

    # 2. Local Optimization from Inits (L-BFGS-B)
    for x0 in inits:
        if time.time() > _DEADLINE - 150.0:
            break
        def fun(c):
            val, grad = _objective_ratio_grad(c, Y, w)
            return val, grad
        try:
            res = minimize(fun, x0, method="L-BFGS-B", jac=True,
                           options={"maxiter": 200, "maxls": 40, "ftol": 1e-15, "gtol": 1e-12})
            c_opt = res.x
            val_opt, _ = _objective_ratio_grad(c_opt, Y, w)
            if val_opt < best_val and np.isfinite(val_opt):
                best_val = val_opt
                best_c = c_opt.copy()
        except Exception:
            pass

    # 3. Global Optimization: Differential Evolution
    # We search in [-1, 1]^n. The objective is scale-invariant, so normalization inside DE is not strictly needed for the value, 
    # but helps keep numbers stable. However, _objective_ratio_grad works on raw c.
    # We will use a wrapper that normalizes c for the DE objective to keep it in a reasonable range, 
    # but pass the raw c to the gradient? No, DE only evaluates the function.
    # Since the function is scale-invariant, we can just minimize it on the unit sphere.
    # But DE explores a box. We map the box to the sphere?
    # Simpler: Just run DE on the box [-1, 1]^n. The objective value is the same for any scale.
    # The optimizer will find a direction.
    
    def de_obj(x):
        # Normalize to avoid overflow/underflow in intermediate steps if x is very large/small
        norm = np.linalg.norm(x)
        if norm < 1e-12:
            return 1e10
        x_norm = x / norm
        val, _ = _objective_ratio_grad(x_norm, Y, w)
        return val

    bounds = [(-1.0, 1.0)] * n
    
    try:
        if time.time() < _DEADLINE - 100.0:
            # Use a larger population and more iterations
            res_de = differential_evolution(
                de_obj, bounds, seed=42, maxiter=3000, popsize=30, 
                tol=1e-10, mutation=(0.5, 1.0), recombination=0.9,
                polish=True
            )
            if res_de.success:
                c_de = res_de.x
                norm = np.linalg.norm(c_de)
                if norm > 1e-12:
                    c_de_norm = c_de / norm
                    val_de, _ = _objective_ratio_grad(c_de_norm, Y, w)
                    if val_de < best_val and np.isfinite(val_de):
                        best_val = val_de
                        best_c = c_de_norm.copy()
    except Exception as e:
        pass

    # 4. Final Polish
    if time.time() < _DEADLINE - 10.0:
        def fun_final(c):
            val, grad = _objective_ratio_grad(c, Y, w)
            return val, grad
        try:
            res_final = minimize(fun_final, best_c, method="L-BFGS-B", jac=True,
                                 options={"maxiter": 2000, "maxls": 50, "ftol": 1e-16, "gtol": 1e-14})
            c_final = res_final.x
            val_final, _ = _objective_ratio_grad(c_final, Y, w)
            if val_final < best_val and np.isfinite(val_final):
                best_val = val_final
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
