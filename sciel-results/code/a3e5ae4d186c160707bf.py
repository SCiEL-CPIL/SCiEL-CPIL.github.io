import numpy as np
import scipy.integrate
import scipy.optimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 75.0

    # ---------- 1. Validation cosmology grid ----------
    # Dense log-spaced low-z (relative error is worst here), plus high-z.
    z_low = np.logspace(np.log10(0.005), np.log10(0.5), 25)
    z_mid = np.linspace(0.5, 3.0, 20)
    z_high = np.logspace(np.log10(3.0), np.log10(10.0), 15)
    z_vals = np.unique(np.concatenate([z_low, z_mid, z_high]))

    w0_vals = np.linspace(-1.5, 0.5, 8)
    w1_vals = np.linspace(-1.5, 1.5, 8)

    cases = [(w0, w1, z) for w0 in w0_vals for w1 in w1_vals for z in z_vals]
    n = len(cases)

    w0_arr = np.array([c[0] for c in cases])
    w1_arr = np.array([c[1] for c in cases])
    z_arr = np.array([c[2] for c in cases])

    # ---------- 2. Fast vectorized 1/E(z') ----------
    def invE(t, w0, w1):
        """Compute 1/E(t) for arrays t (redshift), w0, w1 (scalars or arrays)."""
        t = np.asarray(t, dtype=np.float64)
        lp = np.log1p(t)
        lt1 = 3.0 * lp
        lt2 = (3.0 + 3.0 * w0) * lp + 3.0 * w1 * (t / (1.0 + t))
        m = np.maximum(lt1, lt2)
        s = np.exp(lt1 - m) + np.exp(lt2 - m)
        log_E2 = m + np.log(s)
        return np.exp(-0.5 * log_E2)

    # ---------- 3. Precompute exact integrals via adaptive quad ----------
    exact = np.zeros(n)
    for i in range(n):
        w0, w1, z = cases[i]
        try:
            val, _ = scipy.integrate.quad(
                lambda u: invE(z * u, w0, w1),
                0.0, 1.0, limit=200, epsabs=1e-12, epsrel=1e-12
            )
            exact[i] = val
        except Exception:
            us = np.linspace(0, 1, 2000)
            exact[i] = np.trapz(invE(z * us, w0, w1), us)

    exact_safe = np.maximum(exact, 1e-30)

    # ---------- 4. Objective: worst-case relative error ----------
    def rel_errors(u, w):
        """u: (8,), w: (8,) -> array of n relative errors."""
        t = z_arr[:, None] * u[None, :]
        f = invE(t, w0_arr[:, None], w1_arr[:, None])
        qs = np.sum(f * w[None, :], axis=1)
        return np.abs(qs - exact_safe) / exact_safe

    def objective_linf(params):
        """params: (16,) -> 8 logit nodes, 8 log-weights. Returns scalar."""
        theta = params[:8]
        alpha = params[8:]
        u = 1.0 / (1.0 + np.exp(-np.clip(theta, -40, 40)))
        ea = np.exp(np.clip(alpha, -40, 40) - np.max(np.clip(alpha, -40, 40)))
        w = ea / ea.sum()
        errs = rel_errors(u, w)
        # Robust L∞ proxy: sum of top-5 errors
        top = np.sort(errs)[-5:]
        return np.sum(top)

    def objective_max(params):
        theta = params[:8]
        alpha = params[8:]
        u = 1.0 / (1.0 + np.exp(-np.clip(theta, -40, 40)))
        ea = np.exp(np.clip(alpha, -40, 40) - np.max(np.clip(alpha, -40, 40)))
        w = ea / ea.sum()
        return np.max(rel_errors(u, w))

    # ---------- 5. Chebyshev-inspired initialization ----------
    # Map Chebyshev nodes from [-1,1] to [0,1], slightly shifted to avoid exact 0 and 1.
    k = np.arange(8)
    # Chebyshev-Gauss-Lobatto nodes (include endpoints)
    gl_nodes = -np.cos(np.pi * k / 7.0)  # from -1 to 1
    u_cheb = 0.5 + 0.5 * gl_nodes
    # Clip to strictly interior
    u_cheb = np.clip(u_cheb, 1e-8, 1.0 - 1e-8)
    # Ensure strictly increasing
    for i in range(1, 8):
        if u_cheb[i] <= u_cheb[i-1]:
            u_cheb[i] = u_cheb[i-1] + 1e-6

    # Weights: start with trapezoidal-ish, then normalize
    w_cheb = np.ones(8)
    w_cheb[0] = 0.5
    w_cheb[-1] = 0.5
    w_cheb = w_cheb / w_cheb.sum()

    def to_params(u, w):
        u_safe = np.clip(u, 1e-10, 1 - 1e-10)
        theta = np.log(u_safe / (1.0 - u_safe))
        w_safe = np.clip(w, 1e-10, 1.0)
        alpha = np.log(w_safe)
        return np.concatenate([theta, alpha])

    p0 = to_params(u_cheb, w_cheb)

    # Also prepare GL initial
    gl_n, gl_w = np.polynomial.legendre.leggauss(8)
    u_gl = 0.5 + 0.5 * gl_n
    w_gl = 0.5 * gl_w
    p_gl = to_params(u_gl, w_gl)

    # ---------- 6. Multi-start Nelder-Mead ----------
    best_params = p0.copy()
    best_score = objective_max(p0)

    # Evaluate GL as baseline
    gl_score = objective_max(p_gl)
    if gl_score < best_score:
        best_params = p_gl.copy()
        best_score = gl_score

    starts = [p0, p_gl]
    # Add a few perturbed starts
    rng = np.random.RandomState(42)
    for _ in range(4):
        p_pert = p0 + rng.randn(16) * 0.3
        starts.append(p_pert)

    for p_init in starts:
        if time.time() > deadline - 8:
            break
        try:
            res = scipy.optimize.minimize(
                objective_linf, p_init, method='Nelder-Mead',
                options={'maxiter': 600, 'xatol': 1e-9, 'fatol': 1e-11,
                         'adaptive': True}
            )
            if res.fun < 1e10:
                s = objective_max(res.x)
                if s < best_score:
                    best_params = res.x.copy()
                    best_score = s
        except Exception:
            pass

    # ---------- 7. Final polish with tighter Nelder-Mead on max ----------
    if time.time() < deadline - 4:
        try:
            res2 = scipy.optimize.minimize(
                objective_max, best_params, method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-12,
                         'adaptive': True}
            )
            if res2.fun < best_score:
                best_params = res2.x.copy()
                best_score = res2.fun
        except Exception:
            pass

    # ---------- 8. Extract and validate ----------
    theta_f = best_params[:8]
    alpha_f = best_params[8:]
    u_f = 1.0 / (1.0 + np.exp(-np.clip(theta_f, -40, 40)))
    ea = np.exp(np.clip(alpha_f, -40, 40) - np.max(np.clip(alpha_f, -40, 40)))
    w_f = ea / ea.sum()

    # Sort by node
    idx = np.argsort(u_f)
    u_f = u_f[idx]
    w_f = w_f[idx]

    # Enforce strictly increasing with minimum separation
    for i in range(1, 8):
        if u_f[i] - u_f[i-1] < 1e-10:
            u_f[i] = u_f[i-1] + 1e-10
    u_f = np.clip(u_f, 1e-12, 1.0 - 1e-12)

    # Renormalize weights
    w_f = np.maximum(w_f, 1e-15)
    w_f = w_f / w_f.sum()

    # Final check: compare with GL, keep better
    err_f = np.max(rel_errors(u_f, w_f))
    err_gl = np.max(rel_errors(u_gl, w_gl))
    if err_gl < err_f:
        u_f = u_gl.copy()
        w_f = w_gl.copy()
        w_f = w_f / w_f.sum()

    return {"nodes": u_f, "weights": w_f}
