"""Refined Cohn-Goncalves Laguerre double-root construction for Fourier sign uncertainty.

Minimizes the C4 upper bound  C4 = I2 / I1^2,
    I1 = int_0^inf P(x) e^{-x} dx,   I2 = int_0^inf P(x)^2 e^{-x} dx,
    P(x) = prod_j (x - x_j)^2,   x_j > 0,  (m=12, n=1).

Design (kept in the same family as the parent: 12-DOF log-root coordinate optimization):
  * analytic gradient via partial fractions (robust, no polynomial-division blowup),
  * log-variable change removes the positivity constraint,
  * dense 1D scan over generalized-Laguerre alpha for a strong seed,
  * L-BFGS-B (unconstrained in log-space) + a time-budgeted Newton solve on the
    KKT system for a machine-precision constrained polish.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_genlaguerre, roots_laguerre, exp1, erfcx, gammaln

_FACTS = np.array([math.factorial(k) for k in range(60)], dtype=np.float64)


def _poly_moment(c):
    """int_0^inf (sum c_k x^k) e^{-x} dx = sum c_k k!  (ascending coeffs)."""
    n = len(c)
    if n == 0:
        return 0.0
    m = min(n, 60)
    return float(np.dot(c[:m], _FACTS[:m]))


def _build_p_squared_coeffs(roots):
    from numpy.polynomial.polynomial import polyfromroots, polymul
    B = polyfromroots(roots)
    P = polymul(B, B)
    P2 = polymul(P, P)
    return P, P2


def _c4_and_grad(roots):
    """Return (C4, dC4/dx_i) with an analytic partial-fraction gradient."""
    roots = np.asarray(roots, dtype=np.float64)
    n = len(roots)
    if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(~np.isfinite(roots)):
        return np.inf, np.zeros(n)

    P, P2 = _build_p_squared_coeffs(roots)
    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        return np.inf, np.zeros(n)

    # A = sum_{j!=i} 1/(x_i - x_j)
    d = roots[:, None] - roots[None, :]
    np.fill_diagonal(d, 1.0)
    A = np.sum(1.0 / d, axis=1)
    # B = sum_{j!=i} 1/(x_i - x_j)^2
    B = np.sum(1.0 / (d * d), axis=1)

    # int (x-x_i)^{-1} e^{-x} dx and int (x-x_i)^{-2} e^{-x} dx
    E1 = exp1(roots)                       # = int_{x_i}^inf t^{-1} e^{-t} dt
    inv_x = 1.0 / roots
    E2 = 0.5 * (math.log(2.0 * math.pi) - inv_x - E1 + erfcx(roots) * inv_x)

    dI1 = -2.0 * (I1 * A - E1)
    dI2 = -4.0 * (I2 * A - I1 * B + E2)

    I1_sq = I1 * I1
    I1_cu = I1_sq * I1
    grad = dI2 / I1_sq - 2.0 * I2 * dI1 / I1_cu
    c4 = I2 / I1_sq
    return float(c4), grad


def _objective(x):
    c, _ = _c4_and_grad(np.exp(x))
    return c


def _objective_grad(x):
    r = np.exp(x)
    c, g = _c4_and_grad(r)
    if not np.isfinite(c):
        return c, np.zeros_like(x)
    return c, g * r


def _alpha_c4(alpha):
    try:
        xg, _ = roots_genlaguerre(12, float(alpha))
        xg = np.sort(xg)
        if not (np.all(xg > 0) and np.all(np.isfinite(xg))):
            return np.inf, xg
        c, _ = _c4_and_grad(xg)
        return c, xg
    except Exception:
        return np.inf, None


def _lbfgs(x0, maxiter=400, ftol=1e-15, gtol=1e-13):
    try:
        res = minimize(_objective_grad, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': ftol,
                                'gtol': gtol, 'maxls': 50, 'maxfun': 5000})
        if np.isfinite(res.fun) and np.all(np.isfinite(res.x)):
            return res.x
    except Exception:
        pass
    return x0


def _newton_kkt(x, deadline):
    """Time-budgeted Newton solve of the KKT system for min C4(e^x) s.t. e^x >= 1e-10.

    Returns (x, converged_flag).  Uses the analytic gradient and a damped line
    search; terminates early if the deadline is approached.
    """
    lo = math.log(1e-10)
    x = np.asarray(x, dtype=np.float64).copy()
    it = 0
    for it in range(250):
        if time.time() > deadline:
            break
        r = np.exp(x)
        f, g = _c4_and_grad(r)
        if not np.isfinite(f):
            break
        g = g * r                       # gradient w.r.t. x

        # Hessian of C4(e^x) via finite differences of the analytic gradient
        h = np.empty((12, 12))
        for i in range(12):
            e = 1e-6 * max(1.0, abs(x[i]))
            xp = x.copy(); xp[i] += e
            xm = x.copy(); xm[i] -= e
            fp = _objective_grad(xp)[0]
            fm = _objective_grad(xm)[0]
            h[i, :] = (fp - fm) / (2.0 * e)

        # KKT with active set A = {i : x[i] < lo + 1e-6}
        A = x < (lo + 1e-6)
        nA = int(np.sum(A))
        if nA == 0:
            # free problem
            try:
                dx = np.linalg.solve(h, -g)
            except Exception:
                break
        else:
            idxA = np.where(A)[0]
            idxF = np.where(~A)[0]
            # build block system
            H_ff = h[np.ix_(idxF, idxF)]
            H_fa = h[np.ix_(idxF, idxA)]
            H_af = h[np.ix_(idxA, idxF)]
            H_aa = h[np.ix_(idxA, idxA)]
            gf = g[idxF]
            ga = g[idxA]
            m = len(idxF)
            if m == 0:
                try:
                    lam = np.linalg.solve(H_aa, -ga)
                except Exception:
                    break
                dx = np.zeros(12)
                dx[idxA] = lam
            else:
                # [H_ff  H_fa] [dx_f]   [-g_f]
                # [H_af  H_aa] [dx_a] = [-g_a]
                # [H_af  H_aa] [lam ]   [ 0   ]   (complementarity: dx_a = 0 at boundary)
                # Solve with dx_a=0  =>  H_ff dx_f = -g_f - H_fa lam ;  H_af dx_f + H_aa lam = -g_a
                # => [H_ff  -H_fa] [dx_f] = [-g_f]
                #    [H_af   H_aa] [lam ]   [-g_a]
                blk = np.block([[H_ff, -H_fa], [H_af, H_aa]])
                rhs = np.concatenate([-gf, -ga])
                try:
                    sol = np.linalg.solve(blk, rhs)
                except Exception:
                    try:
                        sol = np.linalg.lstsq(blk, rhs, rcond=None)[0]
                    except Exception:
                        break
                dx = np.zeros(12)
                dx[idxF] = sol[:m]
                dx[idxA] = sol[m:]

        # damped line search
        step = 1.0
        f0 = f
        for _ in range(30):
            xn = x + step * dx
            if np.any(xn < lo):
                step *= 0.5
                continue
            fn, _ = _c4_and_grad(np.exp(xn))
            if np.isfinite(fn) and fn <= f0 + 1e-4 * step * float(np.dot(g, dx)):
                x = xn
                break
            step *= 0.5
        else:
            break
        if abs(step * np.max(np.abs(dx))) < 1e-13:
            break
    return x, (it < 250)


def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 25.0

    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    # ---- Phase 1: dense 1D scan over generalized-Laguerre alpha ----
    best_c4 = np.inf
    best_x = None
    best_alpha = 1.0

    alpha_values = np.unique(np.concatenate([
        np.logspace(-3, 4, 200),
        np.linspace(0.05, 60.0, 200),
    ]))
    for a in alpha_values:
        if not time_ok():
            break
        c, xg = _alpha_c4(a)
        if np.isfinite(c) and c < best_c4:
            best_c4 = c
            best_x = np.log(xg)
            best_alpha = float(a)

    if best_x is None:
        xg, _ = roots_laguerre(12)
        best_x = np.log(np.sort(xg))
        best_c4 = _objective(best_x)

    # refine alpha locally
    for _ in range(3):
        if not time_ok():
            break
        a_lo = max(best_alpha * 0.5, 1e-4)
        a_hi = best_alpha * 2.0
        for a in np.linspace(a_lo, a_hi, 40):
            c, xg = _alpha_c4(a)
            if np.isfinite(c) and c < best_c4:
                best_c4 = c
                best_x = np.log(xg)
                best_alpha = float(a)

    # ---- Phase 2: L-BFGS-B from best alpha point ----
    if time_ok():
        x2 = _lbfgs(best_x.copy(), maxiter=600, ftol=1e-15, gtol=1e-14)
        c2 = _objective(x2)
        if np.isfinite(c2) and c2 < best_c4:
            best_c4, best_x = c2, x2

    # ---- Phase 3: multi-start L-BFGS-B from perturbations ----
    if time_ok():
        rng = np.random.default_rng(12345)
        inits = [best_x.copy()]
        for scale in [0.02, 0.05, 0.1, 0.2, 0.4]:
            for _ in range(2):
                inits.append(best_x + scale * rng.standard_normal(12))
        for init in inits:
            if not time_ok():
                break
            xr = _lbfgs(init, maxiter=400, ftol=1e-15, gtol=1e-13)
            cr = _objective(xr)
            if np.isfinite(cr) and cr < best_c4:
                best_c4, best_x = cr, xr

    # ---- Phase 4: high-precision Newton polish on the KKT system ----
    for _ in range(3):
        if not time_ok():
            break
        x3, _ = _newton_kkt(best_x, DEADLINE - SAFETY)
        c3 = _objective(x3)
        if np.isfinite(c3) and c3 < best_c4:
            best_c4, best_x = c3, x3
        else:
            break

    # ---- Final safety check ----
    roots = np.exp(best_x)
    if not (np.all(np.isfinite(roots)) and np.all(roots > 0)):
        xg, _ = roots_laguerre(12)
        roots = np.sort(xg)
    return [float(xi) for xi in roots]
# EVOLVE-BLOCK-END
