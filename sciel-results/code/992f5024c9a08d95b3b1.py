# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize

_K = 10
_NPHI = 130
_NTH = 70
_NPTS = _NTH * _NPHI
_NC = 2 * _K + 1


def _build_quadrature():
    x, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI
    W = np.outer(w, np.full(_NPHI, dphi))

    Y = np.zeros((_NTH, _NPHI, _NC))
    for idx, m in enumerate(range(-_K, _K + 1)):
        am = abs(m)
        denom = 1.0
        for n in range(_K - am + 1, _K + am + 1):
            denom *= n
        fact_ratio = 1.0 / denom
        Nm = np.sqrt((2 * _K + 1) / (4.0 * np.pi) * fact_ratio)
        P = lpmv(am, _K, x)
        if m == 0:
            Y[:, :, idx] = Nm * P[:, None]
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.sin(am * phi)[None, :]
    return Y, W


_Y, _W = _build_quadrature()
_Wflat = _W.ravel()
_Yflat = _Y.reshape(_NPTS, _NC)


def _objective_L4(c):
    f = _Yflat @ c
    return float(np.dot(_Wflat, f * f * f * f))


def _gradient_L4(c):
    f = _Yflat @ c
    return 4.0 * (_Yflat.T @ (_Wflat * f * f * f))


def _objective_unconstrained(c):
    norm_sq = float(np.dot(c, c))
    if norm_sq < 1e-30:
        return 1e30
    f = _Yflat @ c
    H = float(np.dot(_Wflat, f * f * f * f))
    return H / (norm_sq * norm_sq)


def _gradient_unconstrained(c):
    norm_sq = float(np.dot(c, c))
    if norm_sq < 1e-30:
        return np.zeros(_NC)
    f = _Yflat @ c
    H = float(np.dot(_Wflat, f * f * f * f))
    dH = 4.0 * (_Yflat.T @ (_Wflat * f * f * f))
    grad = (dH - H * c / norm_sq) / (norm_sq * norm_sq)
    return grad


def _conjugate_gradient_sphere(c0, max_iter=5000, ftol=1e-14):
    """
    Riemannian conjugate gradient on the unit sphere.
    Uses Polak-Ribiere beta and backtracking line search.
    """
    c = np.asarray(c0, dtype=np.float64).copy()
    c /= np.linalg.norm(c)

    f_curr = _objective_L4(c)
    g_curr = _gradient_L4(c)
    # Project to tangent space
    gt_curr = g_curr - np.dot(g_curr, c) * c

    d = -gt_curr.copy()
    best_c = c.copy()
    best_val = f_curr
    f_prev = f_curr
    gt_prev = gt_curr.copy()

    for k in range(max_iter):
        if np.linalg.norm(gt_curr) < ftol:
            break

        # Backtracking line search
        alpha = 1.0
        c_trial = c + alpha * d
        # Project back to sphere
        c_trial = c_trial / np.linalg.norm(c_trial)
        f_trial = _objective_L4(c_trial)

        # If not decreasing, backtrack
        backtracks = 0
        while f_trial > f_curr - 1e-16 * alpha * np.dot(d, gt_curr) and backtracks < 30:
            alpha *= 0.5
            c_trial = c + alpha * d
            c_trial = c_trial / np.linalg.norm(c_trial)
            f_trial = _objective_L4(c_trial)
            backtracks += 1

        if f_trial < best_val:
            best_val = f_trial
            best_c = c_trial.copy()

        # Update
        c = c_trial
        f_curr = f_trial
        g_curr = _gradient_L4(c)
        gt_curr = g_curr - np.dot(g_curr, c) * c

        # Polak-Ribiere beta
        beta = max(0.0, np.dot(gt_curr, gt_curr - gt_prev) / (np.dot(gt_prev, gt_prev) + 1e-30))
        d = -gt_curr + beta * d

        # Re-orthogonalize direction to tangent space
        d = d - np.dot(d, c) * c

        gt_prev = gt_curr.copy()
        f_prev = f_curr

    return best_c, best_val


def _hessian_polish(c0, max_iter=200):
    """
    Second-order polish using Gauss-Newton style on the sphere.
    Approximate Hessian via finite differences of gradient.
    """
    c = np.asarray(c0, dtype=np.float64).copy()
    c /= np.linalg.norm(c)

    f_curr = _objective_L4(c)
    g_curr = _gradient_L4(c)
    gt_curr = g_curr - np.dot(g_curr, c) * c

    best_c = c.copy()
    best_val = f_curr

    for it in range(max_iter):
        if np.linalg.norm(gt_curr) < 1e-13:
            break

        # Build approximate Hessian (tangent-tangent) via finite differences
        eps = 1e-5
        H_tan = np.zeros((_NC, _NC))
        for i in range(_NC):
            ei = np.zeros(_NC)
            ei[i] = 1.0
            ei = ei - np.dot(ei, c) * c
            ei /= np.linalg.norm(ei)

            c_plus = c + eps * ei
            c_plus /= np.linalg.norm(c_plus)
            g_plus = _gradient_L4(c_plus)
            gt_plus = g_plus - np.dot(g_plus, c_plus) * c_plus

            c_minus = c - eps * ei
            c_minus /= np.linalg.norm(c_minus)
            g_minus = _gradient_L4(c_minus)
            gt_minus = g_minus - np.dot(g_minus, c_minus) * c_minus

            H_tan[:, i] = (gt_plus - gt_minus) / (2.0 * eps)

        # Solve H_tan * step = -gt_curr
        try:
            step = np.linalg.solve(H_tan + 1e-8 * np.eye(_NC), -gt_curr)
        except np.linalg.LinAlgError:
            step = -gt_curr

        # Line search
        alpha = 1.0
        c_trial = c + alpha * step
        c_trial /= np.linalg.norm(c_trial)
        f_trial = _objective_L4(c_trial)

        backtracks = 0
        while f_trial > f_curr and backtracks < 20:
            alpha *= 0.5
            c_trial = c + alpha * step
            c_trial /= np.linalg.norm(c_trial)
            f_trial = _objective_L4(c_trial)
            backtracks += 1

        if f_trial < best_val:
            best_val = f_trial
            best_c = c_trial.copy()

        c = c_trial
        f_curr = f_trial
        g_curr = _gradient_L4(c)
        gt_curr = g_curr - np.dot(g_curr, c) * c

    return best_c, best_val


def _optimize_sphere(c0, deadline):
    c0 = np.asarray(c0, dtype=np.float64)
    norm = np.linalg.norm(c0)
    if norm < 1e-12:
        c0 = np.ones(_NC)
        c0 /= np.linalg.norm(c0)
    else:
        c0 = c0 / norm

    best_c = c0.copy()
    best_val = _objective_L4(c0)

    # Stage 1: Riemannian CG
    try:
        c_cg, val_cg = _conjugate_gradient_sphere(c0, max_iter=3000, ftol=1e-14)
        if val_cg < best_val:
            best_val = val_cg
            best_c = c_cg.copy()
    except Exception:
        pass

    # Stage 2: SLSQP polish
    if time.time() < deadline - 15.0:
        try:
            def fun(c):
                return _objective_L4(c)

            def jac(c):
                return _gradient_L4(c)

            def cons_fun(c):
                return np.sum(c * c) - 1.0

            def cons_jac(c):
                return 2.0 * c

            constraints = [{'type': 'eq', 'fun': cons_fun, 'jac': cons_jac}]
            res = minimize(fun, best_c, jac=jac, method='SLSQP', constraints=constraints,
                           options={'maxiter': 500, 'ftol': 1e-14, 'disp': False})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
                best_c /= np.linalg.norm(best_c)
        except Exception:
            pass

    # Stage 3: Unconstrained L-BFGS-B on F(c) = H(c)/||c||^4
    if time.time() < deadline - 10.0:
        try:
            res2 = minimize(_objective_unconstrained, best_c, jac=_gradient_unconstrained,
                            method='L-BFGS-B', options={'maxiter': 3000, 'ftol': 1e-15, 'disp': False})
            if res2.fun < best_val:
                nc = res2.x / np.linalg.norm(res2.x)
                val_check = _objective_L4(nc)
                if val_check < best_val:
                    best_val = val_check
                    best_c = nc
        except Exception:
            pass

    # Stage 4: Hessian polish
    if time.time() < deadline - 5.0:
        try:
            c_hp, val_hp = _hessian_polish(best_c, max_iter=100)
            if val_hp < best_val:
                best_val = val_hp
                best_c = c_hp.copy()
        except Exception:
            pass

    # Stage 5: Final CG + SLSQP
    if time.time() < deadline - 3.0:
        try:
            c_cg2, val_cg2 = _conjugate_gradient_sphere(best_c, max_iter=2000, ftol=1e-15)
            if val_cg2 < best_val:
                best_val = val_cg2
                best_c = c_cg2.copy()
        except Exception:
            pass

    return best_c, best_val


def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0

    # Generate diverse initial candidates
    rng = np.random.default_rng(2024)

    starts = []

    # All-ones (symmetric)
    c_all = np.ones(_NC)
    c_all /= np.linalg.norm(c_all)
    starts.append(c_all)

    # Only m=0
    c_m0 = np.zeros(_NC)
    c_m0[10] = 1.0
    starts.append(c_m0)

    # Only even m
    c_even = np.zeros(_NC)
    for i in range(_NC):
        m = i - _K
        if m % 2 == 0:
            c_even[i] = 1.0
    c_even /= np.linalg.norm(c_even)
    starts.append(c_even)

    # Only odd m
    c_odd = np.zeros(_NC)
    for i in range(_NC):
        m = i - _K
        if m % 2 != 0:
            c_odd[i] = 1.0
    c_odd /= np.linalg.norm(c_odd)
    starts.append(c_odd)

    # Only |m| >= 4 (high order)
    c_high = np.zeros(_NC)
    for i in range(_NC):
        m = i - _K
        if abs(m) >= 4:
            c_high[i] = 1.0
    c_high /= np.linalg.norm(c_high)
    starts.append(c_high)

    # Only |m| <= 3 (low order)
    c_low = np.zeros(_NC)
    for i in range(_NC):
        m = i - _K
        if abs(m) <= 3:
            c_low[i] = 1.0
    c_low /= np.linalg.norm(c_low)
    starts.append(c_low)

    # Alternating signs
    c_alt = np.zeros(_NC)
    for i in range(_NC):
        m = i - _K
        c_alt[i] = 1.0 if m % 4 == 0 else -1.0
    c_alt /= np.linalg.norm(c_alt)
    starts.append(c_alt)

    # Random starts with different seeds
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        r = np.random.default_rng(seed)
        c_rand = r.standard_normal(_NC)
        c_rand /= np.linalg.norm(c_rand)
        starts.append(c_rand)

    # Structured: Gaussian decay in |m|
    for sigma in [1.0, 2.0, 3.0, 5.0]:
        c_gauss = np.zeros(_NC)
        for i in range(_NC):
            m = i - _K
            c_gauss[i] = np.exp(-m * m / (2.0 * sigma * sigma))
        c_gauss /= np.linalg.norm(c_gauss)
        starts.append(c_gauss)

    best_c = starts[0].copy()
    best_val = _objective_L4(starts[0])

    for c0 in starts:
        if time.time() > deadline - 20.0:
            break
        try:
            c_opt, val_opt = _optimize_sphere(c0, deadline)
            if val_opt < best_val:
                best_val = val_opt
                best_c = c_opt.copy()
        except Exception:
            continue

    # Final polish
    if time.time() < deadline - 5.0:
        try:
            c_final, val_final = _conjugate_gradient_sphere(best_c, max_iter=5000, ftol=1e-15)
            if val_final < best_val:
                best_val = val_final
                best_c = c_final.copy()
        except Exception:
            pass

    # Safety check
    best_c = np.asarray(best_c, dtype=np.float64)
    if not np.all(np.isfinite(best_c)):
        best_c = starts[0]
    mx = float(np.max(np.abs(best_c)))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)

    return best_c

# EVOLVE-BLOCK-END
