"""Improved growth-index coefficient search with smooth surrogate and polish."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import solve_ivp
import time

_t0 = time.time()
_DEADLINE = 82.0  # safety margin before 90s kill

# Validation cosmologies (Om0, w0, wa)
_COSMOS = [
    (0.30, -1.0,  0.0),
    (0.30, -1.0,  0.3),
    (0.30, -1.0, -0.3),
    (0.27, -0.9,  0.2),
    (0.30, -1.1,  0.1),
    (0.30, -0.95, -0.1),
]

# Dense a-grid for reference computation
_A_GRID = np.concatenate([
    np.linspace(0.05, 1.0, 200),
    np.array([0.999, 0.9995, 1.0]),
])
_A_GRID = np.unique(_A_GRID)
_LN_A = np.log(_A_GRID)

# Precomputed references per cosmology: f_ref[i,j], D_ref[i,j]
_F_REF = None
_D_REF = None


def _Om_of_a(a, Om0, w0, wa):
    """Matter density parameter at scale factor a for CPL dark energy."""
    a = np.asarray(a, dtype=float)
    a_safe = np.maximum(a, 1e-12)
    # DE density ratio: rho_de(a)/rho_de(1) = a^(3(1+w0+wa)) * exp(3*wa*(1-a))
    de_factor = np.power(a_safe, 3.0 * (1.0 + w0 + wa)) * np.exp(3.0 * wa * (1.0 - a_safe))
    Om = Om0 / (Om0 / np.power(a_safe, 3.0) + (1.0 - Om0) * de_factor)
    return Om


def _compute_gamma(a, c, Om0, w0, wa):
    """Gamma from the 6-coefficient ansatz."""
    c0, c1, c2, c3, c4, c5 = c
    Om = _Om_of_a(a, Om0, w0, wa)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q


def _f_model(a, c, Om0, w0, wa):
    """f = Om(a)^gamma."""
    Om = _Om_of_a(a, Om0, w0, wa)
    gamma = _compute_gamma(a, c, Om0, w0, wa)
    return np.power(np.maximum(Om, 1e-300), gamma)


def _D_model(a, c, Om0, w0, wa):
    """Normalized growth factor via numerical integration of f(a')/a' from a to 1."""
    a_arr = np.asarray(a, dtype=float)
    a_safe = np.maximum(a_arr, 1e-12)
    D = np.ones_like(a_safe)
    for i, ai in enumerate(a_safe):
        if ai >= 1.0 - 1e-12:
            D[i] = 1.0
            continue
        # Integrate f(a')/a' da' from ai to 1
        npts = 300
        a_grid = np.linspace(ai, 1.0, npts)
        f_vals = _f_model(a_grid, c, Om0, w0, wa) / a_grid
        integral = np.trapz(f_vals, a_grid)
        D[i] = np.exp(-integral)
    return D


def _solve_ode_f(Om0, w0, wa):
    """Solve the GR ODE for f = d ln D / d ln a on _A_GRID."""
    def ode_rhs(ln_a, f):
        a_val = np.exp(ln_a)
        Om = _Om_of_a(a_val, Om0, w0, wa)
        return -(1.5 + 0.5 * Om) * f - 1.5 * Om

    sol = solve_ivp(ode_rhs, [0.0, _LN_A[0]], [0.0],
                    t_eval=_LN_A, method='RK45', rtol=1e-11, atol=1e-13)
    return sol.y[0]


def _compute_references():
    """Precompute f_ref and D_ref for all cosmologies on _A_GRID."""
    global _F_REF, _D_REF
    _F_REF = []
    _D_REF = []
    for Om0, w0, wa in _COSMOS:
        f_ref = _solve_ode_f(Om0, w0, wa)
        _F_REF.append(f_ref)
        # D_ref: integrate f/a from a to 1
        D = np.ones_like(_A_GRID)
        for i, ai in enumerate(_A_GRID):
            if ai >= 1.0 - 1e-12:
                D[i] = 1.0
                continue
            mask = _A_GRID >= ai
            a_sub = _A_GRID[mask]
            f_sub = f_ref[mask]
            integrand = f_sub / a_sub
            integral = np.trapz(integrand, a_sub)
            D[i] = np.exp(-integral)
        _D_REF.append(D)


def _gamma_constraint_penalty(c):
    """Penalty if gamma leaves [0.1, 1.2] at any checked location."""
    max_violation = 0.0
    for Om0, w0, wa in _COSMOS:
        gamma = _compute_gamma(_A_GRID, c, Om0, w0, wa)
        lo_viol = np.maximum(0.1 - gamma, 0.0)
        hi_viol = np.maximum(gamma - 1.2, 0.0)
        max_violation = max(max_violation, np.max(lo_viol), np.max(hi_viol))
    return max_violation


def _objective(c, beta=10.0):
    """
    Smooth surrogate for worst-case relative error.
    Uses soft-max (log-sum-exp) over all (cosmo, a) pairs for f and D.
    """
    if time.time() > _DEADLINE:
        return 1e6

    c = np.asarray(c, dtype=float)
    if np.any(np.abs(c) > 10.0):
        return 1e6

    # Gamma constraint penalty
    gp = _gamma_constraint_penalty(c)
    if gp > 1e-6:
        return 1e6 + gp

    # Collect log-errors for soft-max
    log_errs = []
    for i, (Om0, w0, wa) in enumerate(_COSMOS):
        f_m = _f_model(_A_GRID, c, Om0, w0, wa)
        D_m = _D_model(_A_GRID, c, Om0, w0, wa)
        f_r = _F_REF[i]
        D_r = _D_REF[i]

        # Relative errors
        f_err = np.abs(f_m - f_r) / np.maximum(np.abs(f_r), 1e-10)
        D_err = np.abs(D_m - D_r) / np.maximum(np.abs(D_r), 1e-10)

        # Clip tiny values to avoid log(0)
        f_err = np.maximum(f_err, 1e-15)
        D_err = np.maximum(D_err, 1e-15)

        log_errs.append(np.log(f_err))
        log_errs.append(np.log(D_err))

    all_log_errs = np.concatenate(log_errs)

    # Soft-max: (1/beta) * log(sum(exp(beta * x))) approximates max(x)
    m = np.max(all_log_errs)
    soft_max = m + np.log(np.sum(np.exp(beta * (all_log_errs - m)))) / beta

    return soft_max


def construct():
    """Find optimal 6 coefficients minimizing worst-case relative error."""
    global _F_REF, _D_REF

    if _F_REF is None:
        _compute_references()

    # Seed: constant gamma
    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_obj = _objective(best_c, beta=10.0)

    # Phase 1: Global search with differential evolution
    if time.time() < _DEADLINE - 10.0:
        bounds = [(-2.0, 2.0)] * 6
        try:
            de_result = differential_evolution(
                _objective, bounds, args=(10.0,),
                seed=42, maxiter=40, popsize=20, tol=1e-7,
                polish=True, init='sobol'
            )
            if de_result.fun < best_obj:
                best_c = de_result.x
                best_obj = de_result.fun
        except Exception:
            pass

    # Phase 2: Local polish with Nelder-Mead (high precision)
    if time.time() < _DEADLINE - 3.0:
        try:
            nm_result = minimize(
                _objective, best_c, args=(20.0,),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
                         'adaptive': True}
            )
            if nm_result.fun < best_obj:
                best_c = nm_result.x
                best_obj = nm_result.fun
        except Exception:
            pass

    # Phase 3: Final fine polish
    if time.time() < _DEADLINE - 1.0:
        try:
            fine_result = minimize(
                _objective, best_c, args=(50.0,),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14,
                         'adaptive': True}
            )
            if fine_result.fun < best_obj:
                best_c = fine_result.x
                best_obj = fine_result.fun
        except Exception:
            pass

    # Clamp to [-10, 10]
    best_c = np.clip(best_c, -10.0, 10.0)
    return list(best_c)
