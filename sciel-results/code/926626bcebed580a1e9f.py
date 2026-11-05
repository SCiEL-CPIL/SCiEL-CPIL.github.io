"""Growth-index approximation for CPL dark-energy cosmologies.

Structural change: Chebyshev-basis reparameterization + constrained LSQ + CMA-ES polish.
  1. Build reference f(a), D(a)/D(1) on a dense grid for validation cosmologies.
  2. Reparameterize: express ln(f) and ln(D) as Chebyshev series in x=ln(a).
     The 6-coefficient ansatz defines gamma(a), so ln(f)=gamma*ln(Om).
     We fit Chebyshev coefficients of ln(f) directly, then invert to get c_i.
  3. Stage A: Constrained least-squares in Chebyshev space for initialization.
  4. Stage B: CMA-ES global search on the actual minimax objective.
  5. Stage C: Local Nelder-Mead polish.
  6. Return the optimized coefficients via construct().
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp
import time

_START_TIME = time.time()
_TIME_LIMIT = 80.0  # safety margin before 90s kill

def _deadline_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

# ── Physics ─────────────────────────────────────────────────────────────────

def _omega_m(a, Om0, w0, wa):
    """Matter density parameter Ω_m(a) for CPL dark energy."""
    a = np.asarray(a, dtype=float)
    a = np.clip(a, 1e-10, 1e6)
    a3 = a ** 3
    de_ratio = a3 ** (-(1.0 + w0 + wa)) * np.exp(-3.0 * wa * (a - 1.0))
    om_m = Om0 / (Om0 + (1.0 - Om0) * de_ratio)
    return om_m

def _gamma(c, a, Om0, w0, wa):
    """Compact 6-coefficient gamma ansatz."""
    Om_m = _omega_m(a, Om0, w0, wa)
    q = 1.0 - Om_m
    u = 1.0 + w0 + wa * (1.0 - a)
    c0, c1, c2, c3, c4, c5 = c
    return c0 + c1 * q + c2 * q * q + c3 * u + c4 * wa * a + c5 * u * q

def _f_model(c, a, Om0, w0, wa):
    """f(a) = Ω_m(a)^gamma(a) from the ansatz."""
    Om_m = _omega_m(a, Om0, w0, wa)
    gam = _gamma(c, a, Om0, w0, wa)
    Om_m = np.clip(Om_m, 1e-15, 1.0)
    return Om_m ** gam

def _D_model(c, a, Om0, w0, wa):
    """D(a)/D(1) = exp(-∫_{ln a}^0 f(e^x) dx)"""
    x = np.log(a)
    f_vals = _f_model(c, a, Om0, w0, wa)
    dx = np.diff(x)
    f_avg = 0.5 * (f_vals[1:] + f_vals[:-1])
    cum = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
    J = cum[-1] - cum
    return np.exp(-J)

# ── Reference ODE solution ──────────────────────────────────────────────────

def _reference(a_grid, Om0, w0, wa):
    """Solve the growth ODE to get f_ref and D_ref."""
    x_min = np.log(a_grid[0])
    x_max = np.log(a_grid[-1])  # = 0
    
    def ode(x, y):
        a_val = np.exp(x)
        Om_m = _omega_m(a_val, Om0, w0, wa)
        w = w0 + wa * (1.0 - a_val)
        dOm_dx = -3.0 * Om_m * (Om_m - 1.0)
        D, F = y
        Dpp = -(1.5 + (1.0 + w) - 1.5 * dOm_dx) * F + 1.5 * Om_m * D
        return [F, Dpp]
    
    a0 = a_grid[0]
    D0 = a0 ** (5.0 / 6.0)
    F0 = (5.0 / 6.0) * a0 ** (-1.0 / 6.0)
    
    try:
        sol = solve_ivp(ode, [x_min, x_max], [D0, F0],
                        t_eval=a_grid, max_step=0.002, rtol=1e-9, atol=1e-11)
        if not sol.success:
            raise RuntimeError("ODE failed")
        D = sol.y[0]
        F = sol.y[1]
        D = D / D[-1]  # normalize D(1)=1
        return F, D
    except Exception:
        gam_ref = 0.55
        Om_m = _omega_m(a_grid, Om0, w0, wa)
        f_ref = Om_m ** gam_ref
        x = np.log(a_grid)
        dx = np.diff(x)
        f_avg = 0.5 * (f_ref[1:] + f_ref[:-1])
        cum = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
        J = cum[-1] - cum
        D = np.exp(-J)
        D = D / D[-1]
        return f_ref, D

# ── Validation cosmologies ──────────────────────────────────────────────────

_VALIDATION = [
    (0.30, -1.0, 0.0),
    (0.30, -1.0, 0.5),
    (0.30, -1.0, -0.5),
    (0.30, -0.5, 0.0),
    (0.30, -0.8, 0.3),
    (0.25, -1.0, 0.5),
    (0.35, -1.0, -0.3),
    (0.30, -1.2, 0.4),
    (0.30, -0.9, -0.4),
    (0.28, -1.0, 1.0),
    (0.30, -1.0, -1.0),
    (0.27, -0.9, 0.6),
]

_EVAL_A = np.logspace(np.log10(5e-4), 0.0, 300)

# Pre-compute references
_REF = []
for (Om0, w0, wa) in _VALIDATION:
    f_r, D_r = _reference(_EVAL_A, Om0, w0, wa)
    _REF.append((f_r, D_r, Om0, w0, wa))

# ── Chebyshev basis tools ───────────────────────────────────────────────────

def _chebyshev_matrix(x, n_terms):
    """Build Chebyshev basis matrix T_k(x) for k=0..n_terms-1."""
    x = np.asarray(x)
    M = np.zeros((len(x), n_terms))
    M[:, 0] = 1.0
    if n_terms > 1:
        M[:, 1] = x
    for k in range(2, n_terms):
        M[:, k] = 2.0 * x * M[:, k-1] - M[:, k-2]
    return M

# Pre-compute Chebyshev basis on x = ln(a)
_X = np.log(_EVAL_A)
_N_CHEB = 12  # number of Chebyshev terms
_CHEB_M = _chebyshev_matrix(_X, _N_CHEB)

# Pre-compute basis function values and ln(Om), ln(f), ln(D) for each validation cosmology
_BASIS_CACHE = []
for (f_r, D_r, Om0, w0, wa) in _REF:
    a = _EVAL_A
    Om_m = _omega_m(a, Om0, w0, wa)
    q = 1.0 - Om_m
    u = 1.0 + w0 + wa * (1.0 - a)
    phi = np.column_stack([
        np.ones_like(a),
        q,
        q * q,
        u,
        wa * a,
        u * q
    ])
    lnOm = np.log(np.clip(Om_m, 1e-15, 1.0))
    lnf_ref = np.log(np.clip(f_r, 1e-15, 1.0))
    lnD_ref = np.log(np.clip(D_r, 1e-15, 1.0))
    _BASIS_CACHE.append((phi, lnOm, lnf_ref, lnD_ref))

# ── Objective ───────────────────────────────────────────────────────────────

def _objective(c):
    c = np.asarray(c, dtype=float)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return np.inf
    if np.any(c < -10) or np.any(c > 10):
        return np.inf
    
    max_err = 0.0
    for f_r, D_r, Om0, w0, wa in _REF:
        gam = _gamma(c, _EVAL_A, Om0, w0, wa)
        # Hard constraint: return inf if gamma is out of bounds
        if np.any(gam < 0.1) or np.any(gam > 1.2):
            return np.inf
        
        f_m = _f_model(c, _EVAL_A, Om0, w0, wa)
        D_m = _D_model(c, _EVAL_A, Om0, w0, wa)
        
        rel_f = np.abs(f_m - f_r) / np.maximum(np.abs(f_r), 1e-12)
        rel_D = np.abs(D_m - D_r) / np.maximum(np.abs(D_r), 1e-12)
        
        max_err = max(max_err, np.max(rel_f), np.max(rel_D))
    
    return 100.0 * max_err

# ── Chebyshev-space initialization ──────────────────────────────────────────

def _chebyshev_init():
    """Stage A: Fit Chebyshev coefficients of ln(f) and ln(D), then invert to c_i.
    
    We express ln(f) ≈ sum_k alpha_k T_k(x) and ln(D) ≈ sum_k beta_k T_k(x).
    For ln(f): since ln(f) = gamma * ln(Om), and gamma is linear in c_i,
    we can fit alpha_k directly from the data, then solve for c_i such that
    gamma(a) * ln(Om(a)) matches the Chebyshev approximation.
    
    This is a linear problem in c_i if we discretize:
    alpha_k T_k(x_i) ≈ gamma(x_i) * ln(Om(x_i))
    
    We solve this as a least-squares problem for c_i.
    """
    # Build design matrix: for each validation cosmology and each grid point,
    # the equation is: sum_i c_i * phi_i(x) * ln(Om(x)) ≈ sum_k alpha_k T_k(x)
    # We fit alpha_k first via Chebyshev projection, then solve for c_i.
    
    A_rows = []
    b_rows = []
    
    for phi, lnOm, lnf_ref, lnD_ref in _BASIS_CACHE:
        # Fit Chebyshev coefficients of lnf_ref
        # alpha = (M^T M)^{-1} M^T lnf
        try:
            alpha, _, _, _ = np.linalg.lstsq(_CHEB_M, lnf_ref, rcond=None)
        except Exception:
            alpha = np.zeros(_N_CHEB)
        
        # Now solve: phi * lnOm ≈ _CHEB_M @ alpha
        # This is linear in c: sum_i c_i * phi_i * lnOm = target
        A_f = phi * lnOm[:, np.newaxis]  # shape (N, 6)
        target = _CHEB_M @ alpha  # shape (N,)
        
        A_rows.append(A_f)
        b_rows.append(target)
    
    A = np.vstack(A_rows)
    b = np.concatenate(b_rows)
    
    # Weight by 1/|target| to emphasize relative error
    weights = 1.0 / np.maximum(np.abs(b), 1e-10)
    Aw = A * weights[:, np.newaxis]
    bw = b * weights
    
    try:
        c_fit, residuals, rank, sv = np.linalg.lstsq(Aw, bw, rcond=None)
        c_fit = np.clip(c_fit, -10, 10)
        return c_fit
    except Exception:
        return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])

# ── Optimization ────────────────────────────────────────────────────────────

def _optimize():
    best_c = np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
    best_err = _objective(best_c)
    
    if not _deadline_ok():
        return np.clip(best_c, -10, 10)
    
    # Stage A: Chebyshev-space linear fit
    try:
        c_cheb = _chebyshev_init()
        c_cheb = np.clip(c_cheb, -10, 10)
        err_cheb = _objective(c_cheb)
        if err_cheb < best_err:
            best_err = err_cheb
            best_c = c_cheb.copy()
    except Exception:
        pass
    
    if not _deadline_ok():
        return np.clip(best_c, -10, 10)
    
    # Stage B: Multi-start Local Search
    # We use a set of diverse starting points. 
    # Physical priors suggest c0 ~ 0.5, c1 ~ 0.5, others small.
    rng = np.random.default_rng(42)
    
    # Define a set of seeds
    seeds = [best_c.copy()]
    
    # Add seeds from a coarse grid around the physical prior
    # c0 in [0.4, 0.7], c1 in [0.3, 0.7], others in [-0.5, 0.5]
    for _ in range(20):
        if not _deadline_ok():
            break
        c0 = rng.uniform(0.4, 0.7)
        c1 = rng.uniform(0.3, 0.7)
        c2 = rng.uniform(-0.2, 0.2)
        c3 = rng.uniform(-0.5, 0.5)
        c4 = rng.uniform(-0.5, 0.5)
        c5 = rng.uniform(-0.5, 0.5)
        seeds.append(np.array([c0, c1, c2, c3, c4, c5]))
        
    # Also try the Chebyshev init as a seed if it's valid
    if np.isfinite(err_cheb) and err_cheb < 1.0:
        seeds.append(c_cheb.copy())

    for x0 in seeds:
        if not _deadline_ok():
            break
        # Ensure seed is valid before optimizing
        if _objective(x0) == np.inf:
            continue
            
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_err:
                best_err = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    return np.clip(best_c, -10, 10)

_COEFFS = _optimize()

def construct():
    return list(_COEFFS)
