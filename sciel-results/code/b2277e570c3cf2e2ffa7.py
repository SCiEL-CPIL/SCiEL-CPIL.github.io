"""
Cosmology: compact dark-energy growth approximation.
Implements a modified growth-index ansatz with sqrt(q) term and robust optimization.
"""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize

_START = time.time()
_DEADLINE = 82.0  # Safety margin before 90s kill

def _elapsed():
    return time.time() - _START

def _check_deadline():
    if _elapsed() > _DEADLINE:
        raise TimeoutError()

def _om_matter_vec(a, om0):
    """Matter density parameter at scale factor a (vectorized)."""
    return om0 / (a ** 3)

def _gamma_of_a_vec(a, om0, w0, wa, c):
    """
    Compute gamma from the six-coefficient ansatz (vectorized).
    Modified ansatz: gamma = c0 + c1*q + c2*sqrt(q) + c3*u + c4*wa*a + c5*u*q
    where q = 1 - Om(a), u = 1 + w0 + wa*(1-a)
    """
    c0, c1, c2, c3, c4, c5 = c
    om_a = _om_matter_vec(a, om0)
    # q = 1 - Om(a). In valid cosmologies, Om(a) <= 1, so q >= 0.
    # Clip to 0 to avoid numerical noise causing negative q
    q = np.clip(1.0 - om_a, 0.0, None)
    u = 1.0 + w0 + wa * (1.0 - a)
    
    # Use sqrt(q) instead of q**2 for the second term
    sqrt_q = np.sqrt(q)
    
    gamma = c0 + c1 * q + c2 * sqrt_q + c3 * u + c4 * wa * a + c5 * u * q
    return gamma

def _f_of_a_vec(a, om0, w0, wa, c):
    """Compute f = Om(a)^gamma (vectorized)."""
    om_a = _om_matter_vec(a, om0)
    gamma = _gamma_of_a_vec(a, om0, w0, wa, c)
    # Clip gamma for numerical stability in exponentiation, 
    # but the constraint check will use the raw gamma later.
    gamma_safe = np.clip(gamma, 0.1, 1.2)
    return om_a ** gamma_safe

def _solve_growth_ode(om0, w0, wa, a_min=0.01):
    """Solve the GR growth ODE to get reference f and D(a)/D(1)."""
    def ode(x, y):
        a = np.exp(x)
        E2 = om0 / (a**3) + (1.0 - om0) * np.exp(3.0 * (1.0 + w0 + wa) * (1.0 - a))
        om_m = om0 / (a**3)
        om_de = (1.0 - om0) * np.exp(3.0 * (1.0 + w0 + wa) * (1.0 - a))
        w = w0 + wa * (1.0 - a)
        E = np.sqrt(E2)
        
        # Handle case where om_m is very small to avoid division by zero in A
        # If om_m is tiny, the term is negligible or we are in pure DE where f~0
        if om_m < 1e-10:
            # In pure DE, growth is frozen or decays. 
            # The ODE coefficient A becomes problematic.
            # Standard approximation: if om_m is negligible, D is constant or decays slowly.
            # For numerical stability in the ODE solver, we can regularize.
            om_m_safe = 1e-10
        else:
            om_m_safe = om_m

        A = 3.0 / E * (1.0 + 0.5 * om_de * (1.0 + 3.0 * w) / om_m_safe) + 2.0
        B = -3.0 * om_m / (2.0 * E2)
        return [y[1], -A * y[1] + B * y[0]]

    x_min = np.log(a_min)
    x_max = np.log(1.0)
    a_init = a_min
    D_init = a_init ** (6.0 / 5.0)
    Dp_init = (6.0 / 5.0) * a_init ** (1.0 / 5.0)
    
    sol = solve_ivp(ode, [x_min, x_max], [D_init, Dp_init], 
                    method='DOP853', rtol=1e-10, atol=1e-12,
                    dense_output=True, max_step=0.01)
    
    if not sol.success:
        raise RuntimeError("ODE solve failed")
    
    return sol

def _get_reference(om0, w0, wa, a_grid):
    """Get reference f and D(a)/D(1) on a_grid for a given cosmology."""
    sol = _solve_growth_ode(om0, w0, wa)
    x_grid = np.log(a_grid)
    y = sol.sol(x_grid)
    D_raw = y[0]
    Dp_raw = y[1]
    D_ref = D_raw / D_raw[-1]
    f_ref = Dp_raw / D_raw
    
    D_ref = np.maximum(D_ref, 1e-10)
    f_ref = np.maximum(f_ref, 1e-10)
    
    return f_ref, D_ref

def _build_validation_set():
    """Build a diverse set of validation cosmologies and a-grid."""
    om0_vals = [0.2, 0.25, 0.27, 0.3, 0.35, 0.4, 0.45]
    w0_vals = [-1.0, -0.95, -0.9, -0.85, -0.8, -1.05, -1.1]
    wa_vals = [0.0, 0.2, 0.4, 0.6, -0.2, -0.4, 0.8, -0.6]
    
    cosmologies = []
    for om0 in om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                if 0.1 < om0 < 0.5:
                    cosmologies.append((om0, w0, wa))
    
    # Stratified sampling to ensure coverage
    np.random.seed(42)
    if len(cosmologies) > 60:
        strata = {}
        for i, (om0, w0, wa) in enumerate(cosmologies):
            key = (om0, w0)
            if key not in strata:
                strata[key] = []
            strata[key].append(i)
        
        selected = set()
        for key, indices in strata.items():
            n_take = min(4, len(indices))
            sample_idx = np.random.choice(indices, n_take, replace=False)
            selected.update(sample_idx.tolist())
        
        if len(selected) > 60:
            selected = set(np.random.choice(list(selected), 60, replace=False))
        
        cosmologies = [cosmologies[i] for i in sorted(selected)]
    
    if (0.27, -1.0, 0.0) not in cosmologies:
        cosmologies.append((0.27, -1.0, 0.0))
    
    # Finer a-grid
    a_grid = np.logspace(np.log10(0.01), 0.0, 200)
    a_grid = np.unique(a_grid)
    
    return cosmologies, a_grid

def _compute_D_vec(a, om0, w0, wa, c, x_grid, f_model_grid):
    """Compute D(a)/D(1) using pre-computed f_model on x_grid via cumulative integration."""
    n = len(x_grid)
    dx = np.diff(x_grid)
    areas = 0.5 * (f_model_grid[:-1] + f_model_grid[1:]) * dx
    cum_int = np.zeros(n)
    cum_int[:-1] = np.cumsum(areas[::-1])[::-1]
    return np.exp(-cum_int)

def _evaluate_score_vec(c, cosmologies, a_grid, f_refs, D_refs, x_grid):
    """Exact minimax score evaluation: 100 * max relative error."""
    max_err = 0.0
    n_a = len(a_grid)
    
    for i, (om0, w0, wa) in enumerate(cosmologies):
        f_ref = f_refs[i]
        D_ref = D_refs[i]
        
        f_model = _f_of_a_vec(a_grid, om0, w0, wa, c)
        err_f = np.abs(f_model - f_ref) / f_ref
        max_err = max(max_err, np.max(err_f))
        
        a_x = np.exp(x_grid)
        f_on_x = _f_of_a_vec(a_x, om0, w0, wa, c)
        D_model = _compute_D_vec(a_grid, om0, w0, wa, c, x_grid, f_on_x)
        
        err_D = np.abs(D_model - D_ref) / D_ref
        max_err = max(max_err, np.max(err_D))
    
    return 100.0 * max_err

def _check_gamma_constraint(c, cosmologies, a_grid):
    """Check if gamma is within [0.1, 1.2] for all cosmologies and a values."""
    for om0, w0, wa in cosmologies:
        g = _gamma_of_a_vec(a_grid, om0, w0, wa, c)
        if np.any(g < 0.1) or np.any(g > 1.2):
            return False
    return True

def _objective_smooth(c, cosmologies, a_grid, f_refs, D_refs, x_grid, beta=50.0):
    """Smooth surrogate using log-sum-exp for the max, plus constraint penalty."""
    n_cosmo = len(cosmologies)
    n_a = len(a_grid)
    total_errs = n_cosmo * 2 * n_a
    errs = np.empty(total_errs)
    
    idx = 0
    gamma_violation = 0.0
    
    for i, (om0, w0, wa) in enumerate(cosmologies):
        f_ref = f_refs[i]
        D_ref = D_refs[i]
        
        f_model = _f_of_a_vec(a_grid, om0, w0, wa, c)
        err_f = np.abs(f_model - f_ref) / f_ref
        errs[idx:idx+n_a] = err_f
        idx += n_a
        
        a_x = np.exp(x_grid)
        f_on_x = _f_of_a_vec(a_x, om0, w0, wa, c)
        D_model = _compute_D_vec(a_grid, om0, w0, wa, c, x_grid, f_on_x)
        err_D = np.abs(D_model - D_ref) / D_ref
        errs[idx:idx+n_a] = err_D
        idx += n_a
        
        # Check gamma constraint for penalty
        g = _gamma_of_a_vec(a_grid, om0, w0, wa, c)
        viol = np.maximum(0.1 - g, 0.0) + np.maximum(g - 1.2, 0.0)
        gamma_violation += np.sum(viol)
    
    if idx == 0:
        return 1e6
    
    errs = errs[:idx]
    m = np.max(errs)
    if m < 1e-15:
        base_score = 0.0
    else:
        lse = m + np.log(np.sum(np.exp(errs - m))) / beta
        base_score = 100.0 * lse
    
    # Add heavy penalty for constraint violation
    penalty = 1000.0 * gamma_violation
    return base_score + penalty

def construct():
    """Construct the six coefficients using differential evolution + local polish."""
    _check_deadline()
    
    cosmologies, a_grid = _build_validation_set()
    n_cosmo = len(cosmologies)
    
    x_grid = np.log(a_grid)
    
    f_refs = []
    D_refs = []
    for i, (om0, w0, wa) in enumerate(cosmologies):
        _check_deadline()
        f_ref, D_ref = _get_reference(om0, w0, wa, a_grid)
        f_refs.append(f_ref)
        D_refs.append(D_ref)
    
    _check_deadline()
    
    bounds = [(-10, 10)] * 6
    # Initial guess: constant gamma ~ 0.55, others 0
    c_best = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Verify initial guess constraint
    if not _check_gamma_constraint(c_best, cosmologies, a_grid):
        # Adjust if necessary, though 0.55 is usually safe
        c_best = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    score_init = _evaluate_score_vec(c_best, cosmologies, a_grid, f_refs, D_refs, x_grid)
    
    # Phase 1: Differential Evolution with smooth objective
    if _elapsed() < _DEADLINE - 30:
        try:
            de_result = differential_evolution(
                _objective_smooth,
                bounds,
                args=(cosmologies, a_grid, f_refs, D_refs, x_grid, 50.0),
                maxiter=300,
                popsize=40,
                tol=1e-8,
                seed=42,
                polish=False,
                updating='immediate'
            )
            if de_result.success and _elapsed() < _DEADLINE - 15:
                c_de = de_result.x
                # Check constraint before accepting
                if _check_gamma_constraint(c_de, cosmologies, a_grid):
                    score_de = _evaluate_score_vec(c_de, cosmologies, a_grid, f_refs, D_refs, x_grid)
                    if score_de < score_init:
                        c_best = c_de
                        score_init = score_de
        except Exception:
            pass
    
    _check_deadline()
    
    # Phase 2: Local refinement with L-BFGS-B on smooth objective
    if _elapsed() < _DEADLINE - 10:
        try:
            res = minimize(
                _objective_smooth,
                c_best,
                args=(cosmologies, a_grid, f_refs, D_refs, x_grid, 50.0),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.success:
                c_polished = res.x
                if _check_gamma_constraint(c_polished, cosmologies, a_grid):
                    score_polished = _evaluate_score_vec(c_polished, cosmologies, a_grid, f_refs, D_refs, x_grid)
                    if score_polished < score_init:
                        c_best = c_polished
                        score_init = score_polished
        except Exception:
            pass
    
    _check_deadline()
    
    # Phase 3: Final exact minimax polish
    if _elapsed() < _DEADLINE - 5:
        try:
            # Use Nelder-Mead for the non-smooth exact objective
            res2 = minimize(
                _evaluate_score_vec,
                c_best,
                args=(cosmologies, a_grid, f_refs, D_refs, x_grid),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.success:
                c_final = res2.x
                if _check_gamma_constraint(c_final, cosmologies, a_grid):
                    score_final = _evaluate_score_vec(c_final, cosmologies, a_grid, f_refs, D_refs, x_grid)
                    if score_final < score_init:
                        c_best = c_final
                        score_init = score_final
        except Exception:
            pass
    
    _check_deadline()
    
    c_best = np.clip(c_best, -10, 10)
    if not np.all(np.isfinite(c_best)):
        c_best = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Final verification
    if not _check_gamma_constraint(c_best, cosmologies, a_grid):
        # Fallback to a safe constant gamma
        c_best = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    return [float(x) for x in c_best]
