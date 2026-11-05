"""Growth-index ansatz with reference-cosmology anchoring and direct ODE-based fitting."""
import numpy as np
from scipy.optimize import minimize
import time

_DEADLINE = time.time() + 78.0

def _deadline_ok():
    return time.time() < _DEADLINE

def _omega_m_of_a(a, Om0, w0, wa):
    """Compute matter density parameter Om(a) for CPL dark energy."""
    a = np.asarray(a, dtype=float)
    scalar = a.ndim == 0
    if scalar:
        a = a.reshape(1)
    
    Om = np.empty_like(a)
    for i in range(len(a)):
        ai = a[i]
        if ai <= 0:
            Om[i] = Om0
            continue
        if abs(ai - 1.0) < 1e-15:
            Om[i] = Om0
            continue
        # rho_de(a)/rho_de(1) = exp(-3 * integral_0^{ln a} [1+w0+wa*(1-e^x)] dx)
        # integral = (1+w0+wa)*ln(a) - wa*(a-1)
        ln_a = np.log(ai)
        integral = (1.0 + w0 + wa) * ln_a - wa * (ai - 1.0)
        rho_de_ratio = np.exp(-3.0 * integral)
        rho_m_ratio = ai ** (-3.0)
        Om[i] = Om0 * rho_m_ratio / (Om0 * rho_m_ratio + (1.0 - Om0) * rho_de_ratio)
    
    if scalar:
        return Om[0]
    return Om

def _gamma_from_coeffs(coeffs, a, Om0, w0, wa):
    """Compute gamma from the six-coefficient ansatz."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om = _omega_m_of_a(a, Om0, w0, wa)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    return gamma

def _f_from_gamma(gamma, Om):
    """Compute f = Om^gamma."""
    return np.power(np.maximum(Om, 1e-15), gamma)

def _growth_ode_reference(a_grid, Om0, w0, wa, n_sub=2000):
    """Compute reference f and D using the growth ODE.
    
    f' = f*(f + 3/2) - 3/2*Om(a)  where ' = d/d ln a
    D(a)/D(1) = exp(-integral_{ln a}^0 f(exp(x)) dx)
    """
    a_grid = np.asarray(a_grid, dtype=float)
    n = len(a_grid)
    
    # Compute Om on a fine grid for the ODE
    x_fine = np.linspace(np.log(a_grid[0]), 0.0, n_sub)
    a_fine = np.exp(x_fine)
    Om_fine = _omega_m_of_a(a_fine, Om0, w0, wa)
    
    # Integrate ODE backward from x=0 (a=1)
    # f'(x) = f*(f+1.5) - 1.5*Om(exp(x))
    # Initial condition at x=0: f(1) ≈ Om0^0.55
    f_fine = np.zeros(n_sub)
    f_fine[-1] = Om0 ** 0.55
    
    # RK4 integration backward
    dx = x_fine[1] - x_fine[0]  # positive step size
    for i in range(n_sub - 2, -1, -1):
        x0 = x_fine[i + 1]
        a0 = np.exp(x0)
        Om0_val = Om_fine[i + 1]
        f0 = f_fine[i + 1]
        
        def _ode_f(x, f):
            a_val = np.exp(x)
            Om_val = _omega_m_of_a(a_val, Om0, w0, wa)
            return f * (f + 1.5) - 1.5 * Om_val
        
        k1 = _ode_f(x0, f0)
        k2 = _ode_f(x0 - dx/2, f0 - dx/2 * k1)
        k3 = _ode_f(x0 - dx/2, f0 - dx/2 * k2)
        k4 = _ode_f(x0 - dx, f0 - dx * k3)
        f_fine[i] = f0 - dx * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    
    # Interpolate f onto the evaluation grid
    f_ref = np.interp(np.log(a_grid), x_fine, f_fine)
    
    # Compute D(a)/D(1) = exp(-integral_{ln a}^0 f(exp(x)) dx)
    # For a < 1: integral from ln(a) to 0 of f(e^x) dx
    D_ref = np.ones(n)
    # Cumulative integral from the right (from x=0 backward)
    # integral_i = integral from x_i to 0 of f dx
    # Use cumulative trapezoidal rule
    x_eval = np.log(a_grid)
    # We need integral from x_i to 0 for each i
    # Reverse: compute integral from 0 to x_i, then negate
    # Actually: D(a)/D(1) = exp(-∫_{ln a}^{0} f(e^x) dx)
    # Let's compute cumulative integral from x=0 downward
    
    # Interpolate f onto a fine grid aligned with x_eval
    x_fine2 = np.linspace(x_eval[0], 0.0, max(n * 10, 1000))
    f_fine2 = np.interp(x_fine2, x_fine, f_fine)
    
    # Cumulative integral from 0 to each x_fine2 point
    # ∫_0^{x} f dx = -∫_{x}^{0} f dx
    # We want ∫_{x}^{0} f dx for x = ln(a_i)
    # = -∫_0^{x} f dx
    
    # Compute ∫_0^{x} f dx using cumulative trapezoid
    from scipy.integrate import cumulative_trapezoid
    # x_fine2 goes from x_eval[0] (negative) to 0
    # We want integral from 0 to x for x in x_eval
    # Let's reverse: x_rev = -x_fine2 goes from -x_eval[0] to 0
    # Actually simpler: compute cumulative integral along x_fine2
    
    # x_fine2 is sorted from x_eval[0] to 0
    # cumtrapz gives ∫_{x_fine2[0]}^{x_fine2[i]} f dx
    # We want ∫_{x_i}^{0} f dx = ∫_{x_fine2[0]}^{0} f dx - ∫_{x_fine2[0]}^{x_i} f dx
    
    cum_int = cumulative_trapezoid(f_fine2, x_fine2, initial=0.0)
    total_int = cum_int[-1]  # ∫_{x_fine2[0]}^{0} f dx
    
    # For each evaluation point x_i, ∫_{x_i}^{0} f dx = total_int - ∫_{x_fine2[0]}^{x_i} f dx
    # ∫_{x_fine2[0]}^{x_i} f dx = interpolate cum_int at x_i
    int_at_xi = np.interp(x_eval, x_fine2, cum_int)
    integral_xi_to_0 = total_int - int_at_xi
    
    D_ref = np.exp(-integral_xi_to_0)
    
    return f_ref, D_ref

def _build_references(val_cosmos, a_grid):
    """Pre-compute reference solutions for all validation cosmologies."""
    refs = []
    for (Om0, w0, wa) in val_cosmos:
        if not _deadline_ok():
            break
        f_ref, D_ref = _growth_ode_reference(a_grid, Om0, w0, wa)
        refs.append((Om0, w0, wa, f_ref, D_ref))
    return refs

def _objective(coeffs, refs, a_grid):
    """Compute max relative error for f and D."""
    max_err = 0.0
    for (Om0, w0, wa, f_ref, D_ref) in refs:
        if not _deadline_ok():
            return 1e10
        Om = _omega_m_of_a(a_grid, Om0, w0, wa)
        gamma = _gamma_from_coeffs(coeffs, a_grid, Om0, w0, wa)
        
        # Check gamma bounds
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
        
        f_ours = _f_from_gamma(gamma, Om)
        
        # Compute D_ours
        # D(a)/D(1) = exp(-∫_{ln a}^0 f_ours(exp(x)) dx)
        x_eval = np.log(a_grid)
        from scipy.integrate import cumulative_trapezoid
        cum_int = cumulative_trapezoid(f_ours, x_eval, initial=0.0)
        total_int = cum_int[-1]
        integral_xi_to_0 = total_int - cum_int
        D_ours = np.exp(-integral_xi_to_0)
        
        # Relative errors (skip very small a where D is tiny)
        valid = a_grid > 0.15
        if not np.any(valid):
            continue
        
        rel_f = np.abs(f_ours[valid] - f_ref[valid]) / np.maximum(np.abs(f_ref[valid]), 1e-10)
        rel_D = np.abs(D_ours[valid] - D_ref[valid]) / np.maximum(np.abs(D_ref[valid]), 1e-10)
        
        max_err = max(max_err, np.max(rel_f), np.max(rel_D))
    
    return 100.0 * max_err

def _run_optimization(refs, a_grid, start_coeffs):
    """Run Nelder-Mead optimization."""
    best_coeff = np.array(start_coeffs, dtype=float)
    best_score = _objective(best_coeff, refs, a_grid)
    
    if not _deadline_ok():
        return best_coeff, best_score
    
    try:
        result = minimize(
            lambda c: _objective(c, refs, a_grid),
            best_coeff,
            method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True}
        )
        if result.fun < best_score and np.all(np.abs(result.x) <= 10):
            # Verify gamma bounds
            valid = True
            for (Om0, w0, wa, _, _) in refs:
                gamma = _gamma_from_coeffs(result.x, a_grid, Om0, w0, wa)
                if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                    valid = False
                    break
            if valid:
                best_coeff = result.x
                best_score = result.fun
    except Exception:
        pass
    
    return best_coeff, best_score

def construct():
    """Return six coefficients [c0..c5] for the growth-index ansatz."""
    # Validation cosmologies
    val_cosmos = [
        (0.3, -1.0, 0.0),
        (0.3, -1.0, 0.1),
        (0.3, -1.0, -0.1),
        (0.3, -0.9, 0.0),
        (0.3, -1.1, 0.0),
        (0.3, -1.0, 0.2),
        (0.28, -1.0, 0.0),
        (0.32, -1.0, 0.0),
        (0.3, -0.8, 0.3),
        (0.3, -1.2, -0.2),
    ]
    
    # Evaluation grid
    a_grid = np.logspace(np.log10(0.15), 0.0, 150)
    
    # Pre-compute references
    refs = _build_references(val_cosmos, a_grid)
    
    if not refs:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Initial guess: constant gamma
    start = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    best_coeff, best_score = _run_optimization(refs, a_grid, start)
    
    # Try additional starting points if time permits
    if _deadline_ok():
        extra_starts = [
            [0.50, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.55, 0.0, 0.05, 0.0, 0.0, 0.0],
            [0.55, 0.0, 0.0, 0.05, 0.0, 0.0],
            [0.55, 0.0, 0.0, 0.0, 0.05, 0.0],
            [0.55, 0.0, 0.0, 0.0, 0.0, 0.05],
            [0.52, 0.05, 0.03, 0.03, 0.03, 0.03],
        ]
        for sp in extra_starts:
            if not _deadline_ok():
                break
            c, s = _run_optimization(refs, a_grid, sp)
            if s < best_score:
                best_coeff, best_score = c, s
    
    best_coeff = np.clip(best_coeff, -10.0, 10.0)
    return best_coeff.tolist()
