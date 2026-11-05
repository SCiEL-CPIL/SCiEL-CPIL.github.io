"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

_start_time = time.time()
_TIME_LIMIT = 55.0  # safety margin before 60s

def _hubble(a, Om0, w0, wa):
    """H(a)/H0 for flat CPL dark energy."""
    OL0 = 1.0 - Om0
    wa_eff = w0 + wa
    # Ω_Λ(a) = Ω_Λ,0 * a^(-3(1+w0+wa))
    # Actually for CPL: w(a) = w0 + wa*(1-a), so ∫(1+3w)da from 1 to a
    # Ω_DE(a) = Ω_DE,0 * a^(-3) * exp(-3*wa*(1-a))... let me use the standard form
    # For CPL: Ω_DE(a) = Ω_DE,0 * a^(-3(1+w0)) * exp(-3*wa*(1-a))
    # More precisely: Ω_DE(a) = Ω_DE,0 * a^(-3) * exp(3*wa*(1-a)) * a^(-3*w0)
    # = Ω_DE,0 * a^(-3-3*w0) * exp(3*wa*(1-a))
    if a <= 0:
        return np.inf
    Om_m = Om0 / a**3
    # Ω_DE(a) = Ω_DE,0 * a^(-3(1+w0+wa*(1-a)))... 
    # Standard CPL: Ω_DE(a) = Ω_DE,0 * a^(-3(1+w0)) * exp(-3*wa*(1-a))
    # Wait, let me be more careful. The continuity equation gives:
    # Ω_DE(a) = Ω_DE,0 * a^(-3(1+w0)) * exp(-3*wa*(1-a))
    # Actually the standard formula is:
    # E^2(a) = Om0/a^3 + (1-Om0)*a^(-3(1+w0+wa)) ... no
    # For CPL w = w0 + wa(1-a):
    # ρ_DE ∝ a^(-3(1+w0)) exp(-3wa(1-a))
    # So Ω_DE(a) = (1-Om0) * a^(-3(1+w0)) * exp(-3*wa*(1-a)) / [Om0/a^3 + (1-Om0)*a^(-3(1+w0))*exp(-3*wa*(1-a))]
    # But for H(a)/H0, we just need E^2(a) = Om0/a^3 + (1-Om0)*a^(-3(1+w0))*exp(-3*wa*(1-a))
    # Wait, I need to double-check. The standard CPL parameterization gives:
    # E^2(a) = Om_m0/a^3 + Om_DE0 * a^(-3(1+w0+wa)) ... 
    # Actually, the correct form for CPL is:
    # E^2(a) = Om0/a^3 + (1-Om0) * a^(-3(1+w0)) * exp(-3*wa*(1-a))
    # Let me use this.
    Om_DE = (1.0 - Om0) * a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a))
    E2 = Om_m + Om_DE
    return np.sqrt(E2)

def _solve_growth(Om0, w0, wa, a_min=0.01, a_max=1.0, n_points=500):
    """Solve the growth ODE and return f(a) and D(a)/D(1) on a grid."""
    a_grid = np.linspace(a_min, a_max, n_points)
    
    def ode_funcs(a, y):
        D, Dp = y
        H = _hubble(a, Om0, w0, wa)
        # H'/H = d ln H / da
        # H^2 = Om0/a^3 + (1-Om0)*a^(-3(1+w0))*exp(-3*wa*(1-a))
        # d(H^2)/da = -3*Om0/a^4 + (1-Om0)*[-3(1+w0)*a^(-4-3*w0)*exp(-3*wa*(1-a)) + 3*wa*a^(-3-3*w0)*exp(-3*wa*(1-a))]
        # H'/H = (1/2) * d(H^2)/da / H^2
        Om_m = Om0 / a**3
        Om_DE = (1.0 - Om0) * a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a))
        E2 = Om_m + Om_DE
        dE2_da = -3.0*Om0/a**4 + (1.0-Om0)*a**(-4.0-3.0*w0)*np.exp(-3.0*wa*(1.0-a))*(-3.0*(1.0+w0) + 3.0*wa)
        Hp_over_H = 0.5 * dE2_da / E2
        Dpp = -[3.0/a + Hp_over_H] * Dp + 3.0*Om_m/(2.0*a) * D
        return [Dp, Dpp]
    
    # Initial conditions: at a=1, D(1)=1, f(1)≈0.55 (for standard cosmology)
    # Actually, let's integrate backward from a=1
    # D(1) = 1, D'(1) = f(1) ≈ 0.55 for Om0~0.3
    f1 = 0.55  # approximate
    y0 = [1.0, f1]
    
    sol = solve_ivp(ode_funcs, [1.0, a_min], y0, t_eval=a_grid[::-1], rtol=1e-10, atol=1e-12, max_step=0.01)
    if not sol.success:
        return None, None
    
    a_vals = sol.t
    D_vals = sol.y[0]
    Dp_vals = sol.y[1]
    
    # f(a) = a * D'(a) / D(a)
    f_vals = a_vals * Dp_vals / D_vals
    # D(a)/D(1) = D_vals / D_vals[0]  (since D(1)=1)
    D_norm = D_vals / D_vals[0]
    
    # Reverse to get increasing a
    a_vals = a_vals[::-1]
    f_vals = f_vals[::-1]
    D_norm = D_norm[::-1]
    
    return a_vals, f_vals, D_norm

def _compute_gamma(a, Om0, w0, wa, coeffs):
    """Compute gamma from the ansatz."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om_a = Om0 / a**3
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa*(1.0 - a)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    return gamma

def _objective(coeffs, a_grid, f_ref, D_ref, Om0, w0, wa):
    """Compute max relative error of f and D."""
    c0, c1, c2, c3, c4, c5 = coeffs
    
    # Check gamma bounds
    Om_a = Om0 / a_grid**3
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa*(1.0 - a_grid)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_grid + c5*u*q
    
    if np.any(gamma < 0.1) or np.any(gamma > 1.2):
        return 1e6
    
    # Compute f_model = Om_a^gamma
    f_model = Om_a**gamma
    
    # Relative error in f
    rel_f = np.abs(f_model - f_ref) / np.abs(f_ref)
    
    # Compute D_model/D(1) = exp(-∫_a^1 f(exp(x)) dx)
    # x = ln(a), so dx = da/a
    # ∫_a^1 f(a')/a' da'
    # Use trapezoidal rule
    x = np.log(a_grid)
    # We need ∫_x^0 f(exp(t)) dt
    # f(exp(t)) = f_model at a=exp(t)
    # Let's compute the cumulative integral
    integrand = f_model / a_grid  # f(a)/a
    # ∫_a^1 f(a')/a' da' = ∫_x^0 f(exp(t)) dt
    # Use cumulative trapezoidal from the right
    # integral from a_i to a_{i+1} ≈ (integrand[i] + integrand[i+1])/2 * (a_{i+1} - a_i)
    # But we want ∫_a^1, so we need the integral from a_i to 1
    # Let's compute cumulative from the end
    da = np.diff(a_grid)
    trap = 0.5 * (integrand[:-1] + integrand[1:]) * da
    # cumulative from right: cumsum from the end
    cum_int = np.zeros_like(a_grid)
    for i in range(len(a_grid)-2, -1, -1):
        cum_int[i] = cum_int[i+1] + trap[i]
    # cum_int[i] = ∫_{a_i}^{a_max} f(a')/a' da' ≈ ∫_{a_i}^1 f(a')/a' da'
    D_model = np.exp(-cum_int)
    
    # D_ref is already normalized to D(1)=1
    rel_D = np.abs(D_model - D_ref) / np.abs(D_ref)
    
    max_err = max(np.max(rel_f), np.max(rel_D))
    return max_err

def construct():
    """Find 6 coefficients minimizing max relative error."""
    global _start_time, _TIME_LIMIT
    
    # Validation cosmologies (typical range)
    cosmologies = [
        (0.30, 0.0, 0.0),    # ΛCDM
        (0.30, -0.5, 0.0),   # w0=-0.5
        (0.30, -0.5, 0.5),   # CPL
        (0.30, -1.0, 0.0),   # phantom
        (0.25, -0.5, 0.5),   # different Om
        (0.35, -0.5, 0.0),   # different Om
        (0.30, -0.5, -0.5),  # negative wa
        (0.30, -1.5, 0.5),   # more phantom
    ]
    
    a_grid = np.linspace(0.05, 1.0, 200)
    
    # Solve reference ODEs
    ref_data = []
    for Om0, w0, wa in cosmologies:
        if time.time() - _start_time > _TIME_LIMIT:
            break
        try:
            a_vals, f_vals, D_vals = _solve_growth(Om0, w0, wa, a_min=0.05, a_max=1.0, n_points=200)
            if a_vals is not None:
                # Interpolate to our grid
                f_interp = np.interp(a_grid, a_vals, f_vals)
                D_interp = np.interp(a_grid, a_vals, D_vals)
                # Filter out bad values
                mask = np.isfinite(f_interp) & np.isfinite(D_interp) & (f_interp > 0) & (D_interp > 0)
                if np.sum(mask) > 50:
                    ref_data.append((a_grid[mask], f_interp[mask], D_interp[mask], Om0, w0, wa))
        except Exception:
            continue
    
    if not ref_data:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Optimize
    best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_err = np.inf
    
    # Try multiple starting points
    starts = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.1, 0.0, -0.5, 0.0, 0.0],
        [0.55, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.0, -0.3, 0.0, 0.0],
        [0.55, 0.1, 0.0, -0.5, 0.1, 0.0],
        [0.55, -0.1, 0.0, 0.3, 0.0, 0.0],
    ]
    
    for start in starts:
        if time.time() - _start_time > _TIME_LIMIT - 5:
            break
        
        def obj(coeffs):
            global _start_time, _TIME_LIMIT
            if time.time() - _start_time > _TIME_LIMIT - 2:
                return 1e6
            max_err = 0.0
            for a_g, f_r, D_r, Om0, w0, wa in ref_data:
                err = _objective(coeffs, a_g, f_r, D_r, Om0, w0, wa)
                max_err = max(max_err, err)
            return max_err
        
        try:
            res = minimize(obj, start, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < best_err:
                best_err = res.fun
                best_coeffs = res.x.tolist()
        except Exception:
            continue
    
    # Clip to [-10, 10]
    best_coeffs = [max(-10.0, min(10.0, c)) for c in best_coeffs]
    
    return best_coeffs
# EVOLVE-BLOCK-END
