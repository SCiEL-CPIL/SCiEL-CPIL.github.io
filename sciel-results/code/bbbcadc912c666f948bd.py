"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import random

def _get_reference_growth(w0, wa, a_vals):
    """
    Computes reference f(a) and D(a)/D(1) for a given CPL model.
    Solves the ODE: D'' + (1 + 3w) D' - 1.5 Om D = 0
    where ' denotes d/d ln a.
    Uses early-time condition D(a) ~ a for a << 1.
    """
    a_min = 0.1
    a_max = 1.0
    
    # Define Om(a) and w(a)
    def Om_func(a):
        # CPL: w(a) = w0 + wa * (1 - a)
        # Integral of (1+w) da / a^3 ...
        # E^2 = Om_m a^-3 + Om_r a^-4 + Om_k a^-2 + Om_de
        # We assume flat universe, radiation negligible for a > 0.1
        # Om_de(a) = Om_de0 * exp(3 * integral_1^a (1+w(a'))/a' da')
        # For CPL, integral is analytical.
        # Let's just use a simple numerical integration or assume flat LCDM-like structure?
        # The task says "smooth CPL dark-energy cosmologies".
        # Usually implies flat universe.
        # Om_m(a) = Om_m0 * a^-3 / E^2(a)
        # E^2(a) = Om_m0 a^-3 + (1 - Om_m0) * exp(3 * (w0 + 1 + wa/a)) ... wait.
        # Let's use the standard CPL density evolution.
        # rho_de(a) = rho_de0 * exp(3 * integral_1^a (1+w(a'))/a' da')
        # integral_1^a (w0+1+wa(1-a'))/a' da' = (w0+1) ln(a) + wa (1/a - 1)
        # So rho_de(a) = rho_de0 * a^(3(w0+1)) * exp(3*wa*(1/a - 1))
        # Let's compute E^2(a) = Om_m0 a^-3 + (1-Om_m0) * [a^(3(w0+1)) * exp(3*wa*(1/a - 1))]
        # We need Om_m0. The task doesn't specify Om_m0 for the validation models.
        # "disjoint public validation cosmologies".
        # I must assume a standard set or that the function is robust to Om_m0.
        # The ansatz depends on q = 1 - Om(a).
        # If Om_m0 varies, the fit should be robust.
        # I will assume Om_m0 = 0.3 for the reference generation in the search,
        # but the actual evaluation uses the specific validation models.
        # Wait, the `construct` function returns coefficients. The evaluator uses THEM.
        # My search must use a representative set of models.
        # I will use Om_m0 in {0.2, 0.3, 0.4} and w0 in {-1.5, -1.0, -0.5} and wa in {-0.5, 0.0, 0.5}.
        
        pass # Placeholder, logic below
    
    # Since I don't know the exact validation models, I will create a diverse set for fitting.
    # I need to define Om(a) correctly for the reference.
    # Assumption: Flat universe, Radiation ignored.
    # Om_m0 is a parameter of the cosmology.
    
    def compute_Om(a, Om_m0, w0, wa):
        # E^2(a) = Om_m0 a^-3 + (1 - Om_m0) * exp(3 * (w0 + 1) * ln(a) + 3 * wa * (1/a - 1))
        # term_de = (1 - Om_m0) * a**(3*(w0+1)) * np.exp(3*wa*(1/a - 1))
        # E2 = Om_m0 * a**-3 + term_de
        # Om = Om_m0 * a**-3 / E2
        term_de = (1 - Om_m0) * np.power(a, 3 * (w0 + 1)) * np.exp(3 * wa * (1 / a - 1))
        E2 = Om_m0 * np.power(a, -3) + term_de
        return (Om_m0 * np.power(a, -3)) / E2

    def ode_func(ln_a, y, Om_m0, w0, wa):
        a = np.exp(ln_a)
        D, Dp = y
        Om = compute_Om(a, Om_m0, w0, wa)
        w = w0 + wa * (1 - a)
        # D'' + (1 + 3w) D' - 1.5 Om D = 0
        # D'' = 1.5 Om D - (1 + 3w) D'
        return [Dp, 1.5 * Om * D - (1 + 3 * w) * Dp]

    # Solve from a_min to a_max
    # Initial conditions at a_min: D = a_min, D' = 1 (since D ~ a)
    # Note: D' is dD/d ln a. If D ~ a, D = C a, D' = C. So D'/D = 1.
    # Let D(a_min) = a_min, D'(a_min) = 1.
    
    ln_a_min = np.log(a_min)
    ln_a_max = np.log(a_max)
    
    sol = solve_ivp(ode_func, [ln_a_min, ln_a_max], [a_min, 1.0], 
                    args=(Om_m0, w0, wa), 
                    t_eval=np.log(a_vals), 
                    rtol=1e-8, atol=1e-10, method='RK45')
    
    if not sol.success:
        return None, None

    D_sol = sol.y[0]
    Dp_sol = sol.y[1]
    
    # Normalize D(1) = 1
    # Find index where a is closest to 1.0
    idx_1 = np.argmin(np.abs(a_vals - 1.0))
    D_norm = D_sol[idx_1]
    D_ref = D_sol / D_norm
    f_ref = Dp_sol / D_sol
    
    return f_ref, D_ref

def _evaluate_error(coeffs, models, a_vals):
    c0, c1, c2, c3, c4, c5 = coeffs
    
    max_err = 0.0
    
    for Om_m0, w0, wa in models:
        # Compute reference
        f_ref, D_ref = _get_reference_growth(w0, wa, a_vals)
        if f_ref is None:
            continue
            
        # Compute ansatz
        a = a_vals
        # Compute Om(a) for ansatz (same as reference)
        # Re-use compute_Om logic or inline it
        term_de = (1 - Om_m0) * np.power(a, 3 * (w0 + 1)) * np.exp(3 * wa * (1 / a - 1))
        E2 = Om_m0 * np.power(a, -3) + term_de
        Om_a = (Om_m0 * np.power(a, -3)) / E2
        
        q = 1 - Om_a
        u = 1 + w0 + wa * (1 - a)
        
        gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
        
        # Check gamma bounds [0.1, 1.2]
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return float('inf')
            
        f_ana = np.power(Om_a, gamma)
        
        # Compute D_ana
        # D(a)/D(1) = exp(- integral_a^1 f(a') d ln a')
        # Since we have discrete points, use cumulative trapezoidal integration
        # We need integral from a to 1.
        # Sort a_vals to be increasing for integration? 
        # a_vals is likely increasing.
        # integral_a^1 f d ln a = - integral_1^a f d ln a
        # Let's compute cumulative integral from a[0] to a[i].
        # Then integral_a^1 = TotalIntegral - IntegralToA.
        
        # Ensure a_vals is sorted
        sort_idx = np.argsort(a_vals)
        a_sorted = a_vals[sort_idx]
        f_sorted = f_ana[sort_idx]
        f_ref_sorted = f_ref[sort_idx]
        D_ref_sorted = D_ref[sort_idx]
        
        ln_a_sorted = np.log(a_sorted)
        
        # Integrate f d ln a
        # trapezoidal rule
        d_ln_a = np.diff(ln_a_sorted)
        avg_f = (f_sorted[:-1] + f_sorted[1:]) / 2
        integral_diffs = avg_f * d_ln_a
        integral_cum = np.concatenate(([0], np.cumsum(integral_diffs)))
        
        total_integral = integral_cum[-1]
        
        # integral from a_i to 1 (which is a_sorted[-1] if 1 is included)
        # Assuming a_vals includes 1.0
        # If not, we interpolate. Let's assume 1.0 is in a_vals.
        idx_1 = np.argmin(np.abs(a_sorted - 1.0))
        
        # integral_a^1 f d ln a = total_integral - integral_cum[i]
        integrals_to_1 = total_integral - integral_cum
        
        D_ana = np.exp(-integrals_to_1)
        # Normalize D_ana such that D_ana(1) = 1
        # D_ana[idx_1] should be exp(0) = 1.
        
        # Unsort
        f_ana_unsorted = np.empty_like(f_ana)
        f_ana_unsorted[sort_idx] = f_sorted
        D_ana_unsorted = np.empty_like(D_ana)
        D_ana_unsorted[sort_idx] = D_ana
        
        # Calculate errors
        # Relative error for f
        # Avoid division by zero
        mask = np.abs(f_ref) > 1e-6
        if np.any(mask):
            err_f = np.abs((f_ana_unsorted[mask] - f_ref[mask]) / f_ref[mask])
            max_err = max(max_err, np.max(err_f))
            
        # Relative error for D
        mask_D = np.abs(D_ref) > 1e-6
        if np.any(mask_D):
            err_D = np.abs((D_ana_unsorted[mask_D] - D_ref[mask_D]) / D_ref[mask_D])
            max_err = max(max_err, np.max(err_D))
            
    return max_err

def construct():
    """
    Returns six coefficients for the growth-index ansatz.
    Uses a numerical optimization to find the best fit for a representative set of cosmologies.
    """
    
    # Representative models: (Om_m0, w0, wa)
    models = [
        (0.3, -1.0, 0.0),   # LCDM
        (0.3, -1.0, 0.5),   # CPL
        (0.3, -0.5, 0.0),   # w0 different
        (0.3, -1.5, 0.0),   # Phantom
        (0.2, -1.0, 0.0),   # Low matter
        (0.4, -1.0, 0.0),   # High matter
        (0.3, -1.0, -0.5),  # wa negative
        (0.3, -0.5, 0.5),   # Combined
    ]
    
    a_vals = np.linspace(0.1, 1.0, 20)
    
    best_coeffs = [0.55, 0.15, -0.05, -0.5, 0.3, 0.1]
    best_err = float('inf')
    
    # Random sampling
    num_samples = 200
    for _ in range(num_samples):
        c = [random.uniform(-10, 10) for _ in range(6)]
        err = _evaluate_error(c, models, a_vals)
        if err < best_err:
            best_err = err
            best_coeffs = c
            
    # Local refinement
    # Take best few and perturb
    # For simplicity, just refine the single best
    for _ in range(500):
        # Perturb best_coeffs
        new_c = [c + random.gauss(0, 0.5) for c in best_coeffs]
        # Clamp to [-10, 10]
        new_c = [max(-10, min(10, c)) for c in new_c]
        err = _evaluate_error(new_c, models, a_vals)
        if err < best_err:
            best_err = err
            best_coeffs = new_c
        else:
            # Decay step size? Not implemented, just random restarts implicitly by many samples
            pass

    return best_coeffs

# EVOLVE-BLOCK-END
