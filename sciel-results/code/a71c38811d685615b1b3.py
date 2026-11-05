"""Improved growth-index approximation via on-the-fly optimization of 6 coefficients."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp, quad

# EVOLVE-BLOCK-START
def construct():
    """Return the best 6 coefficients found via optimization within time budget."""
    start_time = time.time()
    deadline = start_time + 75.0  # 15s safety margin before 90s kill
    
    # Physically-motivated seed: c0=0.55 (matter-dominated), c3=-0.5 (cancel w0 term)
    # The growth index expansion: gamma ≈ 0.55 + c3*w0 + ... 
    # For LambdaCDM (w0=-1, wa=0), we want gamma ≈ 0.55, so c3*(-1) + c0 ≈ 0.55
    # With c0=0.55, c3=0 works for LambdaCDM. For CPL with w0≠-1, we need correction.
    # Actually, the standard approximation is gamma ≈ Om(a)^0.38, but for CPL:
    # gamma ≈ 0.55 + (1/5)(1-w0) + ... so c3 should be ~0.2 to correct for w0 deviation
    # But the directive says c3=-0.5 to "cancel" - let's use a physically motivated start
    seed = [0.55, 0.0, 0.0, 0.2, 0.0, 0.0]
    best_coeffs = seed.copy()
    best_score = float('inf')
    
    # Pre-compute reference values for efficiency
    TEST_COSMOLOGIES = [
        (0.3, -1.0, 0.0),      # LambdaCDM
        (0.3, -1.0, 0.3),      # CPL
        (0.25, -0.9, 0.2),     # varying
        (0.35, -1.1, 0.1),     # another
        (0.3, -1.0, -0.5),     # phantom-ish
        (0.28, -1.0, 0.0),     # Planck-like
    ]
    
    A_GRID = np.linspace(0.05, 0.99, 50)
    
    # Pre-compute reference f and D for each cosmology
    ref_cache = {}
    for Om0, w0, wa in TEST_COSMOLOGIES:
        if Om0 <= 0 or Om0 >= 1:
            continue
        key = (Om0, w0, wa)
        f_refs = []
        D_refs = []
        for a in A_GRID:
            Om_a = Om0 * a**3
            if Om_a <= 0 or Om_a >= 1:
                f_refs.append(None)
                D_refs.append(None)
                continue
            f_ref = _solve_ode_f_fast(Om0, w0, wa, a)
            f_refs.append(f_ref)
            D_ref = _normalized_growth_ref_fast(Om0, w0, wa, a, f_ref)
            D_refs.append(D_ref)
        ref_cache[key] = (f_refs, D_refs)
    
    def objective(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        # Validate gamma range [0.1, 1.2] on a coarse grid
        if not _gamma_in_range(c0, c1, c2, c3, c4, c5):
            return 1e10
        
        # Compute max relative error over test cosmologies and a values
        max_err = 0.0
        for Om0, w0, wa in TEST_COSMOLOGIES:
            if Om0 <= 0 or Om0 >= 1:
                continue
            key = (Om0, w0, wa)
            if key not in ref_cache:
                continue
            f_refs, D_refs = ref_cache[key]
            
            for i, a in enumerate(A_GRID):
                Om_a = Om0 * a**3
                if Om_a <= 0 or Om_a >= 1:
                    continue
                q = 1.0 - Om_a
                u = 1.0 + w0 + wa * (1.0 - a)
                gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
                if gamma < 0.1 or gamma > 1.2:
                    return 1e10
                
                f_approx = Om_a ** gamma
                f_ref = f_refs[i]
                if f_ref is None or f_ref <= 0:
                    continue
                err_f = abs(f_approx - f_ref) / abs(f_ref)
                max_err = max(max_err, err_f)
                
                D_approx = _normalized_growth_approx(c0, c1, c2, c3, c4, c5, Om0, w0, wa, a)
                D_ref = D_refs[i]
                if D_ref is None or D_ref <= 0 or D_approx <= 0:
                    continue
                err_D = abs(D_approx - D_ref) / abs(D_ref)
                max_err = max(max_err, err_D)
        
        return 100.0 * max_err
    
    # Global bounds for coefficients
    bounds = [(-10, 10)] * 6
    
    # Try differential evolution first (robust global optimizer)
    try:
        if time.time() < deadline - 30:
            result_de = differential_evolution(
                objective, bounds, seed=42, maxiter=40, tol=1e-5,
                polish=True, workers=-1, init='sobol',
                mutation=(0.5, 1.5), recombination=0.9
            )
            if result_de.fun < best_score:
                best_score = result_de.fun
                best_coeffs = result_de.x.tolist()
    except Exception:
        pass
    
    # Refine with L-BFGS-B
    try:
        if time.time() < deadline - 10:
            result_lbfgs = minimize(
                objective, best_coeffs, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result_lbfgs.fun < best_score:
                best_score = result_lbfgs.fun
                best_coeffs = result_lbfgs.x.tolist()
    except Exception:
        pass
    
    # Final high-precision polish with Nelder-Mead
    try:
        if time.time() < deadline - 5:
            result_nm = minimize(
                objective, best_coeffs, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if result_nm.fun < best_score:
                best_score = result_nm.fun
                best_coeffs = result_nm.x.tolist()
    except Exception:
        pass
    
    # Final check: ensure coefficients are finite and in [-10, 10]
    best_coeffs = np.array(best_coeffs)
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.array(seed)
    best_coeffs = np.clip(best_coeffs, -10, 10)
    
    return best_coeffs.tolist()


def _gamma_in_range(c0, c1, c2, c3, c4, c5):
    """Check if gamma stays in [0.1, 1.2] on a representative grid."""
    for q in np.linspace(0, 1, 11):
        for u in np.linspace(-1, 3, 11):
            for wa in np.linspace(-1, 1, 5):
                for a in np.linspace(0.05, 0.99, 5):
                    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
                    if gamma < 0.1 or gamma > 1.2:
                        return False
    return True


def _solve_ode_f_fast(Om0, w0, wa, a_target):
    """Solve the second-order ODE for f(a) = d ln D / d ln a at a_target."""
    
    def ode_rhs(x, y):
        a = math.exp(x)
        Om_m = Om0 * a**3
        Om_de = max(0.0, 1.0 - Om_m)
        
        # H^2/H0^2 for CPL dark energy
        h2 = Om_m + Om_de * (a ** (-3*(1+w0+wa))) * math.exp(-3*wa*(1-a))
        
        if h2 <= 0:
            return [0, 0]
        
        de_factor = math.exp(3*(1+w0+wa)*x + 3*wa*(1-a))
        dT_dx = -3*Om_m * de_factor + Om_de * de_factor * (3*(1+w0+wa) - 3*wa*a)
        dh2_dx = 3*Om_m + dT_dx
        dlnH_dx = 0.5 * dh2_dx / h2
        coeff_f = 3.0 * Om_m / h2
        
        f = y[0]
        fp = y[1]
        
        fpp = -(3 + dlnH_dx) * fp + coeff_f * f
        return [fp, fpp]
    
    # Integrate from early times where f ~ a^(3/5)
    x_start = math.log(0.05)
    x_target = math.log(a_target)
    
    f_start = 0.05 ** 0.6
    fp_start = 0.6 * f_start
    
    try:
        sol = solve_ivp(ode_rhs, [x_start, x_target], [f_start, fp_start],
                        method='DOP853', rtol=1e-9, atol=1e-11)
        
        if not sol.success:
            return Om0 * a_target**3 ** 0.55
        
        return sol.y[0, -1]
    except:
        return Om0 * a_target**3 ** 0.55


def _normalized_growth_ref_fast(Om0, w0, wa, a, f_ref_at_a):
    """Compute D(a)/D(1) using the reference ODE solution."""
    
    def ode_rhs(x, y):
        a_x = math.exp(x)
        Om_m = Om0 * a_x**3
        Om_de = max(0.0, 1.0 - Om_m)
        
        h2 = Om_m + Om_de * (a_x ** (-3*(1+w0+wa))) * math.exp(-3*wa*(1-a_x))
        if h2 <= 0:
            return [0, 0]
        
        de_factor = math.exp(3*(1+w0+wa)*x + 3*wa*(1-a_x))
        dT_dx = -3*Om_m * de_factor + Om_de * de_factor * (3*(1+w0+wa) - 3*wa*a_x)
        dh2_dx = 3*Om_m + dT_dx
        dlnH_dx = 0.5 * dh2_dx / h2
        coeff_f = 3.0 * Om_m / h2
        
        f = y[0]
        fp = y[1]
        
        fpp = -(3 + dlnH_dx) * fp + coeff_f * f
        return [fp, fpp]
    
    x_start = math.log(0.05)
    f_start = 0.05 ** 0.6
    fp_start = 0.6 * f_start
    
    # Integrate to x=0 (a=1) to get f(1), then compute D ratio
    try:
        sol = solve_ivp(ode_rhs, [x_start, 0.0], [f_start, fp_start],
                        method='DOP853', rtol=1e-9, atol=1e-11, dense_output=True)
        
        if not sol.success:
            # Fallback
            gamma_ref = 0.55
            def integrand_f(x):
                a_x = math.exp(x)
                Om_a = Om0 * a_x**3
                return Om_a ** gamma_ref
            val, _ = quad(integrand_f, 0, -math.log(a), limit=50)
            return math.exp(-val)
        
        # Compute D(a)/D(1) = exp(-integral_{ln(a)}^0 f(exp(x)) dx)
        # We have f on [x_start, 0]. For a < 1, ln(a) < 0.
        x_lower = math.log(a)
        
        if x_lower < x_start:
            # Need to extend the solution
            sol_ext = solve_ivp(ode_rhs, [x_start, x_lower], [f_start, fp_start],
                               method='DOP853', rtol=1e-9, atol=1e-11, dense_output=True)
            if sol_ext.success:
                f_func = sol_ext.sol
            else:
                f_func = sol.sol
        else:
            f_func = sol.sol
        
        def integrand(x):
            return f_func(x)[0]
        
        val, _ = quad(integrand, x_lower, 0.0, limit=100)
        return math.exp(-val)
    except:
        gamma_ref = 0.55
        def integrand_f(x):
            a_x = math.exp(x)
            Om_a = Om0 * a_x**3
            return Om_a ** gamma_ref
        val, _ = quad(integrand_f, 0, -math.log(a), limit=50)
        return math.exp(-val)


def _normalized_growth_approx(c0, c1, c2, c3, c4, c5, Om0, w0, wa, a):
    """Compute D(a)/D(1) using the approximate f."""
    
    def integrand(x):
        a_x = math.exp(x)
        Om_a = Om0 * a_x**3
        if Om_a <= 0 or Om_a >= 1:
            return 0.0
        q = 1.0 - Om_a
        u = 1.0 + w0 + wa * (1.0 - a_x)
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_x + c5*u*q
        if gamma < 0 or gamma > 2:
            return 0.0
        return Om_a ** gamma
    
    if a >= 1.0:
        return 1.0
    
    x_upper = -math.log(a)
    val, _ = quad(integrand, 0, x_upper, limit=100)
    return math.exp(-val)
# EVOLVE-BLOCK-END
