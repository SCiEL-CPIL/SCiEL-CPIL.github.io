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
    deadline = start_time + 82.0  # safety margin before 90s kill
    
    # Known good constant-gamma seed
    seed = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_coeffs = seed.copy()
    best_score = float('inf')
    
    # Test cosmologies
    TEST_COSMOLOGIES = [
        (0.3, -1.0, 0.0),      # LambdaCDM
        (0.3, -1.0, 0.3),      # CPL
        (0.25, -0.9, 0.2),     # varying
        (0.35, -1.1, 0.1),     # another
        (0.3, -1.0, -0.5),     # phantom-ish
        (0.28, -1.0, 0.0),     # Planck-like
        (0.32, -1.2, 0.15),    # additional
        (0.27, -0.8, -0.2),    # additional
    ]
    
    # Precompute reference data for speed
    # For each cosmology, compute f_ref and D_ref at a grid of a values
    a_grid = np.linspace(0.05, 0.99, 40)
    ref_data = {}
    
    for Om0, w0, wa in TEST_COSMOLOGIES:
        if Om0 <= 0 or Om0 >= 1:
            continue
        f_refs = {}
        D_refs = {}
        for a in a_grid:
            Om_a = Om0 * a**3
            if Om_a <= 0 or Om_a >= 1:
                continue
            f_ref = _solve_ode_f_fast(Om0, w0, wa, a)
            D_ref = _normalized_growth_ref_fast(Om0, w0, wa, a)
            f_refs[a] = f_ref
            D_refs[a] = D_ref
        ref_data[(Om0, w0, wa)] = (f_refs, D_refs)
    
    def objective(coeffs):
        nonlocal best_score, best_coeffs
        c0, c1, c2, c3, c4, c5 = coeffs
        
        # Validate gamma range quickly
        if not _gamma_in_range_fast(c0, c1, c2, c3, c4, c5):
            return 1e10
        
        # Compute max relative error over test cosmologies and a values
        max_err = 0.0
        for Om0, w0, wa in TEST_COSMOLOGIES:
            if Om0 <= 0 or Om0 >= 1:
                continue
            if (Om0, w0, wa) not in ref_data:
                continue
            f_refs, D_refs = ref_data[(Om0, w0, wa)]
            
            for a in a_grid:
                Om_a = Om0 * a**3
                if Om_a <= 0 or Om_a >= 1:
                    continue
                
                q = 1.0 - Om_a
                u = 1.0 + w0 + wa * (1.0 - a)
                gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
                if gamma < 0.1 or gamma > 1.2:
                    return 1e10
                
                f_approx = Om_a ** gamma
                
                f_ref = f_refs.get(a)
                if f_ref is None or f_ref <= 0:
                    continue
                err_f = abs(f_approx - f_ref) / abs(f_ref)
                max_err = max(max_err, err_f)
                
                D_approx = _normalized_growth_approx_fast(c0, c1, c2, c3, c4, c5, Om0, w0, wa, a)
                D_ref = D_refs.get(a)
                if D_ref is None or D_ref <= 0 or D_approx <= 0:
                    continue
                err_D = abs(D_approx - D_ref) / abs(D_ref)
                max_err = max(max_err, err_D)
        
        score = 100.0 * max_err
        if score < best_score:
            best_score = score
            best_coeffs = coeffs.copy()
        
        return score
    
    # Global bounds for coefficients
    bounds = [(-10, 10)] * 6
    
    # Try differential evolution first (robust global optimizer)
    try:
        if time.time() < deadline - 25:
            result_de = differential_evolution(
                objective, bounds, seed=42, maxiter=40, tol=1e-5,
                polish=True, workers=-1, init='sobol',
                maxfevals=5000
            )
            if result_de.fun < best_score:
                best_score = result_de.fun
                best_coeffs = result_de.x.tolist()
    except Exception:
        pass
    
    # Refine with L-BFGS-B
    try:
        if time.time() < deadline - 15:
            result_lbfgs = minimize(
                objective, best_coeffs, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result_lbfgs.fun < best_score:
                best_score = result_lbfgs.fun
                best_coeffs = result_lbfgs.x.tolist()
    except Exception:
        pass
    
    # Try a second refinement from a slightly perturbed starting point
    try:
        if time.time() < deadline - 10:
            perturbed = [c + 0.1 * (np.random.rand() - 0.5) for c in best_coeffs]
            result_lbfgs2 = minimize(
                objective, perturbed, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result_lbfgs2.fun < best_score:
                best_score = result_lbfgs2.fun
                best_coeffs = result_lbfgs2.x.tolist()
    except Exception:
        pass
    
    # Final check: ensure coefficients are finite and in [-10, 10]
    best_coeffs = np.array(best_coeffs)
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.array(seed)
    best_coeffs = np.clip(best_coeffs, -10, 10)
    
    return best_coeffs.tolist()


def _gamma_in_range_fast(c0, c1, c2, c3, c4, c5):
    """Check if gamma stays in [0.1, 1.2] on a representative grid."""
    # Check on a coarser grid for speed
    for q in np.linspace(0, 1, 7):
        for u in np.linspace(-1, 3, 7):
            for wa in np.linspace(-1, 1, 3):
                for a in np.linspace(0.05, 0.99, 3):
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
        
        # H^2/H0^2
        de_factor = math.exp(3*(1+w0+wa)*x + 3*wa*(1-a))
        h2 = Om_m + Om_de * de_factor
        
        if h2 <= 0:
            return [0, 0]
        
        dT_dx = -3*Om_m * de_factor + Om_de * de_factor * (3*(1+w0+wa) - 3*wa*a)
        dh2_dx = 3*Om_m + dT_dx
        dlnH_dx = 0.5 * dh2_dx / h2
        coeff_f = 3.0 * Om_m / h2
        
        f = y[0]
        fp = y[1]
        fpp = -(3 + dlnH_dx) * fp + coeff_f * f
        return [fp, fpp]
    
    x_target = math.log(a_target)
    x_start = math.log(0.02)
    f_start = 0.02 ** 0.6
    fp_start = 0.6 * f_start
    
    try:
        sol = solve_ivp(ode_rhs, [x_start, x_target], [f_start, fp_start],
                        method='DOP853', rtol=1e-9, atol=1e-11)
        if sol.success:
            return sol.y[0, -1]
    except Exception:
        pass
    
    # Fallback to constant gamma
    return (Om0 * a_target**3) ** 0.55


def _normalized_growth_approx_fast(c0, c1, c2, c3, c4, c5, Om0, w0, wa, a):
    """Compute D(a)/D(1) using the approximate f."""
    
    def integrand(x):
        a_x = math.exp(x)
        Om_a = Om0 * a_x**3
        if Om_a <= 0 or Om_a >= 1:
            return 0.0
        q = 1.0 - Om_a
        u = 1.0 + w0 + wa * (1.0 - a_x)
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_x + c5*u*q
        if gamma < 0.1 or gamma > 1.2:
            gamma = 0.55  # safety fallback
        return Om_a ** gamma
    
    if a >= 1.0:
        return 1.0
    
    x_upper = -math.log(a)
    try:
        val, _ = quad(integrand, 0, x_upper, limit=100, epsabs=1e-10, epsrel=1e-10)
    except Exception:
        # Simple numerical integration fallback
        n = 100
        xs = np.linspace(0, x_upper, n)
        fs = np.array([integrand(x) for x in xs])
        val = np.trapz(fs, xs)
    
    return math.exp(-val)


def _normalized_growth_ref_fast(Om0, w0, wa, a):
    """Compute D(a)/D(1) using the reference ODE solution."""
    
    def ode_rhs(x, y):
        a_x = math.exp(x)
        Om_m = Om0 * a_x**3
        Om_de = max(0.0, 1.0 - Om_m)
        
        de_factor = math.exp(3*(1+w0+wa)*x + 3*wa*(1-a_x))
        h2 = Om_m + Om_de * de_factor
        if h2 <= 0:
            return [0, 0]
        
        dT_dx = -3*Om_m * de_factor + Om_de * de_factor * (3*(1+w0+wa) - 3*wa*a_x)
        dh2_dx = 3*Om_m + dT_dx
        dlnH_dx = 0.5 * dh2_dx / h2
        coeff_f = 3.0 * Om_m / h2
        
        f = y[0]
        fp = y[1]
        fpp = -(3 + dlnH_dx) * fp + coeff_f * f
        return [fp, fpp]
    
    x_start = math.log(0.02)
    f_start = 0.02 ** 0.6
    fp_start = 0.6 * f_start
    
    # Integrate from x_start to 0 (a=1)
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
        
        # Now compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
        # We have dense solution on [x_start, 0]
        # For a < 1, ln(a) < 0, so we integrate from ln(a) to 0
        
        x_lower = math.log(a)
        
        # Use the dense solution to evaluate f at points in [x_lower, 0]
        n = 50
        xs = np.linspace(x_lower, 0.0, n)
        # Evaluate the ODE solution at these points
        f_vals = np.array([sol.sol(x)[0] for x in xs])
        
        val = np.trapz(f_vals, xs)
        return math.exp(-val)
        
    except Exception:
        # Fallback to constant gamma
        gamma_ref = 0.55
        def integrand_f(x):
            a_x = math.exp(x)
            Om_a = Om0 * a_x**3
            return Om_a ** gamma_ref
        try:
            val, _ = quad(integrand_f, math.log(a), 0.0, limit=50)
            return math.exp(-val)
        except Exception:
            return 0.5
# EVOLVE-BLOCK-END
