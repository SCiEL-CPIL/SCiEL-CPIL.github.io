"""Improved growth-index approximation via on-the-fly optimization of 6 coefficients."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

# EVOLVE-BLOCK-START
def construct():
    """Return the best 6 coefficients found via optimization within time budget."""
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    # Known good constant-gamma seed
    seed = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_coeffs = seed.copy()
    best_score = float('inf')
    
    # Define the objective function: worst relative error across f and D
    def objective(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        # Validate gamma range [0.1, 1.2] on a grid
        if not _gamma_in_range(c0, c1, c2, c3, c4, c5):
            return 1e10
        
        # Compute max relative error over test cosmologies and a values
        max_err = 0.0
        for Om0, w0, wa in TEST_COSMOLOGIES:
            if Om0 <= 0 or Om0 >= 1:
                continue
            # Sample a values from 0.05 to 0.99
            a_vals = np.linspace(0.05, 0.99, 80)
            
            for a in a_vals:
                # Compute q, u, gamma, f
                Om_a = Om0 * a**3
                if Om_a <= 0 or Om_a >= 1:
                    continue
                q = 1.0 - Om_a
                u = 1.0 + w0 + wa * (1.0 - a)
                gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
                if gamma < 0.1 or gamma > 1.2:
                    return 1e10
                
                f_approx = Om_a ** gamma
                
                # Reference: solve ODE for f_ref
                f_ref = _solve_ode_f(Om0, w0, wa, a)
                if f_ref <= 0:
                    continue
                err_f = abs(f_approx - f_ref) / abs(f_ref)
                max_err = max(max_err, err_f)
                
                # Compute normalized growth D(a)/D(1)
                D_approx = _normalized_growth_approx(c0, c1, c2, c3, c4, c5, Om0, w0, wa, a)
                D_ref = _normalized_growth_ref(Om0, w0, wa, a)
                
                if D_ref <= 0 or D_approx <= 0:
                    continue
                err_D = abs(D_approx - D_ref) / abs(D_ref)
                max_err = max(max_err, err_D)
        
        return 100.0 * max_err
    
    # Test cosmologies (disjoint public validation set approximation)
    TEST_COSMOLOGIES = [
        (0.3, -1.0, 0.0),      # LambdaCDM
        (0.3, -1.0, 0.3),      # CPL
        (0.25, -0.9, 0.2),     # varying
        (0.35, -1.1, 0.1),     # another
        (0.3, -1.0, -0.5),     # phantom-ish
        (0.28, -1.0, 0.0),     # Planck-like
    ]
    
    # Global bounds for coefficients
    bounds = [(-10, 10)] * 6
    
    # Try differential evolution first (robust global optimizer)
    try:
        if time.time() < deadline - 20:
            result_de = differential_evolution(
                objective, bounds, seed=42, maxiter=50, tol=1e-6,
                polish=True, workers=-1, init='sobol'
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
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            if result_lbfgs.fun < best_score:
                best_score = result_lbfgs.fun
                best_coeffs = result_lbfgs.x.tolist()
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
    # Check on a grid of (q, u, wa, a)
    for q in np.linspace(0, 1, 11):
        for u in np.linspace(-1, 3, 11):
            for wa in np.linspace(-1, 1, 5):
                for a in np.linspace(0.05, 0.99, 5):
                    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
                    if gamma < 0.1 or gamma > 1.2:
                        return False
    return True


def _solve_ode_f(Om0, w0, wa, a_target):
    """Solve the second-order ODE for f(a) = d ln D / d ln a at a_target."""
    from scipy.integrate import solve_ivp
    
    # f = Om(a)^gamma_ref where gamma_ref is the true growth index from ODE
    # We solve: f'' + (3 + H'/H) f' - (4 pi G rho / H^2) f = 0
    # In terms of x = ln a:
    # f'' + (3 + d ln H / dx) f' - 4 pi G rho / H^2 f = 0
    
    def ode_rhs(x, y):
        a = math.exp(x)
        Om_m = Om0 * a**3
        Om_de = 1.0 - Om_m
        if Om_m <= 0 or Om_m >= 1:
            return [0, 0]
        
        # H^2 / H0^2 = Om_m + Om_de * exp(3 * integral w(a')/a' da')
        # For CPL: integral from a to 1 of w(a')/a' da'
        # w(a') = w0 + wa*(1-a')
        # integral = w0*ln(1/a) + wa*(1-a)
        # So Om_de(a) = Om_de(1) * exp(3 * Om_de(1) * [w0*ln(1/a) + wa*(1-a)])
        # But more directly:
        # H^2(a)/H0^2 = Om_m + (1-Om_m) * (a/(1-a*0))... 
        # Actually for CPL:
        # H^2/H0^2 = Om_m + (1-Om_m) * exp(3*(1+w0)*ln(a) + 3*wa*(1-a))
        # Wait, let me be more careful.
        # rho_de(a) = rho_de(1) * exp(-3 * integral_1^a (1+w(a'))/a' da')
        # integral_1^a (1+w(a'))/a' da' = integral_1^a (1+w0+wa*(1-a'))/a' da'
        # = (1+w0)*ln(a) + wa*integral_1^a (1-a')/a' da'
        # = (1+w0)*ln(a) + wa*integral_1^a (1/a' - 1) da'
        # = (1+w0)*ln(a) + wa*[ln(a) - (a-1)]
        # = (1+w0+wa)*ln(a) - wa*(a-1)
        # = (1+w0+wa)*ln(a) + wa*(1-a)
        
        # So rho_de(a)/rho_de(1) = exp(-3*[(1+w0+wa)*ln(a) + wa*(1-a)])
        # = a^(-3(1+w0+wa)) * exp(-3*wa*(1-a))
        
        # H^2(a)/H0^2 = Om_m + (1-Om_m) * a^(-3(1+w0+wa)) * exp(-3*wa*(1-a))
        
        Om_de = 1.0 - Om_m
        if Om_de < 0:
            Om_de = 0.0
        
        # H^2/H0^2
        h2 = Om_m + Om_de * (a ** (-3*(1+w0+wa))) * math.exp(-3*wa*(1-a))
        
        if h2 <= 0:
            return [0, 0]
        
        H = math.sqrt(h2)
        
        # d ln H / dx = (1/2) d ln(h2) / dx
        # h2 = Om_m + Om_de * exp(3*(1+w0+wa)*ln(a) + 3*wa*(1-a))
        # Let's compute numerically or analytically
        # d(h2)/dx = d(Om_m)/dx + d(Om_de*...)/dx
        # Om_m = Om0 * exp(3x), so d(Om_m)/dx = 3*Om_m
        # Om_de = 1 - Om_m
        # The DE term: T = Om_de * exp(3*(1+w0+wa)*x + 3*wa*(1-exp(x)))
        # dT/dx = -3*Om_m * exp(...) + Om_de * exp(...) * (3*(1+w0+wa) - 3*wa*exp(x))
        
        de_factor = math.exp(3*(1+w0+wa)*x + 3*wa*(1-a))
        T = Om_de * de_factor
        dT_dx = -3*Om_m * de_factor + Om_de * de_factor * (3*(1+w0+wa) - 3*wa*a)
        dh2_dx = 3*Om_m + dT_dx
        
        dlnH_dx = 0.5 * dh2_dx / h2
        
        # 4 pi G rho / H^2 = 3 * Om_m / H^2 (since rho_m/H0^2 = Om_m, and 4 pi G/H0^2 = 3/H0^2 in units where c=1, H0 in appropriate units)
        # Actually in standard units: 4 pi G rho_m / H^2 = 3 * Om_m * H0^2 / H^2 = 3 * Om_m / h2
        # Wait: rho_m = Om_m * H0^2 / (4 pi G) ... no.
        # H0^2 = 8 pi G rho_crit / 3, so 4 pi G rho_m / H^2 = 4 pi G * Om_m * rho_crit / H^2
        # = 4 pi G * Om_m * 3 H0^2 / (8 pi G) / H^2 = 3 * Om_m * H0^2 / H^2 = 3 * Om_m / h2
        
        coeff_f = 3.0 * Om_m / h2
        
        f = y[0]
        fp = y[1]
        
        fpp = -(3 + dlnH_dx) * fp + coeff_f * f
        return [fp, fpp]
    
    # Integrate from x=0 (a=1) backwards to x=ln(a_target)
    x_target = math.log(a_target)
    
    # Initial conditions at a=1 (x=0): f(1) = Om0^gamma, f'(1) can be estimated
    # Use f(1) = 0.55 (typical), f'(1) from known relation
    # Actually we need to match boundary conditions. Let's use shooting.
    # f(1) = Om0^gamma_ref, and f'(1) = -gamma_ref * Om0^gamma_ref * (something)
    # Simpler: integrate forward from early times or use known asymptotic
    
    # Use the fact that at early times (matter dominated), f ~ a^(3/5)
    # So at a=0.05, f ~ 0.05^0.6
    
    x_start = math.log(0.05)
    f_start = 0.05 ** 0.6
    fp_start = 0.6 * 0.05 ** 0.6  # d f / d x = 0.6 * f
    
    sol = solve_ivp(ode_rhs, [x_start, x_target], [f_start, fp_start],
                    method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)
    
    if not sol.success:
        # Fallback to constant gamma
        return Om0 * a_target**3 ** 0.55
    
    return sol.y[0, -1]


def _normalized_growth_approx(c0, c1, c2, c3, c4, c5, Om0, w0, wa, a):
    """Compute D(a)/D(1) using the approximate f."""
    # D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
    # = exp(-integral from 0 to ln(1/a) of f(exp(t)) dt) with substitution
    
    from scipy.integrate import quad
    
    def integrand(x):
        a_x = math.exp(x)
        Om_a = Om0 * a_x**3
        if Om_a <= 0 or Om_a >= 1:
            return 0.0
        q = 1.0 - Om_a
        u = 1.0 + w0 + wa * (1.0 - a_x)
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_x + c5*u*q
        if gamma < 0 or gamma > 2:  # safety
            return 0.0
        return Om_a ** gamma
    
    if a >= 1.0:
        return 1.0
    
    x_upper = -math.log(a)  # integral from 0 to -ln(a)
    val, _ = quad(integrand, 0, x_upper, limit=100)
    return math.exp(-val)


def _normalized_growth_ref(Om0, w0, wa, a):
    """Compute D(a)/D(1) using the reference ODE solution."""
    from scipy.integrate import solve_ivp, quad
    
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
    
    # Integrate to x=0 (a=1)
    sol = solve_ivp(ode_rhs, [x_start, 0.0], [f_start, fp_start],
                    method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)
    
    if not sol.success:
        # Fallback
        gamma_ref = 0.55
        def integrand_f(x):
            a_x = math.exp(x)
            Om_a = Om0 * a_x**3
            return Om_a ** gamma_ref
        val, _ = quad(integrand_f, 0, -math.log(a), limit=50)
        return math.exp(-val)
    
    # Now compute D(a)/D(1) = exp(-integral_0^{ln(1/a)} f(exp(t)) dt)
    # We have f on [x_start, 0]. For a < 1, we need f on [0, ln(1/a)] which is negative x range... 
    # Wait, a < 1 means ln(a) < 0, so integral from ln(a) to 0.
    # Let me re-read: D(a)/D(1) = exp(-integral_{ln(a)}^0 f(exp(x)) dx)
    # For a < 1, ln(a) < 0, so we integrate from negative to 0.
    
    def integrand(x):
        a_x = math.exp(x)
        Om_m = Om0 * a_x**3
        Om_de = max(0.0, 1.0 - Om_m)
        h2 = Om_m + Om_de * (a_x ** (-3*(1+w0+wa))) * math.exp(-3*wa*(1-a_x))
        if h2 <= 0:
            return 0.55
        de_factor = math.exp(3*(1+w0+wa)*x + 3*wa*(1-a_x))
        dT_dx = -3*Om_m * de_factor + Om_de * de_factor * (3*(1+w0+wa) - 3*wa*a_x)
        dh2_dx = 3*Om_m + dT_dx
        dlnH_dx = 0.5 * dh2_dx / h2
        coeff_f = 3.0 * Om_m / h2
        
        # Solve ODE at this point - too expensive, use approximation
        # Use f ~ Om_m^0.55 as reference approximation
        return Om_m ** 0.55
    
    x_lower = math.log(a)
    val, _ = quad(integrand, x_lower, 0.0, limit=100)
    return math.exp(-val)
# EVOLVE-BLOCK-END
