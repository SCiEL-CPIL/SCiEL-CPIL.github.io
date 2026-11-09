"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize, differential_evolution
import time

# EVOLVE-BLOCK-START
def construct():
    """Return six coefficients for the growth-index approximation."""
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Default fallback: constant gamma approximation
    default_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    try:
        # Helper functions
        def compute_Om(a, Om0, w0, wa):
            """Compute Om(a) for CPL dark energy."""
            Edm = Om0 * a**(-3)
            Ede = (1 - Om0) * a**(-3*(1 + w0 + wa))
            return Edm / (Edm + Ede)
        
        def gamma_from_coeffs(coeffs, a, Om0, w0, wa):
            """Compute gamma from the six-coefficient formula."""
            c0, c1, c2, c3, c4, c5 = coeffs
            q = 1 - compute_Om(a, Om0, w0, wa)
            u = 1 + w0 + wa * (1 - a)
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
            return gamma
        
        def f_approx(coeffs, a, Om0, w0, wa):
            """Compute f = Om(a)^gamma from the approximation."""
            gamma = gamma_from_coeffs(coeffs, a, Om0, w0, wa)
            Om_a = compute_Om(a, Om0, w0, wa)
            return Om_a ** gamma
        
        def D_normalized_approx(coeffs, a, Om0, w0, wa):
            """Compute D(a)/D(1) using the approximation via numerical integration."""
            if a >= 1.0:
                return 1.0
            
            # Use a fixed grid for integration - much faster than quad
            n_points = 50
            x_vals = np.linspace(np.log(a), 0.0, n_points)
            a_vals = np.exp(x_vals)
            
            # Vectorized computation of f
            Om_vals = np.array([compute_Om(ai, Om0, w0, wa) for ai in a_vals])
            q_vals = 1 - Om_vals
            u_vals = 1 + w0 + wa * (1 - a_vals)
            
            c0, c1, c2, c3, c4, c5 = coeffs
            gamma_vals = c0 + c1*q_vals + c2*q_vals**2 + c3*u_vals + c4*wa*a_vals + c5*u_vals*q_vals
            f_vals = Om_vals ** gamma_vals
            
            # Integrate -ln(f) using trapezoidal rule
            integrand = -np.log(f_vals)
            integral = np.trapz(integrand, x_vals)
            return np.exp(-integral)
        
        def solve_reference(Om0, w0, wa, a_values):
            """Solve the ODE to get reference f and D for given cosmology."""
            def ode_system(x, y):
                a_x = np.exp(x)
                Om_x = compute_Om(a_x, Om0, w0, wa)
                w_a = w0 + wa * (1 - a_x)
                dlnH_dx = -1.5 * (1 + w_a)
                D, dD = y
                d2D = -(3 + dlnH_dx) * dD + 1.5 * Om_x * D
                return [dD, d2D]
            
            # Find f(1) using shooting method
            a_min = min(a_values)
            x_min = np.log(a_min)
            
            def shoot_f1(f1_guess):
                y0 = [1.0, f1_guess]
                try:
                    sol = solve_ivp(ode_system, [0.0, x_min], y0, 
                                  method='RK45', rtol=1e-10, atol=1e-12, dense_output=True)
                    if not sol.success:
                        return 1e10
                    y_min = sol.sol(x_min)
                    D_min, dD_min = y_min
                    # For small a, D ~ a * (const), so D/a should be constant
                    # We normalize so that D ~ a as a -> 0, meaning D/a -> 1
                    ratio = D_min / a_min
                    return abs(ratio - 1.0)
                except:
                    return 1e10
            
            from scipy.optimize import brentq
            try:
                f1 = brentq(shoot_f1, 0.1, 1.5, xtol=1e-10, maxiter=100)
            except:
                f1 = compute_Om(1.0, Om0, w0, wa) ** 0.55
            
            # Solve with correct f1
            y0 = [1.0, f1]
            sol = solve_ivp(ode_system, [0.0, x_min], y0, 
                           method='RK45', rtol=1e-10, atol=1e-12, dense_output=True)
            
            if not sol.success:
                return None
            
            return sol, f1
        
        # Define cosmology models to fit against - diverse set
        cosmologies = [
            (0.3, -1.0, 0.0),    # LambdaCDM
            (0.3, -1.0, 0.3),    # CPL with positive wa
            (0.3, -1.0, -0.3),   # CPL with negative wa
            (0.25, -0.9, 0.0),   # Different Om0
            (0.35, -1.1, 0.2),   # Different w0
            (0.3, -0.8, -0.2),   # Another CPL
            (0.28, -1.05, 0.15), # More variation
            (0.32, -0.95, -0.1), # More variation
        ]
        
        # a values to check - denser sampling
        a_values = np.linspace(0.05, 1.0, 30)
        
        # Precompute reference solutions
        ref_solutions = {}
        for Om0, w0, wa in cosmologies:
            if time.time() > deadline - 10:
                break
            try:
                ref_solutions[(Om0, w0, wa)] = solve_reference(Om0, w0, wa, a_values)
            except:
                ref_solutions[(Om0, w0, wa)] = None
        
        def objective(coeffs):
            """Compute maximum relative error across all cosmologies and a values."""
            max_err = 0.0
            for Om0, w0, wa in cosmologies:
                ref_data = ref_solutions.get((Om0, w0, wa))
                if ref_data is None:
                    continue
                sol, f1_ref = ref_data
                
                for a in a_values:
                    if a >= 1.0:
                        continue
                    x = np.log(a)
                    
                    # Reference values
                    y_ref = sol.sol(x)
                    D_ref, dD_ref = y_ref
                    f_ref = dD_ref / D_ref
                    
                    # Approximation values
                    f_app = f_approx(coeffs, a, Om0, w0, wa)
                    D_app = D_normalized_approx(coeffs, a, Om0, w0, wa)
                    
                    # Relative errors
                    err_f = abs(f_app - f_ref) / abs(f_ref)
                    err_D = abs(D_app - D_ref) / abs(D_ref)
                    
                    max_err = max(max_err, err_f, err_D)
            
            return 100 * max_err
        
        def check_gamma_bounds(coeffs):
            """Check if gamma stays in [0.1, 1.2] for all test points."""
            for Om0, w0, wa in cosmologies:
                for a in a_values:
                    gamma = gamma_from_coeffs(coeffs, a, Om0, w0, wa)
                    if gamma < 0.1 or gamma > 1.2:
                        return False
            return True
        
        # Initial guess - use the known good approximation
        x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        best_coeffs = x0
        best_score = objective(x0)
        
        # Strategy 1: Try physics-informed candidate guesses
        # The growth index has known behavior: gamma ≈ 0.55 for LambdaCDM,
        # varies with q and u for CPL models
        candidate_guesses = [
            [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],  # constant
            [0.55, 0.1, 0.0, 0.0, 0.0, 0.0],  # linear in q
            [0.55, -0.1, 0.0, 0.0, 0.0, 0.0],
            [0.55, 0.0, 0.1, 0.0, 0.0, 0.0],  # quadratic in q
            [0.55, 0.0, 0.0, 0.1, 0.0, 0.0],  # linear in u
            [0.55, 0.0, 0.0, -0.1, 0.0, 0.0],
            [0.55, 0.1, 0.0, 0.05, 0.0, 0.0],
            [0.55, -0.05, 0.05, 0.0, 0.0, 0.0],
            [0.55, 0.05, -0.05, 0.0, 0.05, 0.0],
            [0.55, 0.0, 0.0, 0.0, 0.1, 0.0],  # wa*a term
            [0.55, 0.0, 0.0, 0.0, 0.0, 0.1],  # u*q term
            [0.55, 0.05, 0.02, 0.03, 0.02, 0.02],  # combined
            [0.55, -0.03, 0.03, -0.02, 0.01, 0.01],
        ]
        
        for guess in candidate_guesses:
            if time.time() > deadline - 20:
                break
            score = objective(guess)
            if score < best_score and check_gamma_bounds(guess):
                best_coeffs = guess
                best_score = score
        
        # Strategy 2: Differential Evolution for global search
        if time.time() < deadline - 25:
            try:
                bounds = [(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)]
                de_result = differential_evolution(
                    objective, bounds, 
                    maxiter=30, tol=1e-3, popsize=12,
                    seed=42, polish=False
                )
                if de_result.fun < best_score and check_gamma_bounds(de_result.x):
                    best_coeffs = de_result.x
                    best_score = de_result.fun
            except Exception:
                pass
        
        # Strategy 3: Nelder-Mead refinement
        if time.time() < deadline - 10:
            try:
                nm_result = minimize(
                    objective, best_coeffs, method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-5,
                             'adaptive': True, 'initial_simplex': 0.05}
                )
                if nm_result.fun < best_score and check_gamma_bounds(nm_result.x):
                    best_coeffs = nm_result.x
                    best_score = nm_result.fun
            except Exception:
                pass
        
        # Strategy 4: Final polish
        if time.time() < deadline - 3:
            try:
                polish_result = minimize(
                    objective, best_coeffs, method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-7,
                             'adaptive': True, 'initial_simplex': 0.01}
                )
                if polish_result.fun < best_score and check_gamma_bounds(polish_result.x):
                    best_coeffs = polish_result.x
                    best_score = polish_result.fun
            except Exception:
                pass
        
        # Ensure coefficients are in [-10, 10]
        best_coeffs = np.clip(best_coeffs, -10, 10)
        
        # Final validity check
        if not check_gamma_bounds(best_coeffs):
            # Try to find valid coefficients near the best
            for scale in [0.9, 0.8, 0.7, 0.5, 0.3]:
                candidate = best_coeffs * scale
                if check_gamma_bounds(candidate):
                    best_coeffs = candidate
                    break
            else:
                best_coeffs = default_coeffs
        
        return list(best_coeffs)
    
    except Exception:
        return default_coeffs

# EVOLVE-BLOCK-END
