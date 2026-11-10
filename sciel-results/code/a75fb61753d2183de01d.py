import numpy as np
import scipy.integrate
import scipy.optimize
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 seconds margin for safety

    # 1. Define a representative set of cosmologies
    # We need to cover the likely validation space.
    # Omega_m0 in [0.2, 0.4], w0 in [-1.5, -0.5], wa in [-1.0, 1.0]
    om_m0_vals = np.array([0.2, 0.3, 0.4])
    w0_vals = np.array([-1.5, -1.0, -0.5])
    wa_vals = np.array([-1.0, 0.0, 1.0])
    
    cosmologies = []
    for om in om_m0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                cosmologies.append((om, w0, wa))

    # 2. Define a grid for a values
    # Log-spaced from 0.05 to 1.0
    # Use a moderate grid to balance accuracy and speed
    a_grid = np.logspace(np.log10(0.05), 0.0, 100)
    x_grid = np.log(a_grid)
    
    # 3. Pre-compute reference solutions
    # We compute f_ref and D_ref for each cosmology
    ref_data = []
    
    def get_om_m(a, om_m0, w0, wa):
        om_l0 = 1.0 - om_m0
        de_term = om_l0 * (a ** (-3.0 * (1.0 + w0))) * np.exp(-3.0 * wa * (a - 1.0))
        m_term = om_m0 * (a ** -3.0)
        return m_term / (m_term + de_term)

    for (om_m0, w0, wa) in cosmologies:
        om_l0 = 1.0 - om_m0
        
        def w_of_a(a):
            return w0 + wa * (1.0 - a)
        
        def om_m_of_a(a):
            return get_om_m(a, om_m0, w0, wa)

        # Integrate ODE from a=1 (x=0) backwards to a_min
        # D'' + (3/2 + 3/2 w) D' - 3/2 Om_m D = 0
        # Initial conditions at a=1: D(1)=1, D'(1) = f(1)
        # Use Linder's approximation for gamma at a=1 as seed
        f_seed = om_m0 ** 0.55
        
        def ode_system(x, y):
            a = np.exp(x)
            om_m = om_m_of_a(a)
            w_a = w_of_a(a)
            dD = y[1]
            dDp = -1.5 * (1.0 + w_a) * y[1] + 1.5 * om_m * y[0]
            return [dD, dDp]

        x_min = np.log(0.05)
        
        try:
            sol = scipy.integrate.solve_ivp(ode_system, [0, x_min], [1.0, f_seed], 
                                            method='DOP853', rtol=1e-8, atol=1e-10,
                                            dense_output=True, max_step=0.1)
            
            if not sol.success:
                raise RuntimeError("Integration failed")

            sol_grid = sol.sol(x_grid)
            D_vals = sol_grid[0]
            f_vals = sol_grid[1] / D_vals
            
            # Check for NaNs or infinities
            if not np.all(np.isfinite(D_vals)) or not np.all(np.isfinite(f_vals)):
                raise RuntimeError("Non-finite values")
                
        except Exception:
            # Fallback to constant gamma
            gamma_const = 0.55
            om_m_grid = np.array([om_m_of_a(a) for a in a_grid])
            f_vals = om_m_grid ** gamma_const
            x_rev = -x_grid[::-1]
            f_rev = f_vals[::-1]
            dx = x_rev[1] - x_rev[0]
            integrals_rev = np.zeros_like(x_rev)
            for i in range(1, len(x_rev)):
                integrals_rev[i] = integrals_rev[i-1] + 0.5 * (f_rev[i-1] + f_rev[i]) * dx
            integrals = integrals_rev[::-1]
            D_vals = np.exp(-integrals)

        ref_data.append((a_grid, D_vals, f_vals, om_m0, w0, wa))

    # 4. Define the objective function: Max relative error
    def objective(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0
        
        for (a_grid, D_ref, f_ref, om_m0, w0, wa) in ref_data:
            om_m_grid = get_om_m(a_grid, om_m0, w0, wa)
            q_grid = 1.0 - om_m_grid
            u_grid = 1.0 + w0 + wa * (1.0 - a_grid)
            
            gamma_grid = c0 + c1*q_grid + c2*q_grid**2 + c3*u_grid + c4*wa*a_grid + c5*u_grid*q_grid
            
            # Check gamma bounds
            if np.any(gamma_grid < 0.1) or np.any(gamma_grid > 1.2):
                return 1000.0  # Penalize out-of-bounds heavily
            
            f_approx = om_m_grid ** gamma_grid
            rel_err_f = np.abs((f_approx - f_ref) / f_ref)
            max_err = max(max_err, np.max(rel_err_f))
            
            # Compute D_approx
            # D = exp(-int f dx)
            # Use cumulative trapezoidal rule
            x_rev = -x_grid[::-1]
            f_rev = f_approx[::-1]
            dx = x_rev[1] - x_rev[0]
            integrals_rev = np.zeros_like(x_rev)
            # Vectorized cumulative sum
            # integrals_rev[i] = sum_{j=0}^{i-1} 0.5 * (f[j] + f[j+1]) * dx
            # This is equivalent to cumulative sum of trapezoids
            trapezoids = 0.5 * (f_rev[:-1] + f_rev[1:]) * dx
            integrals_rev[1:] = np.cumsum(trapezoids)
            integrals = integrals_rev[::-1]
            D_approx = np.exp(-integrals)
            
            rel_err_D = np.abs((D_approx - D_ref) / D_ref)
            max_err = max(max_err, np.max(rel_err_D))
            
        return max_err

    # 5. Optimization
    # Initial guess: Constant gamma 0.55
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_coeffs = x0
    best_cost = objective(x0)
    
    # Use Nelder-Mead for derivative-free optimization
    # This is robust for non-smooth max-error objectives
    try:
        if time.time() < deadline - 10.0:
            result = scipy.optimize.minimize(objective, x0, method='Nelder-Mead', 
                                             options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 200, 'maxfev': 200})
            
            if result.fun < best_cost:
                best_cost = result.fun
                best_coeffs = result.x
                
    except Exception:
        pass

    # 6. Final Validation and Safety Check
    # Check if best_coeffs is valid (gamma in [0.1, 1.2])
    valid = True
    # Use a slightly different grid for validation to ensure robustness
    test_a = np.logspace(np.log10(0.01), 0.0, 200)
    for (om_m0, w0, wa) in cosmologies:
        om_m_grid = get_om_m(test_a, om_m0, w0, wa)
        q_grid = 1.0 - om_m_grid
        u_grid = 1.0 + w0 + wa * (1.0 - test_a)
        gamma_grid = best_coeffs[0] + best_coeffs[1]*q_grid + best_coeffs[2]*q_grid**2 + best_coeffs[3]*u_grid + best_coeffs[4]*wa*test_a + best_coeffs[5]*u_grid*q_grid
        if np.any(gamma_grid < 0.1) or np.any(gamma_grid > 1.2):
            valid = False
            break
            
    if not valid:
        # Revert to safe constant
        best_coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    return best_coeffs.tolist()
# EVOLVE-BLOCK-END
