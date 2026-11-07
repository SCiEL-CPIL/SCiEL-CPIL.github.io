"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 148  # Safety margin below 150s

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    m_grid = resource_grid
    n_m = len(m_grid)

    # Use a dense internal grid for value function interpolation.
    # The evaluator interpolates our policy onto a 251-point grid.
    # We solve on a grid slightly denser than the evaluation grid to minimize interpolation error.
    n_solve = 501
    solve_grid = np.geomspace(0.001, 40, n_solve)

    def get_utility(c, gamma):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    def get_utility_prime(c, gamma):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return c ** (-gamma)

    def get_utility_double_prime(c, gamma):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return -1.0 / (c ** 2)
        else:
            return -gamma * c ** (-gamma - 1)

    # Value function V[t, z, :] has shape (horizon, 2, n_solve)
    # Policy c[t, z, :] has shape (horizon, 2, n_solve)
    V = np.zeros((horizon, 2, n_solve))
    c_opt = np.zeros((horizon, 2, n_solve))

    # Terminal period: consume everything
    c_opt[horizon - 1, :, :] = solve_grid[np.newaxis, :]
    V[horizon - 1, :, :] = get_utility(solve_grid[np.newaxis, :], gamma)

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 10:
            # Fallback: simple proportional rule for remaining periods
            for tt in range(t, -1, -1):
                c_opt[tt, :, :] = solve_grid[np.newaxis, :] / (horizon - tt)
                V[tt, :, :] = get_utility(c_opt[tt, :, :], gamma)
            break

        for z in range(2):
            m_vals = solve_grid  # shape (n_solve,)
            
            # We need to find c* that maximizes:
            # u(c) + beta * E_z'[ V_{t+1}(R*(m-c) + income[z'], z') ]
            
            # Initialize c with a reasonable guess (e.g., proportional consumption)
            # For CRRA, c/m is roughly constant. Let's start with 1/(1+beta) or similar.
            # A safer start is a fraction of m, bounded away from 0 and m.
            c = m_vals * 0.5
            
            # Newton-Raphson iterations
            # We want to solve F(c) = u'(c) + beta * d/dc E[V_{t+1}(m_next)] = 0
            # d/dc E[V_{t+1}] = E[ V'_{t+1}(m_next) * (-R) ] = -R * E[ V'_{t+1}(m_next) ]
            # So F(c) = u'(c) - beta * R * E[ V'_{t+1}(m_next) ]
            
            # Second derivative for Newton step:
            # F'(c) = u''(c) - beta * R * d/dc E[ V'_{t+1}(m_next) ]
            # d/dc V'_{t+1}(m_next) = V''_{t+1}(m_next) * (-R)
            # So d/dc E[V'_{t+1}] = -R * E[ V''_{t+1}(m_next) ]
            # F'(c) = u''(c) + beta * R^2 * E[ V''_{t+1}(m_next) ]
            
            # Note: V is concave, so V' > 0 and V'' < 0.
            # F(c) is strictly decreasing (u'' < 0, and the second term is negative? 
            # Wait. V'' is negative. E[V''] is negative.
            # F'(c) = u''(c) + beta * R^2 * E[V''(m_next)]
            # Both terms are negative. So F is strictly decreasing.
            # This means F(c) goes from +infinity (as c->0) to -infinity (as c->m).
            # There is a unique root. Newton's method should converge monotonically if we start in the right place.
            
            # Pre-compute interpolants for V_{t+1} and its derivatives?
            # We need V' and V'' of the next period's value function.
            # We can approximate V' and V'' using finite differences on the grid, 
            # or better, use the fact that we are solving for c_opt[t+1] which is known?
            # No, V_{t+1} is already computed. We can interpolate V_{t+1} and differentiate.
            # Since we are on a log-spaced grid, simple finite differences might be noisy.
            # Let's use a polynomial or spline interpolant for V_{t+1} and take derivatives.
            # However, creating splines every iteration is expensive.
            # Alternative: Use the optimal policy c_opt[t+1] to express V_{t+1}(m) = u(c_opt[t+1](m)) + beta E[V_{t+2}(...)]
            # This is recursive.
            
            # Simpler approach for derivatives:
            # Since we have V[t+1, z, :] on solve_grid, we can compute V' and V'' 
            # using np.gradient or a spline.
            # Let's use a cubic spline for V_{t+1} to get smooth derivatives.
            # We do this once per (t, z) pair before the Newton loop.
            
            # First, check if we have enough time for Newton's method
            # If not, fall back to grid search
            elapsed = time.time() - start_time
            remaining = deadline - elapsed
            
            if remaining < 5:
                # Not enough time, use grid search fallback
                n_c = 100
                c_eval = m_vals[np.newaxis, :] * np.linspace(0.001, 1.0, n_c)[:, np.newaxis]
                c_eval = np.maximum(c_eval, 1e-12)
                m_next_0 = R * (m_vals[np.newaxis, np.newaxis, :] - c_eval[:, np.newaxis, :]) + income[0]
                m_next_1 = R * (m_vals[np.newaxis, np.newaxis, :] - c_eval[:, np.newaxis, :]) + income[1]
                
                m_next_0 = np.clip(m_next_0, solve_grid[0], solve_grid[-1])
                m_next_1 = np.clip(m_next_1, solve_grid[0], solve_grid[-1])
                
                V_next_0 = np.interp(m_next_0, solve_grid, V[t + 1, 0, :])
                V_next_1 = np.interp(m_next_1, solve_grid, V[t + 1, 1, :])
                
                expected_V = transition[z, 0] * V_next_0 + transition[z, 1] * V_next_1
                obj = get_utility(c_eval, gamma) + beta * expected_V
                best_idx = np.argmax(obj, axis=0)
                c = c_eval[best_idx, np.arange(n_solve)]
                c = np.clip(c, 1e-12, m_vals + 1e-10)
                c_opt[t, z, :] = c
                
                m_next_z0 = R * (m_vals - c) + income[0]
                m_next_z1 = R * (m_vals - c) + income[1]
                m_next_z0 = np.clip(m_next_z0, solve_grid[0], solve_grid[-1])
                m_next_z1 = np.clip(m_next_z1, solve_grid[0], solve_grid[-1])
                
                V_next_val = transition[z, 0] * np.interp(m_next_z0, solve_grid, V[t+1, 0, :]) + \
                             transition[z, 1] * np.interp(m_next_z1, solve_grid, V[t+1, 1, :])
                V[t, z, :] = get_utility(c, gamma) + beta * V_next_val
                continue

            # Try Newton's method with spline derivatives
            try:
                from scipy.interpolate import CubicSpline
                spline_V1 = CubicSpline(solve_grid, V[t + 1, 0, :], bc_type='natural')
                spline_V2 = CubicSpline(solve_grid, V[t + 1, 1, :], bc_type='natural')
            except Exception:
                # Spline failed, use grid search
                n_c = 150
                c_eval = m_vals[np.newaxis, :] * np.linspace(0.001, 1.0, n_c)[:, np.newaxis]
                c_eval = np.maximum(c_eval, 1e-12)
                m_next_0 = R * (m_vals[np.newaxis, np.newaxis, :] - c_eval[:, np.newaxis, :]) + income[0]
                m_next_1 = R * (m_vals[np.newaxis, np.newaxis, :] - c_eval[:, np.newaxis, :]) + income[1]
                
                m_next_0 = np.clip(m_next_0, solve_grid[0], solve_grid[-1])
                m_next_1 = np.clip(m_next_1, solve_grid[0], solve_grid[-1])
                
                V_next_0 = np.interp(m_next_0, solve_grid, V[t + 1, 0, :])
                V_next_1 = np.interp(m_next_1, solve_grid, V[t + 1, 1, :])
                
                expected_V = transition[z, 0] * V_next_0 + transition[z, 1] * V_next_1
                obj = get_utility(c_eval, gamma) + beta * expected_V
                best_idx = np.argmax(obj, axis=0)
                c = c_eval[best_idx, np.arange(n_solve)]
                c = np.clip(c, 1e-12, m_vals + 1e-10)
                c_opt[t, z, :] = c
                
                m_next_z0 = R * (m_vals - c) + income[0]
                m_next_z1 = R * (m_vals - c) + income[1]
                m_next_z0 = np.clip(m_next_z0, solve_grid[0], solve_grid[-1])
                m_next_z1 = np.clip(m_next_z1, solve_grid[0], solve_grid[-1])
                
                V_next_val = transition[z, 0] * np.interp(m_next_z0, solve_grid, V[t+1, 0, :]) + \
                             transition[z, 1] * np.interp(m_next_z1, solve_grid, V[t+1, 1, :])
                V[t, z, :] = get_utility(c, gamma) + beta * V_next_val
                continue

            # Newton Iterations with convergence check
            max_iter = 30
            tol = 1e-7
            converged = False
            
            for _ in range(max_iter):
                # m_next for z' = 0 and z' = 1
                m_next_0 = R * (m_vals - c) + income[0]
                m_next_1 = R * (m_vals - c) + income[1]
                
                # Ensure m_next is within grid bounds for interpolation
                m_next_0 = np.clip(m_next_0, solve_grid[0], solve_grid[-1])
                m_next_1 = np.clip(m_next_1, solve_grid[0], solve_grid[-1])
                
                # Evaluate V' and V'' at m_next
                V_prime_0 = spline_V1(m_next_0, 1)
                V_double_prime_0 = spline_V1(m_next_0, 2)
                V_prime_1 = spline_V2(m_next_1, 1)
                V_double_prime_1 = spline_V2(m_next_1, 2)
                
                # Expected derivatives
                E_V_prime = transition[z, 0] * V_prime_0 + transition[z, 1] * V_prime_1
                E_V_double_prime = transition[z, 0] * V_double_prime_0 + transition[z, 1] * V_double_prime_1
                
                # F(c) = u'(c) - beta * R * E[V'(m_next)]
                u_prime = get_utility_prime(c, gamma)
                F = u_prime - beta * R * E_V_prime
                
                # F'(c) = u''(c) + beta * R^2 * E[V''(m_next)]
                u_double_prime = get_utility_double_prime(c, gamma)
                F_prime = u_double_prime + beta * R**2 * E_V_double_prime
                
                # Check if F_prime is too close to zero (numerical issue)
                if np.any(np.abs(F_prime) < 1e-10):
                    # Fall back to bisection-like step
                    c_new = c * 1.1 if np.mean(F) > 0 else c * 0.9
                else:
                    # Newton step with damping to ensure convergence
                    step = F / F_prime
                    # Limit step size to prevent overshooting
                    max_step = 0.5 * np.maximum(m_vals - c, c - 1e-12)
                    step = np.clip(step, -max_step, max_step)
                    c_new = c + step
                
                # Clip to valid range [1e-12, m_vals]
                c_new = np.clip(c_new, 1e-12, m_vals)
                
                # Check convergence
                if np.max(np.abs(c_new - c)) < tol * np.maximum(1.0, np.max(m_vals)):
                    c = c_new
                    converged = True
                    break
                c = c_new
            
            if not converged:
                # Newton didn't converge, try a few more iterations with relaxed tolerance
                for _ in range(10):
                    m_next_0 = R * (m_vals - c) + income[0]
                    m_next_1 = R * (m_vals - c) + income[1]
                    m_next_0 = np.clip(m_next_0, solve_grid[0], solve_grid[-1])
                    m_next_1 = np.clip(m_next_1, solve_grid[0], solve_grid[-1])
                    
                    V_prime_0 = spline_V1(m_next_0, 1)
                    V_double_prime_0 = spline_V1(m_next_0, 2)
                    V_prime_1 = spline_V2(m_next_1, 1)
                    V_double_prime_1 = spline_V2(m_next_1, 2)
                    
                    E_V_prime = transition[z, 0] * V_prime_0 + transition[z, 1] * V_prime_1
                    E_V_double_prime = transition[z, 0] * V_double_prime_0 + transition[z, 1] * V_double_prime_1
                    
                    u_prime = get_utility_prime(c, gamma)
                    F = u_prime - beta * R * E_V_prime
                    u_double_prime = get_utility_double_prime(c, gamma)
                    F_prime = u_double_prime + beta * R**2 * E_V_double_prime
                    
                    if np.any(np.abs(F_prime) < 1e-10):
                        c_new = c * 1.05 if np.mean(F) > 0 else c * 0.95
                    else:
                        step = F / F_prime
                        max_step = 0.3 * np.maximum(m_vals - c, c - 1e-12)
                        step = np.clip(step, -max_step, max_step)
                        c_new = c + step
                    
                    c_new = np.clip(c_new, 1e-12, m_vals)
                    if np.max(np.abs(c_new - c)) < 1e-6 * np.maximum(1.0, np.max(m_vals)):
                        c = c_new
                        break
                    c = c_new
            
            c_opt[t, z, :] = c
            # Calculate V for this period
            m_next_z0 = R * (m_vals - c) + income[0]
            m_next_z1 = R * (m_vals - c) + income[1]
            m_next_z0 = np.clip(m_next_z0, solve_grid[0], solve_grid[-1])
            m_next_z1 = np.clip(m_next_z1, solve_grid[0], solve_grid[-1])
            
            V_next_val = transition[z, 0] * spline_V1(m_next_z0) + transition[z, 1] * spline_V2(m_next_z1)
            V[t, z, :] = get_utility(c, gamma) + beta * V_next_val

    # Interpolate the solution from solve_grid (n_solve points) to resource_grid (101 points)
    # Use monotone cubic interpolation for better accuracy
    consumption = np.zeros((horizon, 2, n_m))
    for z in range(2):
        for t in range(horizon):
            try:
                from scipy.interpolate import PchipInterpolator
                pchip = PchipInterpolator(solve_grid, c_opt[t, z, :])
                consumption[t, z, :] = pchip(resource_grid)
            except Exception:
                consumption[t, z, :] = np.interp(resource_grid, solve_grid, c_opt[t, z, :])

    # Final check and clipping
    consumption = np.clip(consumption, 1e-12, resource_grid[np.newaxis, np.newaxis, :] + 1e-10)

    # Ensure c[5] is exactly the grid
    consumption[horizon - 1, :, :] = resource_grid[np.newaxis, :]

    # Final validity check
    consumption = np.where(np.isfinite(consumption), consumption, resource_grid[np.newaxis, np.newaxis, :])
    consumption = np.clip(consumption, 1e-12, resource_grid[np.newaxis, np.newaxis, :] + 1e-10)

    return consumption
# EVOLVE-BLOCK-END
