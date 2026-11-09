"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import brentq
import time

def solve_consumption(model, resource_grid, budget):
    t_start = time.time()
    deadline = 145.0  # safety margin before 150s
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = len(income)
    n_grid = len(resource_grid)
    
    # Utility derivatives
    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    def u_prime_scalar(c):
        c = max(c, 1e-12)
        if gamma == 1:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    # Policy: consumption[t, z, grid_index]
    policy = np.zeros((horizon, n_states, n_grid))
    
    # Last period: consume all
    policy[horizon - 1, :, :] = resource_grid[None, :]
    
    # Backward induction from t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        # Check time
        if time.time() > deadline - 5.0:
            # Fallback: simple rule
            for z in range(n_states):
                policy[t, z, :] = np.minimum(resource_grid, resource_grid * 0.5)
            break
        
        for z in range(n_states):
            for i in range(n_grid):
                m = resource_grid[i]
                
                # Define the Euler equation residual
                def euler_residual(c):
                    c = max(c, 1e-12)
                    c = min(c, m)  # can't consume more than resources
                    
                    lhs = u_prime_scalar(c)
                    
                    # Compute expected RHS
                    rhs = 0.0
                    for zp in range(n_states):
                        prob = transition[z, zp]
                        if prob < 1e-15:
                            continue
                        m_next = R * (m - c) + income[zp]
                        m_next = max(m_next, 1e-12)
                        # Interpolate next period's policy
                        c_next = np.interp(m_next, resource_grid, policy[t + 1, zp, :])
                        c_next = max(c_next, 1e-12)
                        rhs += prob * u_prime_scalar(c_next)
                    
                    rhs *= beta * R
                    
                    return lhs - rhs
                
                # Solve for c
                # c must be in (0, m]
                c_lo = 1e-12
                c_hi = m
                
                try:
                    f_lo = euler_residual(c_lo)
                    f_hi = euler_residual(c_hi)
                    
                    if f_lo * f_hi > 0:
                        # No sign change; use boundary
                        if f_lo < 0:
                            c_sol = c_hi
                        else:
                            c_sol = c_lo
                    else:
                        c_sol = brentq(euler_residual, c_lo, c_hi, 
                                        xtol=1e-10, rtol=1e-10, maxiter=50)
                except Exception:
                    c_sol = m * 0.5
                
                policy[t, z, i] = c_sol
    
    # Ensure constraints
    for t in range(horizon):
        for z in range(n_states):
            policy[t, z, :] = np.clip(policy[t, z, :], 1e-12, resource_grid + 1e-10)
    
    # Final period must equal grid
    policy[horizon - 1, :, :] = resource_grid[None, :]
    
    return policy
# EVOLVE-BLOCK-END
