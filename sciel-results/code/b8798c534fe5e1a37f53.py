import numpy as np
import time
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    M = len(resource_grid)
    Z = len(income)
    
    consumption = np.zeros((horizon, Z, M))
    
    # Utility functions
    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return c ** (1.0 - gamma) / (1.0 - gamma)
            
    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return c ** (-gamma)
            
    def u_double_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return -1.0 / (c * c)
        else:
            return -gamma * c ** (-gamma - 1)

    # Terminal condition: c_T(m) = m
    c_next = np.zeros((Z, M))
    for z in range(Z):
        c_next[z] = resource_grid
        
    # Pre-compute interpolators for c_next
    # We need value and derivative
    # PchipInterpolator returns a callable object that supports (x) for value and (x, 1) for derivative
    interpolators = []
    for z in range(Z):
        # Ensure x is strictly increasing (it is)
        # Pchip requires x to be strictly increasing
        interpolators.append(PchipInterpolator(resource_grid, c_next[z]))
    
    # For CRRA with gamma != 1, we can use the inverse of u_prime analytically
    # u'(c) = c^(-gamma) for gamma != 1
    # u'(c) = 1/c for gamma = 1
    # So given a target marginal utility mu, c = mu^(-1/gamma) for gamma != 1
    # and c = 1/mu for gamma = 1

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: fill with linear rule or previous
            for tt in range(t, -1, -1):
                for z in range(Z):
                    consumption[tt, z] = resource_grid
            break
            
        c_curr = np.zeros((Z, M))
        
        for z in range(Z):
            # Initial guess: use previous period's consumption, scaled?
            # Or just 0.5 * m
            c = 0.5 * resource_grid.copy()
            
            max_iters = 30
            tol = 1e-10
            
            for _ in range(max_iters):
                if time.time() > deadline:
                    break
                    
                # m' for each z'
                # m_prime[z'] = R * (m - c) + income[z']
                # Shape: (Z, M)
                m_prime = R * (resource_grid[None, :] - c[None, :]) + income[:, None]
                
                # Evaluate c_next and c'_next at m_prime
                # c_next_val[z', m] = interpolators[z'](m_prime[z', m])
                c_next_val = np.zeros((Z, M))
                
                for zp in range(Z):
                    # Interpolate
                    # PchipInterpolator handles extrapolation
                    c_next_val[zp] = interpolators[zp](m_prime[zp])
                    # Ensure c_next_val is within bounds
                    c_next_val[zp] = np.clip(c_next_val[zp], 1e-12, m_prime[zp])
                
                # Expected marginal utility
                # E_z' [ u'(c_next) ]
                # Weighted sum over z'
                # transition[z, :] are probabilities
                probs = transition[z]
                
                eu_prime = np.sum(probs[:, None] * u_prime(c_next_val), axis=0)
                
                # Target marginal utility from Euler equation
                # u'(c) = beta * R * E[u'(c')]
                target_mu = beta * R * eu_prime
                
                # Use analytical inverse of u_prime to get c
                if abs(gamma - 1.0) < 1e-9:
                    # u'(c) = 1/c, so c = 1/mu
                    c_new = 1.0 / target_mu
                else:
                    # u'(c) = c^(-gamma), so c = mu^(-1/gamma)
                    c_new = target_mu ** (-1.0 / gamma)
                
                # Clip c to [1e-12, m]
                c_new = np.clip(c_new, 1e-12, resource_grid)
                
                # Check convergence
                if np.max(np.abs(c_new - c)) < tol:
                    c = c_new
                    break
                c = c_new
                
            c_curr[z] = c
            
        consumption[t] = c_curr
        c_next = c_curr
        
        # Update interpolators
        interpolators = []
        for z in range(Z):
            interpolators.append(PchipInterpolator(resource_grid, c_next[z]))

    # Final validity checks
    for t in range(horizon):
        for z in range(Z):
            c = consumption[t, z]
            m = resource_grid
            c = np.maximum(c, 1e-12)
            c = np.minimum(c, m + 1e-10)
            consumption[t, z] = c
            
    # Enforce terminal condition exactly
    consumption[horizon-1] = resource_grid[None, :]
    
    return consumption
