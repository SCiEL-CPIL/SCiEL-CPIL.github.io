"""Rewritten consumption-saving policy solver using Euler equation fixed-point with high-resolution interpolation."""
import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'])
    transition = np.array(model['transition'])
    horizon = model['horizon']
    num_z = len(income)
    
    # CRRA utility and marginal utility
    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1.0 - gamma)) / (1.0 - gamma)

    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)

    # Inverse of marginal utility: c from u_prime
    def c_from_mu(mu):
        mu = np.maximum(mu, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / mu
        else:
            return mu ** (-1.0 / gamma)

    # Grid for solving: use a dense log-spaced grid for marginal utility
    # We'll solve for the value function derivative (marginal value of wealth)
    # on a fine grid, then interpolate back to resource_grid

    # Number of points for the fine grid - balance accuracy vs speed
    num_fine = 2000
    
    # Create a fine log-spaced grid for wealth m
    m_fine = np.geomspace(0.001, 40, num_fine)
    
    # We'll store the consumption policy on the fine grid for each (t, z)
    # and interpolate to resource_grid at the end
    
    # Initialize arrays for fine grid
    c_fine = np.zeros((horizon, num_z, num_fine))
    
    # Terminal condition: c[m] = m (consume everything at last period)
    # Actually, at t=5 (last period), c = m
    c_fine[horizon - 1, :, :] = m_fine[np.newaxis, np.newaxis, :]
    
    # Backward induction using Euler equation fixed-point
    # At each t from horizon-2 down to 0:
    #   c_t = argmax_c { u(c) + beta * sum_z' pi(z'|z) * V_{t+1}(R*(m-c) + income[z'], z') }
    # 
    # Using the envelope theorem: V'(m) = u'(c)
    # And the Euler equation: u'(c_t) = beta * R * sum_z' pi(z'|z) * V'_{t+1}(R*(m-c_t) + income[z'], z')
    #
    # So: u'(c_t) = beta * R * sum_z' pi(z'|z) * u'(c_{t+1})
    # where c_{t+1} is the optimal consumption next period given next period's wealth
    #
    # This is a fixed-point equation in c_t. We can solve it iteratively.
    
    for t in range(horizon - 2, -1, -1):
        # Check time budget
        if time.time() > deadline - 5.0:
            break
        
        # For each state z and each m in the fine grid, solve for c_t
        for z in range(num_z):
            # Transition probabilities from z to z'
            pi = transition[z]  # shape (num_z,)
            
            # Initialize c_t with a reasonable guess
            # Start with c = m * some fraction, say 0.5
            c_t = 0.5 * m_fine.copy()
            
            # Fixed-point iteration
            for iter in range(200):
                if time.time() > deadline - 5.0:
                    break
                
                # Compute next period's wealth for each z'
                # m_next[z'] = R * (m - c_t) + income[z']
                m_next = R * (m_fine[np.newaxis, :] - c_t[np.newaxis, :]) + income[:, np.newaxis]
                # m_next has shape (num_z, num_fine)
                
                # Ensure m_next is positive
                m_next = np.maximum(m_next, 1e-12)
                
                # Interpolate next period's consumption policy from c_fine[t+1]
                # c_fine[t+1, z', :] gives consumption at each m_fine point for state z'
                c_next = np.zeros_like(m_next)
                for zp in range(num_z):
                    # Interpolate c_fine[t+1, zp, :] at m_next[zp, :]
                    c_next[zp, :] = np.interp(m_next[zp, :], m_fine, c_fine[t + 1, zp, :])
                
                # Compute marginal utility of next period's consumption
                mu_next = u_prime(c_next)  # shape (num_z, num_fine)
                
                # Weighted sum over z'
                mu_expected = np.sum(pi[:, np.newaxis] * mu_next, axis=0)  # shape (num_fine,)
                
                # Euler equation: u'(c_t) = beta * R * mu_expected
                # So c_t_new = c_from_mu(beta * R * mu_expected)
                c_t_new = c_from_mu(beta * R * mu_expected)
                
                # Clip to valid range: 0 < c_t <= m
                c_t_new = np.clip(c_t_new, 1e-12, m_fine)
                
                # Check convergence
                diff = np.max(np.abs(c_t_new - c_t))
                c_t = c_t_new
                
                if diff < 1e-10:
                    break
            
            c_fine[t, z, :] = c_t
    
    # Now interpolate from the fine grid to resource_grid
    result = np.zeros((horizon, num_z, len(resource_grid)))
    for t in range(horizon):
        for z in range(num_z):
            result[t, z, :] = np.interp(resource_grid, m_fine, c_fine[t, z, :])
    
    # Ensure terminal condition: c[5] = resource_grid
    result[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    
    # Ensure all values are finite and in valid range
    result = np.where(np.isfinite(result), result, 0.0)
    result = np.clip(result, 1e-12, None)
    
    # Ensure c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_z):
            result[t, z, :] = np.minimum(result[t, z, :], resource_grid + 1e-10)
    
    return result
