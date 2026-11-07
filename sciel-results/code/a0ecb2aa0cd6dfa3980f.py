import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145.0
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    n_z = len(income)
    
    # Evaluation grid for internal computation
    # Use 251 points to match evaluation resolution
    m_eval = np.geomspace(0.001, 40, 251)
    n_m = len(m_eval)
    
    # Utility and derivatives
    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        return c**(1-gamma) / (1-gamma)
        
    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        return c**(-gamma)
        
    def u_double_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return -1.0 / (c*c)
        return -gamma * c**(-gamma-1)
        
    # Storage
    V = np.zeros((horizon, n_z, n_m))
    C = np.zeros((horizon, n_z, n_m))
    
    # Terminal condition
    V[horizon-1, :, :] = u(m_eval)[np.newaxis, :, :]
    C[horizon-1, :, :] = m_eval[np.newaxis, :, :]
    
    # Precompute interpolators for V and MU for t=horizon-1
    # We will update these as we go backward
    
    # Backward induction
    for t in range(horizon-2, -1, -1):
        if time.time() > deadline:
            break
            
        # Prepare interpolators for next period (t+1)
        # MU_next[z][m] = u'( C[t+1, z, m] )
        # V_next[z][m] = V[t+1, z, m]
        
        mu_interp = []
        v_interp = []
        for z in range(n_z):
            c_next = C[t+1, z, :]
            mu_vals = u_prime(c_next)
            # Ensure strictly increasing x for Pchip
            # m_eval is strictly increasing
            mu_interp.append(PchipInterpolator(m_eval, mu_vals))
            v_interp.append(PchipInterpolator(m_eval, V[t+1, z, :]))
            
        for z in range(n_z):
            c_opt = np.zeros(n_m)
            
            for i_m in range(n_m):
                m = m_eval[i_m]
                
                # Newton's method to solve Euler equation
                # f(c) = u'(c) - beta * R * E[ MU_next[z']( R(m-c) + income[z'] ) ]
                
                # Helper to evaluate expectation
                def exp_mu(c_val):
                    m_primes = R * (m - c_val) + income
                    m_primes_clipped = np.clip(m_primes, m_eval[0], m_eval[-1])
                    mus = np.array([mu_interp[zp](m_primes_clipped[zp]) for zp in range(n_z)])
                    return np.dot(transition[z, :], mus)
                    
                def f(c_val):
                    return u_prime(c_val) - beta * R * exp_mu(c_val)
                    
                def df(c_val):
                    # Finite difference
                    eps = 1e-6
                    return (f(c_val + eps) - f(c_val - eps)) / (2 * eps)
                    
                c = 0.5 * m
                c_low = 1e-12
                c_high = m - 1e-12
                
                for _ in range(30):
                    val = f(c)
                    if abs(val) < 1e-10:
                        break
                    dval = df(c)
                    if abs(dval) < 1e-12:
                        # Bisection
                        if val > 0:
                            c_high = c
                        else:
                            c_low = c
                        c = (c_low + c_high) / 2
                        continue
                    
                    c_new = c - val / dval
                    if c_new <= c_low or c_new >= c_high:
                        c_new = (c_low + c_high) / 2
                        
                    if val > 0:
                        c_high = c
                    else:
                        c_low = c
                        
                    c = c_new
                    if abs(c - c_new) < 1e-10:
                        break
                        
                c_opt[i_m] = c
                
            C[t, z, :] = c_opt
            
            # Compute V
            for i_m in range(n_m):
                m = m_eval[i_m]
                c = c_opt[i_m]
                m_primes = R * (m - c) + income
                m_primes_clipped = np.clip(m_primes, m_eval[0], m_eval[-1])
                vs = np.array([v_interp[zp](m_primes_clipped[zp]) for zp in range(n_z)])
                V[t, z, i_m] = u(c) + beta * np.dot(transition[z, :], vs)
                
    # Interpolate to result grid
    result = np.zeros((horizon, n_z, len(resource_grid)))
    for t in range(horizon):
        if time.time() > deadline:
            break
        for z in range(n_z):
            try:
                c_interp = PchipInterpolator(m_eval, C[t, z, :])
                result[t, z, :] = c_interp(resource_grid)
            except Exception:
                result[t, z, :] = np.interp(resource_grid, m_eval, C[t, z, :])
                
    # Enforce constraints
    for t in range(horizon):
        if t == horizon - 1:
            result[t, :, :] = resource_grid[np.newaxis, :, :]
        else:
            m_grid_expanded = resource_grid[np.newaxis, :, :]
            result[t] = np.clip(result[t], 1e-12, m_grid_expanded + 1e-10)
            
    return result
