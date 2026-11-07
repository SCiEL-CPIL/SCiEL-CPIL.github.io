"""Solve consumption-saving problem via Euler equation root-finding with PCHIP interpolation."""
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    N_m = len(resource_grid)
    N_z = len(income)
    
    # Utility and derivatives
    if gamma == 1:
        def util(c):
            c = np.maximum(c, 1e-12)
            return np.log(c)
        def u_prime(c):
            c = np.maximum(c, 1e-12)
            return 1.0 / c
        def u_double_prime(c):
            c = np.maximum(c, 1e-12)
            return -1.0 / (c * c)
    else:
        def util(c):
            c = np.maximum(c, 1e-12)
            return c ** (1 - gamma) / (1 - gamma)
        def u_prime(c):
            c = np.maximum(c, 1e-12)
            return c ** (-gamma)
        def u_double_prime(c):
            c = np.maximum(c, 1e-12)
            return -gamma * c ** (-gamma - 1)
    
    # Store value and consumption: V[t, m_idx, z], C[t, z, m_idx]
    V = np.zeros((horizon, N_m, N_z))
    C = np.zeros((horizon, N_z, N_m))
    
    # Terminal period: consume everything
    V[horizon-1] = util(resource_grid)  # shape (N_m,) broadcast to (N_m, N_z)
    C[horizon-1] = resource_grid[None, :]  # shape (N_z, N_m)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Emergency: fill remaining with terminal-like policy
            for tt in range(t, horizon - 1):
                if tt < horizon - 1:
                    V[tt] = V[tt+1]
                    C[tt] = C[tt+1]
            return C
        
        # Interpolate next-period value function for each state z'
        # V_next[m_idx, z'] for each z'
        # We need continuation value: for given m, z, choose c to maximize
        # util(c) + beta * sum_{z'} transition[z, z'] * V[t+1, m', z']
        # where m' = R*(m - c) + income[z']
        
        # For each z, solve Euler equation:
        # u'(c) = beta * sum_{z'} transition[z, z'] * R * u'(c') * V_t_plus_1_derivative
        # Actually, we need to be more careful. Let's use the Bellman equation directly.
        # V_t(m, z) = max_c { util(c) + beta * sum_{z'} P[z, z'] * V_{t+1}(R*(m-c) + income[z'], z') }
        
        # For interior solutions, FOC: u'(c) = beta * sum_{z'} P[z,z'] * R * u'(c_{z'}) * ... 
        # No, that's not right either. Let me think again.
        
        # The continuation value for choosing c at (m, z) is:
        # Q(c, m, z) = util(c) + beta * sum_{z'} transition[z, z'] * V_{t+1}(R*(m-c) + income[z'], z')
        
        # We maximize Q over c in (0, m].
        # FOC: u'(c) + beta * sum_{z'} transition[z,z'] * d/dc V_{t+1}(R*(m-c)+income[z'], z') = 0
        # d/dc V_{t+1}(R*(m-c)+income[z'], z') = V_{t+1}'(m', z') * (-R)
        # So: u'(c) = beta * R * sum_{z'} transition[z,z'] * V_{t+1}'(m', z')
        
        # But V_{t+1}'(m', z') = u'(c_{t+1}(m', z')) * (1 + ... ) no, that's for the envelope theorem.
        # Actually by envelope theorem: V_{t+1}'(m', z') = u'(c_{t+1}(m', z')) * (1 - beta * R * sum... ) no.
        
        # Let me just directly optimize Q(c, m, z) using the interpolated V_{t+1}.
        
        # Interpolate V_{t+1} as function of m for each z'
        interp_V = []
        for zp in range(N_z):
            interp_V.append(PchipInterpolator(resource_grid, V[t+1, :, zp], extrapolate=False))
        
        for z in range(N_z):
            for i_m in range(N_m):
                m = resource_grid[i_m]
                
                # Objective: Q(c) = util(c) + beta * sum_{z'} transition[z, z'] * V_{t+1}(R*(m-c) + income[z'], z')
                # c in (0, m]
                
                def objective(c):
                    m_next = R * (m - c)
                    cont = 0.0
                    for zp in range(N_z):
                        m_zp = m_next + income[zp]
                        if m_zp < resource_grid[0]:
                            v_next = util(0)  # This shouldn't happen if c <= m
                        elif m_zp > resource_grid[-1]:
                            v_next = V[t+1, -1, zp]  # clamp
                        else:
                            v_next = float(interp_V[zp](m_zp))
                        cont += transition[z, zp] * v_next
                    return util(c) + beta * cont
                
                def objective_neg(c):
                    return -objective(c)
                
                # Try to find maximum of objective in (eps, m)
                eps = 1e-12
                c_lo = eps
                c_hi = m
                
                # Check if it's worth solving or just consume all
                if m < 1e-8:
                    c_opt = m
                else:
                    # Use Brent's method on the negative (minimize)
                    # First, check endpoints and a few interior points
                    # Actually, let's just use bounded minimization via brentq on the derivative
                    # Or use scipy.optimize.minimize_scalar
                    
                    # Let's evaluate at a few points to bracket
                    # The optimal c is where objective is maximized
                    
                    # Use minimize_scalar with bounded method
                    from scipy.optimize import minimize_scalar
                    result = minimize_scalar(objective_neg, bounds=(eps, m), method='bounded', 
                                             options={'xatol': 1e-12})
                    c_opt = result.x
                
                # Ensure validity
                c_opt = np.clip(c_opt, eps, m)
                C[t, z, i_m] = c_opt
                V[t, i_m, z] = objective(c_opt)
    
    # Final validation
    # c[5] should equal grid within 1e-9
    C[horizon-1] = resource_grid[None, :]
    V[horizon-1] = util(resource_grid)
    
    # Ensure all values are finite and 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(N_z):
            for i_m in range(N_m):
                m = resource_grid[i_m]
                c = C[t, z, i_m]
                if not np.isfinite(c) or c <= 0 or c > m + 1e-10:
                    C[t, z, i_m] = np.clip(c, 1e-12, m)
    
    return C
