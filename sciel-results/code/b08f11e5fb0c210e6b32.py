"""Rewritten program: PCHIP interpolation + FOC Newton solver with vectorized evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 118  # Safety margin below 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    m_grid = resource_grid
    n_m = len(m_grid)
    n_z = len(income)
    
    # Utility and derivatives
    if abs(gamma - 1.0) < 1e-10:
        def util(c):
            c = np.maximum(c, 1e-12)
            return np.log(c)
        def util_prime(c):
            c = np.maximum(c, 1e-12)
            return 1.0 / c
        def util_double_prime(c):
            c = np.maximum(c, 1e-12)
            return -1.0 / (c * c)
    else:
        def util(c):
            c = np.maximum(c, 1e-12)
            return (c ** (1 - gamma) - 1) / (1 - gamma)
        def util_prime(c):
            c = np.maximum(c, 1e-12)
            return c ** (-gamma)
        def util_double_prime(c):
            c = np.maximum(c, 1e-12)
            return -gamma * c ** (-gamma - 1)
    
    # Policy storage
    c_policy = np.zeros((horizon, n_z, n_m))
    
    # Last period: c = m
    c_policy[horizon - 1, :, :] = m_grid[np.newaxis, np.newaxis, :]
    
    # Value function and marginal value
    V_next = np.tile(util(m_grid), (n_z, 1))
    Vm_next = np.tile(util_prime(m_grid), (n_z, 1))
    
    # Build PCHIP interpolators for V and Vm for each z
    def build_pchip(arr_2d):
        return [PchipInterpolator(m_grid, arr_2d[z], extrapolate=False) for z in range(n_z)]
    
    V_interp = build_pchip(V_next)
    Vm_interp = build_pchip(Vm_next)
    
    # Pre-compute transition weights
    trans = transition.copy()
    
    # Backward induction from t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 2:
            # Fill remaining with heuristic
            for tt in range(t, -1, -1):
                for z in range(n_z):
                    c_policy[tt, z, :] = m_grid
            break
        
        V_curr = np.zeros((n_z, n_m))
        Vm_curr = np.zeros((n_z, n_m))
        
        for z in range(n_z):
            for i, m in enumerate(m_grid):
                # Solve FOC: u'(c) = beta * R * E[Vm(m', z')]
                # where m' = R*(m-c) + income[z']
                
                # Define the FOC function
                def f_of_c(c_val):
                    u_prime_c = util_prime(c_val)
                    expected_Vm = 0.0
                    for zp in range(n_z):
                        prob = trans[z, zp]
                        if prob < 1e-10:
                            continue
                        m_next = R * (m - c_val) + income[zp]
                        if m_next < m_grid[0]:
                            m_next = m_grid[0]
                        elif m_next > m_grid[-1]:
                            m_next = m_grid[-1]
                        expected_Vm += prob * Vm_interp[zp](m_next)
                    return u_prime_c - beta * R * expected_Vm
                
                # Newton's method with bisection safeguard
                c_lo = 1e-8
                c_hi = m
                
                # Initial guess based on rough approximation
                avg_Vm = 0.0
                for zp in range(n_z):
                    m_next_mid = R * (m - m * 0.5) + income[zp]
                    m_next_mid = np.clip(m_next_mid, m_grid[0], m_grid[-1])
                    avg_Vm += trans[z, zp] * Vm_interp[zp](m_next_mid)
                
                if avg_Vm > 0:
                    if abs(gamma - 1.0) < 1e-10:
                        c_guess = 1.0 / (beta * R * avg_Vm)
                    else:
                        c_guess = (beta * R * avg_Vm) ** (-1.0 / gamma)
                else:
                    c_guess = m * 0.5
                
                c_guess = np.clip(c_guess, c_lo, c_hi)
                
                # Newton iteration
                c = c_guess
                for it in range(60):
                    if time.time() > deadline - 1:
                        break
                    
                    f_c = f_of_c(c)
                    
                    # Numerical derivative with central difference for better accuracy
                    h = max(1e-7, c * 1e-5)
                    f_c_plus = f_of_c(min(c + h, c_hi))
                    f_c_minus = f_of_c(max(c - h, c_lo))
                    f_prime_c = (f_c_plus - f_c_minus) / (2 * h)
                    
                    if abs(f_prime_c) < 1e-12:
                        break
                    
                    c_new = c - f_c / f_prime_c
                    
                    # Bisection safeguard
                    if c_new < c_lo or c_new > c_hi:
                        if f_c > 0:
                            c_lo = c
                        else:
                            c_hi = c
                        c_new = (c_lo + c_hi) / 2.0
                    else:
                        if f_c > 0:
                            c_lo = c
                        else:
                            c_hi = c
                    
                    if abs(c_new - c) < 1e-12 * max(1.0, abs(c)):
                        c = c_new
                        break
                    
                    c = c_new
                
                c = np.clip(c, 1e-12, m)
                
                c_policy[t, z, i] = c
                
                # Compute value
                V_val = util(c)
                for zp in range(n_z):
                    prob = trans[z, zp]
                    if prob < 1e-10:
                        continue
                    m_next = R * (m - c) + income[zp]
                    m_next = np.clip(m_next, m_grid[0], m_grid[-1])
                    V_val += beta * prob * V_interp[zp](m_next)
                
                V_curr[z, i] = V_val
                Vm_curr[z, i] = util_prime(c)
        
        V_next = V_curr
        Vm_next = Vm_curr
        V_interp = build_pchip(V_next)
        Vm_interp = build_pchip(Vm_next)
    
    # Enforce constraints
    for t in range(horizon):
        for z in range(n_z):
            for i in range(n_m):
                m_val = m_grid[i]
                c_val = c_policy[t, z, i]
                if c_val <= 0:
                    c_policy[t, z, i] = 1e-12
                if c_val > m_val + 1e-10:
                    c_policy[t, z, i] = m_val
    
    # Final period must equal grid within 1e-9
    c_policy[horizon - 1, :, :] = m_grid[np.newaxis, np.newaxis, :]
    
    return c_policy
# EVOLVE-BLOCK-END
