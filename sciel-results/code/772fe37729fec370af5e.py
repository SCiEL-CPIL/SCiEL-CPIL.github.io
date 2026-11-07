"""Solve consumption-saving policy via backward induction with PCHIP interpolation and FOC-based policy computation."""
import numpy as np
from scipy.interpolate import PchipInterpolator
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'])
    transition = np.asarray(model['transition'])
    horizon = model['horizon']
    m_grid = np.asarray(resource_grid)
    N_m = len(m_grid)
    N_z = len(income)

    # CRRA utility and derivatives
    def util(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return c ** (1.0 - gamma) / (1.0 - gamma)

    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return 1.0 / c
        else:
            return c ** (-gamma)

    def u_double_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return -1.0 / (c * c)
        else:
            return -gamma * c ** (-gamma - 1.0)

    # Terminal condition: at t = horizon, consume everything
    V = np.zeros((N_z, N_m))
    for z in range(N_z):
        V[z, :] = util(m_grid)

    # Policy array to fill
    policy = np.zeros((horizon, N_z, N_m))

    # Backward induction from t = horizon-1 down to 0
    for t in range(horizon - 1, -1, -1):
        # Check deadline
        if time.time() - start_time > DEADLINE - 5.0:
            break

        # Build continuation value function for each z
        # V_next[z', m'] where m' = R*(m - c) + income[z']
        # We need to interpolate V_next over m'
        
        # First, compute the "effective" next-period value as a function of m' for each z'
        # V_next_interp[z'] is a function of m'
        
        # For each current z, the Bellman equation is:
        # V_t(z, m) = max_c { u(c) + beta * sum_{z'} transition[z, z'] * V_{t+1}(z', R*(m-c) + income[z']) }
        
        # We'll compute the optimal policy for each (z, m) using the FOC.
        # FOC: u'(c) = beta * sum_{z'} transition[z, z'] * R * u'(c') * V_{t+1}''(z', m')
        # But this is for the case where we can differentiate through. 
        # More robustly, we can solve the FOC directly.
        
        # Actually, let's use a different approach:
        # For each z and m, we want to find c that maximizes:
        # u(c) + beta * sum_{z'} transition[z, z'] * V_{t+1}(z', R*(m-c) + income[z'])
        
        # The derivative w.r.t. c is:
        # u'(c) - beta * R * sum_{z'} transition[z, z'] * V_{t+1}'(z', R*(m-c) + income[z'])
        
        # We need V_{t+1}' and V_{t+1}'' as functions of m'.
        
        # Let's precompute PCHIP interpolants for V_{t+1}(z', m') and its derivatives.
        
        # Build interpolants for next-period value
        V_interp = []
        V_prime_interp = []
        V_double_prime_interp = []
        
        for z in range(N_z):
            # V[z, m_grid]
            # Use PCHIP for smooth interpolation
            V_interp.append(PchipInterpolator(m_grid, V[z], extrapolate=False))
            
            # For derivatives, we can use the fact that V is smooth.
            # Numerical differentiation or use PCHIP's derivative method.
            # PchipInterpolator has a derivative() method.
            V_prime_interp.append(V_interp[-1].derivative())
            V_double_prime_interp.append(V_prime_interp[-1].derivative())
        
        # Now solve for optimal c at each (z, m)
        for z in range(N_z):
            for i_m in range(N_m):
                m = m_grid[i_m]
                
                # Define the objective function for c
                def objective(c_val):
                    if c_val <= 0 or c_val > m:
                        return -1e30
                    c_val = max(c_val, 1e-12)
                    total = util(c_val)
                    for zp in range(N_z):
                        m_next = R * (m - c_val) + income[zp]
                        if m_next < 0:
                            m_next = 0
                        # Interpolate V_{t+1}(zp, m_next)
                        if m_next >= m_grid[0] and m_next <= m_grid[-1]:
                            total += beta * transition[z, zp] * V_interp[zp](m_next)
                        else:
                            # Extrapolate: if m_next < m_grid[0], use V at m_grid[0]
                            # if m_next > m_grid[-1], use V at m_grid[-1]
                            if m_next < m_grid[0]:
                                total += beta * transition[z, zp] * V[zp, 0]
                            else:
                                total += beta * transition[z, zp] * V[zp, -1]
                    return total
                
                # Use FOC to find optimal c
                # u'(c) = beta * R * sum_{z'} transition[z, z'] * V_{t+1}'(z', R*(m-c) + income[z'])
                
                def foc_residual(c_val):
                    if c_val <= 0 or c_val > m:
                        return 1e10
                    c_val = max(c_val, 1e-12)
                    lhs = u_prime(c_val)
                    rhs = 0.0
                    for zp in range(N_z):
                        m_next = R * (m - c_val) + income[zp]
                        if m_next >= m_grid[0] and m_next <= m_grid[-1]:
                            rhs += beta * R * transition[z, zp] * V_prime_interp[zp](m_next)
                        else:
                            if m_next < m_grid[0]:
                                rhs += beta * R * transition[z, zp] * V_prime_interp[zp](m_grid[0])
                            else:
                                rhs += beta * R * transition[z, zp] * V_prime_interp[zp](m_grid[-1])
                    return lhs - rhs
                
                # Solve FOC using Newton with bisection safeguard
                # Initial guess: proportional saving rule
                c_lo = 1e-12
                c_hi = m
                
                # Bisection bounds
                def bisect_foc(lo, hi, tol=1e-10, max_iter=100):
                    for _ in range(max_iter):
                        mid = (lo + hi) / 2.0
                        r = foc_residual(mid)
                        if abs(r) < tol or (hi - lo) / 2.0 < tol:
                            return mid
                        if r > 0:
                            # u'(c) > beta*R*sum(...), so we should increase c
                            # Wait: if u'(c) > continuation marginal value, we should consume more
                            # Actually, the FOC is u'(c) - beta*R*sum(...) = 0
                            # If this is positive, u'(c) is too high, so c is too low? 
                            # No: u'(c) is decreasing in c. If u'(c) > beta*R*sum, we need to decrease u'(c), so increase c.
                            lo = mid
                        else:
                            hi = mid
                    return (lo + hi) / 2.0
                
                # Try Newton first
                c_curr = m * 0.5  # initial guess
                c_opt = None
                for newton_iter in range(50):
                    r = foc_residual(c_curr)
                    if abs(r) < 1e-10:
                        c_opt = c_curr
                        break
                    
                    # Numerical derivative of foc_residual
                    h = 1e-8
                    r_plus = foc_residual(c_curr + h)
                    r_minus = foc_residual(c_curr - h)
                    dr_dc = (r_plus - r_minus) / (2 * h)
                    
                    if abs(dr_dc) < 1e-15:
                        break
                    
                    c_new = c_curr - r / dr_dc
                    
                    # Safeguard: keep within bounds
                    if c_new <= c_lo or c_new >= c_hi:
                        c_new = (c_lo + c_hi) / 2.0
                    
                    # If Newton step is too large, use bisection
                    if c_new <= c_curr:
                        c_lo = c_curr
                    else:
                        c_hi = c_curr
                    
                    if abs(c_new - c_curr) < 1e-12:
                        c_opt = c_curr
                        break
                    
                    c_curr = c_new
                
                if c_opt is None:
                    c_opt = bisect_foc(c_lo, c_hi)
                
                # Clamp
                c_opt = np.clip(c_opt, 1e-12, m)
                policy[t, z, i_m] = c_opt
        
        # Update V for current period
        for z in range(N_z):
            for i_m in range(N_m):
                m = m_grid[i_m]
                c = policy[t, z, i_m]
                V[z, i_m] = objective(c)

    # Final period: consume everything
    for z in range(N_z):
        policy[horizon - 1, z, :] = m_grid

    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(N_z):
            for i_m in range(N_m):
                if policy[t, z, i_m] <= 0:
                    policy[t, z, i_m] = 1e-12
                if policy[t, z, i_m] > m_grid[i_m] + 1e-10:
                    policy[t, z, i_m] = m_grid[i_m]

    return policy
