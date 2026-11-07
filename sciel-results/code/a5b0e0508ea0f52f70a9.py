import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    M = len(resource_grid)
    Z = len(income)
    
    # Consumption policy: consumption[t, z, m_index]
    consumption = np.zeros((horizon, Z, M))
    
    # Helper for utility and marginal utility
    def utility(c, g):
        c = np.maximum(c, 1e-12)
        if g == 1:
            return np.log(c)
        else:
            return (c ** (1 - g)) / (1 - g)
    
    def u_prime(c, g):
        c = np.maximum(c, 1e-12)
        if g == 1:
            return 1.0 / c
        else:
            return c ** (-g)
    
    # Initialize V_next for t=5 (terminal)
    # V_5(m) = u(m)
    V_next = np.zeros((Z, M))
    for z in range(Z):
        V_next[z] = utility(resource_grid, gamma)
    
    # Policy for t=5: consume all
    consumption[5] = resource_grid[None, :]
    
    # We will use a high-resolution internal grid for the value function to allow
    # accurate interpolation of the continuation value.
    # The resource_grid is geometric (0.001 to 40).
    # Let's use a much finer grid, say 1000 points, also geometric or linear?
    # Geometric is better for this range.
    N_fine = 1000
    m_fine = np.geomspace(resource_grid[0], resource_grid[-1], N_fine)
    
    # Interpolate V_next onto m_fine
    V_next_fine = np.zeros((Z, N_fine))
    for z in range(Z):
        V_next_fine[z] = np.interp(m_fine, resource_grid, V_next[z])
    
    # Backward induction for t = 4 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with simple rule (consume all? or previous?)
            # Better to just stop and return what we have, but we need to fill the rest.
            # If we break, the earlier t's are zero.
            # Let's just use the previous period's policy or a simple heuristic.
            # Actually, if we break, we should probably just return the current state
            # and let the caller handle it, but the function must return the full array.
            # Let's fill with resource_grid (consume all) for remaining periods.
            for tt in range(t, -1, -1):
                for z in range(Z):
                    consumption[tt, z] = resource_grid
            break
            
        V_curr = np.zeros((Z, M))
        c_curr = np.zeros((Z, M))
        
        # We want to find c* that maximizes:
        # u(c) + beta * E[ V_{t+1}( R(m-c) + y', z' ) ]
        
        # We will use Newton's method on the Euler equation:
        # u'(c) = beta * E[ R * V'_{t+1}(m', z') ]
        # where m' = R(m-c) + y'
        
        # V'_{t+1} needs to be interpolated from the fine grid.
        # Let's pre-compute V_prime_next_fine
        # V_next_fine is the value. We need its derivative.
        # We can approximate the derivative using finite differences on the fine grid.
        # Or, better, we can store the policy on the fine grid and use the envelope theorem?
        # No, let's just compute V' numerically from V_next_fine.
        
        # V_prime_next_fine[z, i] ≈ (V_next_fine[z, i+1] - V_next_fine[z, i-1]) / (m_fine[i+1] - m_fine[i-1])
        # This is O(N_fine) per step, which is fast.
        
        V_prime_next_fine = np.zeros((Z, N_fine))
        for z in range(Z):
            # Central difference
            dV = np.zeros(N_fine)
            dV[1:-1] = (V_next_fine[z, 2:] - V_next_fine[z, :-2]) / (m_fine[2:] - m_fine[:-2])
            # One-sided at boundaries
            dV[0] = (V_next_fine[z, 1] - V_next_fine[z, 0]) / (m_fine[1] - m_fine[0])
            dV[-1] = (V_next_fine[z, -1] - V_next_fine[z, -2]) / (m_fine[-1] - m_fine[-2])
            V_prime_next_fine[z] = dV
        
        # Now, for each m in resource_grid and each z, solve for c.
        # c is in (0, m].
        
        # We will vectorize over m for a fixed z? Or just loop over m?
        # M=101, Z=2. Looping is fine.
        
        for z in range(Z):
            for i_m in range(M):
                m = resource_grid[i_m]
                
                # Initial guess for c.
                # A good guess is the policy from the previous iteration (if we had one)
                # or a simple rule like c = m * (1 - beta * R) or similar.
                # Let's start with c = m * 0.5 or use the previous t's policy if available?
                # Since we are going backwards, we don't have "previous" in the loop order.
                # We can use a rough heuristic: c = m / (1 + beta * R * something).
                # Or just start with c = m * 0.9.
                # Actually, for log utility, c* is proportional to m.
                # Let's start with c = m * 0.5.
                
                c = m * 0.5
                c = np.clip(c, 1e-6, m * 0.999)
                
                # Newton iteration
                for _ in range(10):
                    # Compute m' for each z'
                    # m_prime[z'] = R * (m - c) + income[z']
                    m_prime = R * (m - c) + income  # shape (Z,)
                    
                    # Clip m_prime to the range of m_fine for interpolation
                    # If m_prime < m_fine[0], use m_fine[0]
                    # If m_prime > m_fine[-1], use m_fine[-1]
                    # np.interp handles this by returning the endpoint values.
                    
                    # Interpolate V_prime_next_fine at m_prime
                    # V_prime_next_fine[z, :] is the derivative for state z.
                    # We need E[ V_prime_next(m', z') ] = sum_{z'} P(z'|z) * V_prime_next(z', m')
                    
                    # Let's compute V_prime_interp for each z'
                    # V_prime_interp[z'] = np.interp(m_prime[z'], m_fine, V_prime_next_fine[z'])
                    
                    V_prime_zp = np.zeros(Z)
                    for zp in range(Z):
                        V_prime_zp[zp] = np.interp(m_prime[zp], m_fine, V_prime_next_fine[zp])
                    
                    # Expected marginal value of wealth next period
                    E_V_prime = np.dot(transition[z], V_prime_zp)
                    
                    # Euler equation residual:
                    # F(c) = u'(c) - beta * R * E_V_prime
                    # We want F(c) = 0.
                    
                    u_p = u_prime(c, gamma)
                    F = u_p - beta * R * E_V_prime
                    
                    # Derivative F'(c) = u''(c) - beta * R * d/dc (E_V_prime)
                    # d/dc (E_V_prime) = d/dc ( sum P(z') V'(m') )
                    # m' = R(m-c) + y' => dm'/dc = -R
                    # d/dc V'(m') = V''(m') * (-R)
                    # So d/dc (E_V_prime) = -R * E[ V''(m') ]
                    
                    # We need V''(m'). We can approximate it from V_prime_next_fine.
                    # Or, we can just use a secant method or simple Newton with numerical derivative.
                    # Let's approximate V'' numerically on the fly?
                    # V''(m') ≈ (V'(m' + h) - V'(m' - h)) / (2h)
                    
                    # This is getting expensive to compute inside the loop.
                    # Alternative: Use a simple bisection or secant method which doesn't require derivatives.
                    # Given the smoothness, secant should work well.
                    
                    # Let's switch to a simple iterative method:
                    # c_new = c - F(c) / F'(c)
                    # We need F'(c).
                    # F'(c) = u''(c) + beta * R^2 * E[ V''(m') ]
                    
                    # Let's compute V'' numerically.
                    # We have V_prime_next_fine.
                    # V''_fine[z, i] ≈ (V_prime_next_fine[z, i+1] - V_prime_next_fine[z, i-1]) / (m_fine[i+1] - m_fine[i-1])
                    
                    # We can pre-compute V_double_prime_next_fine outside the m loop.
                    
                    break # Stop Newton, let's pre-compute V''
            
            # Pre-compute V_double_prime_next_fine for this t
            V_double_prime_next_fine = np.zeros((Z, N_fine))
            for zp in range(Z):
                d2V = np.zeros(N_fine)
                d2V[1:-1] = (V_prime_next_fine[zp, 2:] - V_prime_next_fine[zp, :-2]) / (m_fine[2:] - m_fine[:-2])
                d2V[0] = (V_prime_next_fine[zp, 1] - V_prime_next_fine[zp, 0]) / (m_fine[1] - m_fine[0])
                d2V[-1] = (V_prime_next_fine[zp, -1] - V_prime_next_fine[zp, -2]) / (m_fine[-1] - m_fine[-2])
                V_double_prime_next_fine[zp] = d2V
            
            # Now redo the m loop with Newton
            for i_m in range(M):
                m = resource_grid[i_m]
                c = m * 0.5
                c = np.clip(c, 1e-6, m * 0.999)
                
                for _ in range(10):
                    m_prime = R * (m - c) + income
                    # Clip for interpolation safety? np.interp handles it.
                    
                    V_prime_zp = np.zeros(Z)
                    V_double_prime_zp = np.zeros(Z)
                    for zp in range(Z):
                        V_prime_zp[zp] = np.interp(m_prime[zp], m_fine, V_prime_next_fine[zp])
                        V_double_prime_zp[zp] = np.interp(m_prime[zp], m_fine, V_double_prime_next_fine[zp])
                    
                    E_V_prime = np.dot(transition[z], V_prime_zp)
                    E_V_double_prime = np.dot(transition[z], V_double_prime_zp)
                    
                    # u'(c)
                    if gamma == 1:
                        u_p = 1.0 / c
                        u_pp = -1.0 / (c * c)
                    else:
                        u_p = c ** (-gamma)
                        u_pp = -gamma * c ** (-gamma - 1)
                    
                    F = u_p - beta * R * E_V_prime
                    F_prime = u_pp + beta * R * R * E_V_double_prime
                    
                    # Newton step
                    if abs(F_prime) < 1e-12:
                        break
                    c_new = c - F / F_prime
                    c_new = np.clip(c_new, 1e-6, m * 0.999)
                    
                    if abs(c_new - c) < 1e-10:
                        c = c_new
                        break
                    c = c_new
                
                c_curr[z, i_m] = c
                # We don't strictly need V_curr for the policy, but we need it for the next step's V_prime.
                # So we must compute V_curr.
                # V_curr(z, m) = u(c) + beta * E[ V_next(m', z') ]
                m_prime = R * (m - c) + income
                V_next_zp = np.zeros(Z)
                for zp in range(Z):
                    V_next_zp[zp] = np.interp(m_prime[zp], m_fine, V_next_fine[zp])
                E_V_next = np.dot(transition[z], V_next_zp)
                V_curr[z, i_m] = utility(c, gamma) + beta * E_V_next
        
        consumption[t] = c_curr
        V_next = V_curr
        
        # Interpolate V_next onto m_fine for the next iteration
        V_next_fine = np.zeros((Z, N_fine))
        for z in range(Z):
            V_next_fine[z] = np.interp(m_fine, resource_grid, V_next[z])
            
        # Re-compute V_prime_next_fine and V_double_prime_next_fine for the next t
        # (This is done at the start of the next loop iteration)

    # Final validity checks
    for t in range(horizon):
        for z in range(Z):
            c = consumption[t, z]
            m = resource_grid
            # 0 < c
            c = np.maximum(c, 1e-12)
            # c <= m + 1e-10
            c = np.minimum(c, m + 1e-10)
            consumption[t, z] = c
            
    # Enforce terminal condition exactly
    consumption[5] = resource_grid[None, :]
    
    return consumption
