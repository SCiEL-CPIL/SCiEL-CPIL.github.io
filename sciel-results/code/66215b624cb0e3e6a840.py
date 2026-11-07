import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'])
    transition = np.asarray(model['transition'])
    horizon = model['horizon']
    m_grid = np.asarray(resource_grid)
    N_m = len(m_grid)
    N_z = len(income)

    # Utility and derivatives
    def util(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return 1.0 / c
        else:
            return c ** (-gamma)

    def u_double_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return -1.0 / c ** 2
        else:
            return -gamma * c ** (-gamma - 1)

    # Consumption policy: C[t, z, m_idx]
    C = np.zeros((horizon, N_z, N_m))

    # Terminal condition: at t = horizon - 1 (last period), consume all
    C[horizon - 1, :, :] = m_grid[None, :]

    # Backward induction from t = horizon - 2 down to 0
    # V_next[z, m_idx] = continuation value
    # We need V_next and dV/dm_next for the Euler equation
    # V_next[z, m'] = util(c_next) + beta * E_z'[V_{t+1}(m'', z'')]
    # But we solve for c_t such that the Euler equation holds.

    # We'll store value functions and their derivatives on the grid
    # V[t, z, m_idx] and dV[t, z, m_idx]
    V = np.zeros((horizon, N_z, N_m))
    dV = np.zeros((horizon, N_z, N_m))

    # At the terminal period, the agent consumes everything, so V = util(m)
    V[horizon - 1, :, :] = util(m_grid)[None, :]
    dV[horizon - 1, :, :] = u_prime(m_grid)[None, :]

    for t in range(horizon - 2, -1, -1):
        if time.time() > DEADLINE:
            break

        # For each z and m_idx, solve for c using Newton on the Euler equation:
        # u'(c) = beta * R * E_{z'}[ u'(c_next) * dV_next(z', m') / dm' ]
        # where m' = R*(m - c) + income[z']
        # and c_next is the policy at t+1 for state (z', m')
        
        # We need to interpolate V_next and dV_next onto the implied m' values
        # V_next[z', m'] = V[t+1, z', m']
        # dV_next[z', m'] = dV[t+1, z', m']

        # Precompute for each z, the expected value and derivative of V_next
        # at m' = R*(m - c) + income[z']
        
        # For each z (current state), the Euler equation is:
        # u'(c) = beta * sum_{z'} P(z'|z) * u'(c_next(z', m')) * dV_next(z', m')
        # where m' = R*(m - c) + income[z']
        # and c_next(z', m') = C[t+1, z', m'] (interpolated)

        # Newton iteration: define f(c) = u'(c) - beta * E[u'(c_next) * dV_next]
        # f'(c) = u''(c) - beta * E[ d/dc (u'(c_next) * dV_next) ]
        # c_next = C[t+1, z', m'] where m' depends on c
        # dm'/dc = -R
        # dc_next/dc = dC[t+1](m')/dm' * (-R)  [if we had derivative of policy]
        # This gets complex. Let's use a simpler approach:
        
        # Alternative: Directly maximize the objective using Newton on the 
        # one-dimensional concave function for each (z, m_idx).
        
        # Objective for state (z, m):
        # g(c) = util(c) + beta * sum_{z'} P(z'|z) * V_next(z', R*(m-c) + income[z'])
        # g'(c) = u'(c) - beta * sum_{z'} P(z'|z) * dV_next(z', m') * R
        #        = u'(c) - beta * R * sum_{z'} P(z'|z) * dV_next(z', m')
        # g''(c) = u''(c) + beta * R^2 * sum_{z'} P(z'|z) * d2V_next(z', m')
        
        # We need d2V_next. Let's compute it.
        # Actually, for the Euler equation approach, we just need to solve g'(c) = 0.
        
        # Let's use bisection + Newton hybrid, fully vectorized over m_idx for each z.
        
        # Initialize c with a reasonable guess (e.g., 50% of m)
        c = 0.5 * m_grid[None, :]  # shape (N_z, N_m)
        
        # Newton iterations
        for newton_iter in range(30):
            if time.time() > DEADLINE:
                break
            
            # m' for each z, z', m_idx
            # m' = R * (m - c) + income[z']
            # c shape: (N_z, N_m)
            # m' shape: (N_z, N_z, N_m) where m'[z, z', m_idx]
            m_prime = R * (m_grid[None, None, :] - c[:, None, :]) + income[None, :, None]
            # Clip m' to be within the grid range for interpolation
            m_prime_clipped = np.clip(m_prime, m_grid[0], m_grid[-1])
            
            # Interpolate dV_next[z', m'] for each z, z', m_idx
            # dV_next shape: (N_z, N_m) -> dV[t+1, z', m_idx]
            # For each z', interpolate dV[t+1, z', :] at m_prime_clipped[z, z', :]
            # Result: dV_interp shape (N_z, N_z, N_m)
            dV_interp = np.zeros((N_z, N_z, N_m))
            for zp in range(N_z):
                # Interpolate dV[t+1, zp, :] at m_prime_clipped[:, zp, :]
                # m_prime_clipped[:, zp, :] has shape (N_z, N_m)
                # np.interp works on 1D, so we need to do this per z or use a vectorized approach
                # Let's use np.interp with the grid
                for zi in range(N_z):
                    dV_interp[zi, zp, :] = np.interp(m_prime_clipped[zi, zp, :], m_grid, dV[t+1, zp, :])
            
            # Expected dV_next: E_{z'}[dV_next(z', m')] = sum_{z'} P(z'|z) * dV_interp[z, z', :]
            # transition[z, z'] = P(z'|z)
            exp_dV = np.zeros((N_z, N_m))
            for zi in range(N_z):
                exp_dV[zi, :] = transition[zi, :] @ dV_interp[zi, :, :]  # (N_z,) @ (N_z, N_m) -> (N_m,)
            
            # g'(c) = u'(c) - beta * R * exp_dV
            c_clipped = np.maximum(c, 1e-12)
            g_prime = u_prime(c_clipped) - beta * R * exp_dV
            
            # Check convergence
            if np.max(np.abs(g_prime)) < 1e-10:
                break
            
            # Newton step: need g''(c)
            # g''(c) = u''(c) + beta * R^2 * E[d2V_next(z', m')]
            # We need d2V_next. Let's approximate or compute it.
            # For now, use a damped Newton / bisection hybrid.
            # Since the problem is concave, g'(c) is decreasing in c.
            # If g'(c) > 0, increase c; if g'(c) < 0, decrease c.
            
            # Use a simple update: c_new = c + g_prime / |g_prime| * step
            # Better: use bisection bounds.
            
            # Let's use a different approach: for each (z, m_idx), do a few Newton steps
            # with a fallback to bisection.
            
            # Actually, let's just use the fact that g'(c) is monotone and do a 
            # vectorized bisection + Newton combination.
            
            # For Newton, we need g''. Let's approximate d2V_next by finite differences
            # on the grid, or just use a safe step size.
            
            # Simpler: use the fact that for CRRA, we can derive a closed-form approximation.
            # But let's stick with numerical.
            
            # Let's do a damped Newton: c_new = c - g_prime / g_second
            # Approximate g_second as u''(c) (ignoring the second derivative of V, which is smaller)
            c_clipped2 = np.maximum(c, 1e-12)
            g_second = u_double_prime(c_clipped2)  # negative
            # Ensure g_second is not zero
            g_second = np.where(np.abs(g_second) < 1e-15, -1e-15, g_second)
            
            c_new = c - g_prime / g_second
            
            # Clamp c_new to valid range: 1e-12 <= c_new <= m
            c_new = np.clip(c_new, 1e-12, m_grid[None, :])
            
            # Damping: if the step is too large, reduce it
            # Check if g_prime decreased in magnitude
            # For simplicity, just update
            c = c_new
        
        C[t, :, :] = c
    
    # Final validity checks
    for t in range(horizon):
        for z in range(N_z):
            C[t, z, :] = np.clip(C[t, z, :], 1e-12, m_grid + 1e-10)
    
    # Terminal condition: c[5] must equal the grid
    C[horizon - 1, :, :] = m_grid[None, :]
    
    # Ensure finite values
    C = np.where(np.isfinite(C), C, 1e-12)
    C = np.maximum(C, 1e-12)
    
    return C
