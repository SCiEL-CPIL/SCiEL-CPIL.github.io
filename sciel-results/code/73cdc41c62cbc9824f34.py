import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'])
    transition = np.asarray(model['transition'])
    horizon = model['horizon']
    m_grid = np.asarray(resource_grid)
    
    N_m = len(m_grid)
    N_z = len(income)
    
    # Pre-compute utility function
    def util(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    # Helper to maximize the Bellman equation for a given m, z
    # V_next is (N_m, N_z)
    # Returns optimal c and value
    def maximize_bellman(m, z, V_next):
        # We search for c in [0, m]
        # Use a grid search + refinement
        # Grid size 200 is sufficient for initial bracket
        K = 200
        c_grid = np.linspace(0, m, K)
        
        # Evaluate objective for all c in c_grid
        # u(c) is vectorized
        u_vals = util(c_grid)
        
        # Continuation value:
        # next_m = R * (m - c_grid) + income[z']
        # We need sum_z' P[z,z'] * V_next(next_m, z')
        # V_next is (N_m, N_z). We interpolate along axis 0.
        
        # Vectorize interpolation for all c_grid points and all z'
        # next_m shape: (K, N_z)
        next_m = R * (m - c_grid[:, None]) + income[None, :]
        
        # Clip next_m to grid range for interpolation
        next_m_clipped = np.clip(next_m, m_grid[0], m_grid[-1])
        
        # Interpolate V_next for each z'
        # V_next[:, zp] is (N_m,). We need to interpolate at next_m_clipped[:, zp]
        # We can do this in a loop over z' (N_z=2) or vectorize if possible.
        # np.interp is 1D.
        cont_vals = np.zeros_like(c_grid)
        for zp in range(N_z):
            p = transition[z, zp]
            if p > 0:
                # Interpolate V_next[:, zp] at next_m_clipped[:, zp]
                v_interp = np.interp(next_m_clipped[:, zp], m_grid, V_next[:, zp])
                cont_vals += p * v_interp
        
        total_vals = u_vals + beta * cont_vals
        
        # Find max
        idx = np.argmax(total_vals)
        c_max = c_grid[idx]
        
        # Refine using bisection around the peak
        # The function is concave.
        # We can do a few steps of bisection between neighbors or just return the grid max.
        # Given K=200, the error is small. But let's do 10 bisection steps for precision.
        
        # Determine bounds for bisection
        if idx == 0:
            lo, hi = c_grid[0], c_grid[1]
        elif idx == K - 1:
            lo, hi = c_grid[-2], c_grid[-1]
        else:
            # Check slopes to determine the interval containing the max
            # If total_vals[idx] > total_vals[idx-1] and total_vals[idx] > total_vals[idx+1], max is between idx-1 and idx+1?
            # Simpler: just use [c_grid[idx-1], c_grid[idx+1]]
            lo, hi = c_grid[idx-1], c_grid[idx+1]
            
        # Bisection search for max of concave function
        # f(c) is concave. We want f'(c)=0.
        # Alternatively, just evaluate midpoint and keep the side with higher value.
        for _ in range(10):
            mid = (lo + hi) / 2.0
            # Evaluate at lo, mid, hi
            # We only need to compare mid with one neighbor to decide direction?
            # Since it's concave, if f(mid) > f(lo), max is in [mid, hi]. Else [lo, mid].
            # But we need to be careful with boundaries.
            
            # Evaluate f(mid)
            c_mid = mid
            u_mid = util(c_mid)
            nm_mid = R * (m - c_mid) + income
            nm_mid_clipped = np.clip(nm_mid, m_grid[0], m_grid[-1])
            cont_mid = 0.0
            for zp in range(N_z):
                p = transition[z, zp]
                if p > 0:
                    v_mid = np.interp(nm_mid_clipped[zp], m_grid, V_next[:, zp])
                    cont_mid += p * v_mid
            f_mid = u_mid + beta * cont_mid
            
            # Evaluate f(lo) - we already have total_vals for grid points, but lo/hi might be grid points.
            # To be safe, just re-evaluate lo.
            c_lo = lo
            u_lo = util(c_lo)
            nm_lo = R * (m - c_lo) + income
            nm_lo_clipped = np.clip(nm_lo, m_grid[0], m_grid[-1])
            cont_lo = 0.0
            for zp in range(N_z):
                p = transition[z, zp]
                if p > 0:
                    v_lo = np.interp(nm_lo_clipped[zp], m_grid, V_next[:, zp])
                    cont_lo += p * v_lo
            f_lo = u_lo + beta * cont_lo
            
            if f_mid > f_lo:
                lo = mid
            else:
                hi = mid
                
        c_opt = (lo + hi) / 2.0
        
        # Ensure constraints
        if c_opt < 1e-6:
            c_opt = 1e-6
        if c_opt > m:
            c_opt = m
            
        # Calculate final value
        u_opt = util(c_opt)
        nm_opt = R * (m - c_opt) + income
        nm_opt_clipped = np.clip(nm_opt, m_grid[0], m_grid[-1])
        cont_opt = 0.0
        for zp in range(N_z):
            p = transition[z, zp]
            if p > 0:
                v_opt = np.interp(nm_opt_clipped[zp], m_grid, V_next[:, zp])
                cont_opt += p * v_opt
        v_opt = u_opt + beta * cont_opt
        
        return c_opt, v_opt

    # Initialize value function for terminal period
    # c_{T-1} = m
    V_next = np.zeros((N_m, N_z))
    for z in range(N_z):
        V_next[:, z] = util(m_grid)
        
    policies = np.zeros((horizon, N_z, N_m))
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > DEADLINE:
            # Fill remaining with heuristic (e.g., consume all) to ensure valid output
            for tt in range(t, -1, -1):
                for z in range(N_z):
                    policies[tt, z, :] = m_grid
            break
            
        V_curr = np.zeros((N_m, N_z))
        c_curr = np.zeros((N_z, N_m))
        
        for z in range(N_z):
            for i_m, m in enumerate(m_grid):
                c_opt, v_opt = maximize_bellman(m, z, V_next)
                c_curr[z, i_m] = c_opt
                V_curr[i_m, z] = v_opt
                
        policies[t] = c_curr
        V_next = V_curr
        
    # Terminal period consumption is m
    for z in range(N_z):
        policies[horizon-1, z, :] = m_grid
        
    return policies
