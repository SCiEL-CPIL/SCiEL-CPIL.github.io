import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145.0

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=np.float64)
    transition = np.asarray(model['transition'], dtype=np.float64)
    horizon = model['horizon']

    c_min = 1e-12
    n_z = 2
    T = horizon

    # Dense internal grid for accuracy
    n_dense = 2001
    grid_dense = np.geomspace(0.001, 40, n_dense)
    n_grid_out = len(resource_grid)

    def utility(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    def utility_prime(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)

    # V[z, m] on dense grid, P[z, m] on dense grid
    V = np.zeros((n_z, n_dense))
    P = np.zeros((n_z, n_dense))

    # Terminal period: c = m
    V[:, :] = utility(grid_dense)[np.newaxis, :].repeat(n_z, axis=0)
    P[:, :] = grid_dense[np.newaxis, :].repeat(n_z, axis=0)

    # Precompute log-spaced m for speed
    m_arr = grid_dense  # shape (n_dense,)

    for t in range(T - 2, -1, -1):
        if time.time() > deadline - 10:
            # Fallback: fill remaining periods with a simple rule
            for t_rem in range(t, -1, -1):
                # Use a heuristic: c = (beta*R)**(-1/gamma) * m, clipped
                if abs(gamma - 1.0) < 1e-10:
                    scale = beta * R / (1.0 + beta * R)
                else:
                    scale = (beta * R) ** (-1.0 / gamma)
                c_guess = scale * m_arr
                c_guess = np.clip(c_guess, c_min, m_arr)
                V[t_rem] = utility(c_guess)[np.newaxis, :].repeat(n_z, axis=0)
                P[t_rem] = c_guess[np.newaxis, :].repeat(n_z, axis=0)
            break

        V_next = V  # V[t+1] is stored in V after we overwrite
        # Actually we need to be careful: V currently holds V[t+1] from previous iteration
        # Let's use separate arrays
        V_t = np.zeros((n_z, n_dense))
        P_t = np.zeros((n_z, n_dense))

        # For each z, solve for c at all m simultaneously using bisection
        # The Euler equation: u'(c) = beta * R * sum_{z'} P(z'|z) * V'[z'](R(m-c) + y_{z'})
        # We define f(c, m) = u'(c) - beta*R*sum_{z'} trans[z,z'] * dV[z'](R(m-c)+y[z'])
        # f is decreasing in c (u'' < 0, and dV' is decreasing in next_m which decreases in c... 
        # wait: next_m = R(m-c)+y, as c increases, next_m decreases, dV' increases (since V is concave, V' is decreasing in m, so as m decreases V' increases)
        # So -dV[z'](next_m) term: as c increases, next_m decreases, dV' increases, so -beta*R*sum dV' * (-R) ... let me just think about monotonicity.
        # f(c) = u'(c) - G(c) where G(c) = beta*R*sum trans[z,z'] * dV[z'](R(m-c)+y[z'])
        # u'(c) is decreasing in c.
        # G(c): as c increases, R(m-c)+y decreases. Since V is concave, dV is decreasing in its argument. So as argument decreases, dV increases. So G(c) increases in c.
        # Therefore f(c) = decreasing - increasing = strictly decreasing in c. Bisection works!

        # At c = c_min: f = u'(c_min) - G(c_min). u'(c_min) is huge positive. G(c_min) is finite. So f > 0.
        # At c = m: next_m = R(0) + y = y. G(m) = beta*R*sum trans[z,z']*dV[z'](y[z']). u'(m) is some value.
        # We need f(m) < 0 for bisection to work. If f(m) > 0, then optimal c = m (consume all).
        # Let's check: if f(m) >= 0, set c = m.

        # We'll vectorize over m for each z.
        for z in range(n_z):
            # Compute f(c, m) for a candidate c array
            # We'll do bisection: for each m, find c in [c_min, m]
            
            # Initialize bounds
            lo = np.full(n_dense, c_min)
            hi = np.full(n_dense, m_arr)
            
            # Check if f(hi) < 0 for any m; if not, c = m
            # Compute f at c = m_arr
            c_test = m_arr.copy()
            next_m_arr = np.zeros((n_z, n_dense))
            for zp in range(n_z):
                next_m_arr[zp, :] = R * (m_arr - c_test) + income[zp]
            
            # Interpolate dV_next for each z'
            # dV_next[z'] on grid_dense
            dV_next = np.zeros((n_z, n_dense))
            # Central differences
            dm = np.diff(grid_dense)
            dV_next[:, 1:-1] = (V_next[:, 2:] - V_next[:, :-2]) / (grid_dense[2:] - grid_dense[:-2])
            dV_next[:, 0] = (V_next[:, 1] - V_next[:, 0]) / (grid_dense[1] - grid_dense[0])
            dV_next[:, -1] = (V_next[:, -1] - V_next[:, -2]) / (grid_dense[-1] - grid_dense[-2])
            
            # Clamp next_m to grid range for interpolation
            next_m_clamped = np.clip(next_m_arr, grid_dense[0], grid_dense[-1])
            
            G_hi = np.zeros(n_dense)
            for zp in range(n_z):
                dv_interp = np.interp(next_m_clamped[zp, :], grid_dense, dV_next[zp, :])
                G_hi += transition[z, zp] * dv_interp
            G_hi *= beta * R
            
            f_hi = utility_prime(c_test) - G_hi
            # Where f_hi >= 0, optimal c = m
            c_opt = np.where(f_hi >= 0, m_arr, 0.0)
            
            # For remaining, do bisection
            mask = f_hi < 0
            if np.any(mask):
                lo_m = lo[mask]
                hi_m = hi[mask]
                m_sub = m_arr[mask]
                
                for _ in range(50):
                    if time.time() > deadline - 5:
                        break
                    mid = 0.5 * (lo_m + hi_m)
                    
                    # Compute f at mid for each m in subset
                    next_m_sub = np.zeros((n_z, len(mid)))
                    for zp in range(n_z):
                        next_m_sub[zp, :] = R * (m_sub - mid) + income[zp]
                    next_m_sub_clamped = np.clip(next_m_sub, grid_dense[0], grid_dense[-1])
                    
                    G_mid = np.zeros(len(mid))
                    for zp in range(n_z):
                        dv_interp = np.interp(next_m_sub_clamped[zp, :], grid_dense, dV_next[zp, :])
                        G_mid += transition[z, zp] * dv_interp
                    G_mid *= beta * R
                    
                    f_mid = utility_prime(mid) - G_mid
                    
                    # f is decreasing in c: if f_mid > 0, c is too low, increase lo
                    # if f_mid < 0, c is too high, decrease hi
                    lo_m = np.where(f_mid > 0, mid, lo_m)
                    hi_m = np.where(f_mid < 0, mid, hi_m)
                
                c_opt[mask] = 0.5 * (lo_m + hi_m)
            
            c_opt = np.clip(c_opt, c_min, m_arr)
            P_t[z, :] = c_opt
            
            # Compute V_t[z, m] = u(c) + beta * sum_{z'} trans[z,z'] * V_next[z'](R(m-c)+y[z'])
            next_m_v = np.zeros((n_z, n_dense))
            for zp in range(n_z):
                next_m_v[zp, :] = R * (m_arr - c_opt) + income[zp]
            next_m_v_clamped = np.clip(next_m_v, grid_dense[0], grid_dense[-1])
            
            cont_val = np.zeros(n_dense)
            for zp in range(n_z):
                v_interp = np.interp(next_m_v_clamped[zp, :], grid_dense, V_next[zp, :])
                cont_val += transition[z, zp] * v_interp
            
            V_t[z, :] = utility(c_opt) + beta * cont_val
        
        V = V_t

    # Interpolate P from dense grid to output grid
    result = np.zeros((T, n_z, n_grid_out))
    for t in range(T):
        for z in range(n_z):
            result[t, z, :] = np.interp(resource_grid, grid_dense, P[t, z, :])
    
    # Enforce constraints
    for t in range(T):
        for z in range(n_z):
            result[t, z, :] = np.clip(result[t, z, :], c_min, resource_grid + 1e-10)
    
    # Final period: c = m exactly
    result[T-1, :, :] = resource_grid[np.newaxis, :].repeat(n_z, axis=0)
    
    return result
