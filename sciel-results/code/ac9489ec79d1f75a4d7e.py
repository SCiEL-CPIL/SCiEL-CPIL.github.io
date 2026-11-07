import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    c_min = 1e-12
    T = horizon
    n_z = 2
    grid_out = resource_grid
    n_out = len(grid_out)
    
    # Use a moderately dense internal grid
    n_inner = 2000
    grid = np.geomspace(0.001, 40, n_inner)
    n_grid = len(grid)
    
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
            
    def utility_double_prime(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return -1.0 / (c ** 2)
        else:
            return -gamma * c ** (-gamma - 1)
    
    # Backward induction with Newton's method on the Eulera equation
    # c'(m) = u'(c) = beta * E[ u'(c') * R * V''(m') ] ... 
    # Actually the Euler equation is: u'(c) = beta * E[ R * u'(c') | m, z ]
    # where c' is the next period consumption chosen optimally given m'
    
    # We solve for c(m, z) such that:
    # u'(c) = beta * sum_{z'} trans[z, z'] * R * u'(c*(m', z'))
    # where m' = R*(m - c) + income[z']
    # and c*(m', z') is the optimal consumption next period
    
    # This is a fixed-point / Newton problem. We use the value function approach:
    # V_t(m, z) = max_c { u(c) + beta * sum_{z'} trans[z,z'] * V_{t+1}(R*(m-c)+income[z'], z') }
    # FOC: u'(c) = beta * sum_{z'} trans[z,z'] * R * V'_{t+1}(m', z')
    
    # We represent V on the grid and use interpolation for derivatives.
    
    V = [None] * T
    P = [None] * T
    
    # Terminal condition: consume all wealth
    V[T-1] = utility(grid)[np.newaxis, :].repeat(n_z, axis=0)
    P[T-1] = grid[np.newaxis, :].repeat(n_z, axis=0)
    
    # Precompute grid spacing for finite differences
    # Use a smoother approach: cubic spline or just careful finite differences
    
    for t in range(T-2, -1, -1):
        if time.time() > deadline - 5:
            # Fallback: use previous period's policy
            for t_rem in range(t, -1, -1):
                V[t_rem] = V[t+1].copy()
                P[t_rem] = P[t+1].copy()
            break
            
        V_t = np.zeros((n_z, n_grid))
        P_t = np.zeros((n_z, n_grid))
        V_next = V[t+1]
        
        # Compute derivatives of V_next using finite differences
        # For geomspace grid, use appropriate stencils
        dV_next = np.zeros((n_z, n_grid))
        d2V_next = np.zeros((n_z, n_grid))
        
        for zp in range(n_z):
            Vz = V_next[zp]
            # Central differences for interior
            for i in range(1, n_grid - 1):
                h1 = grid[i] - grid[i-1]
                h2 = grid[i+1] - grid[i]
                # First derivative using 3-point formula
                dV_next[zp, i] = (-h2*Vz[i-1] + (h2-h1)*Vz[i] + h1*Vz[i+1]) / (h1*h2)
                # Second derivative
                d2V_next[zp, i] = 2.0 * ((Vz[i+1]-Vz[i])/h2 - (Vz[i]-Vz[i-1])/h1) / (h1+h2)
            # Boundary points: one-sided differences
            h0 = grid[1] - grid[0]
            h1 = grid[2] - grid[1]
            dV_next[zp, 0] = (-2*h1 - h0)*Vz[0] + (h1 + 2*h0)*Vz[1] - h0*Vz[2] / (h0*(h0+h1))
            d2V_next[zp, 0] = 2.0 * ((Vz[1]-Vz[0])/h0 - (Vz[2]-Vz[1])/h1) / (h0+h1)
            
            h1 = grid[-1] - grid[-2]
            h2 = grid[-2] - grid[-3]
            dV_next[zp, -1] = (2*h1 + h2)*Vz[-1] - (h1 + 2*h2)*Vz[-2] + h2*Vz[-3] / (h1*(h1+h2))
            d2V_next[zp, -1] = 2.0 * ((Vz[-1]-Vz[-2])/h1 - (Vz[-2]-Vz[-3])/h2) / (h1+h2)
        
        for z in range(n_z):
            # Initialize c with previous period's policy or a heuristic
            if t + 1 < T:
                c = P[t+1][z].copy()
            else:
                c = 0.5 * grid.copy()
            c = np.clip(c, c_min, grid)
            
            # Newton's method: solve u'(c) = beta * sum_{z'} trans[z,z'] * R * V'_{t+1}(m', z')
            # where m' = R*(m - c) + income[z']
            for _ in range(50):
                residual = np.zeros(n_grid)
                jacobian = np.zeros(n_grid)
                
                for zp in range(n_z):
                    m_prime = R * (grid - c) + income[zp]
                    m_prime_clipped = np.clip(m_prime, grid[0], grid[-1])
                    
                    # Interpolate V' at m_prime
                    dV_interp = np.interp(m_prime_clipped, grid, dV_next[zp])
                    
                    # The FOC: u'(c) - beta * trans[z,zp] * R * dV(m') = 0
                    residual += transition[z, zp] * R * dV_interp
                    # Derivative w.r.t c: -beta * trans[z,zp] * R * d2V(m') * R (chain rule: dm'/dc = -R)
                    d2V_interp = np.interp(m_prime_clipped, grid, d2V_next[zp])
                    jacobian += transition[z, zp] * R * R * d2V_interp
                
                residual *= beta
                jacobian *= beta
                
                # f(c) = u'(c) - beta * sum ... 
                # f'(c) = u''(c) + beta * sum ... (note: d/dc of -beta*sum R*dV(m') = -beta*sum R*d2V(m')*(-R) = beta*sum R^2*d2V(m'))
                # Wait, let me re-derive:
                # FOC: u'(c) = beta * sum_{z'} trans[z,z'] * R * V'(m')
                # where m' = R(m-c) + income[z']
                # d/dc of RHS: beta * sum trans[z,z'] * R * V''(m') * dm'/dc = beta * sum trans[z,z'] * R * V''(m') * (-R)
                # = -beta * sum trans[z,z'] * R^2 * V''(m')
                # So f(c) = u'(c) - beta * sum trans[z,z'] * R * V'(m')
                # f'(c) = u''(c) - (-beta * sum trans[z,z'] * R^2 * V''(m'))
                #       = u''(c) + beta * sum trans[z,z'] * R^2 * V''(m')
                
                f_val = utility_prime(c) - residual
                f_prime = utility_double_prime(c) + jacobian
                
                # Newton step
                mask = np.abs(f_prime) > 1e-14
                step = np.zeros(n_grid)
                step[mask] = f_val[mask] / f_prime[mask]
                c_new = c - step
                c_new = np.clip(c_new, c_min, grid)
                
                if np.max(np.abs(c_new - c)) < 1e-12:
                    c = c_new
                    break
                c = c_new
            
            P_t[z, :] = c
            
            # Compute value function
            cont_val = np.zeros(n_grid)
            for zp in range(n_z):
                m_prime = R * (grid - c) + income[zp]
                m_prime_clipped = np.clip(m_prime, grid[0], grid[-1])
                cont_val += transition[z, zp] * np.interp(m_prime_clipped, grid, V_next[zp])
            V_t[z, :] = utility(c) + beta * cont_val
        
        V[t] = V_t
        P[t] = P_t
    
    # Interpolate policy to the output grid
    result = np.zeros((T, n_z, n_out))
    for t in range(T):
        for z in range(n_z):
            result[t, z, :] = np.interp(grid_out, grid, P[t][z])
    
    # Enforce constraints
    for t in range(T):
        for z in range(n_z):
            result[t, z, :] = np.clip(result[t, z, :], c_min, grid_out + 1e-10)
    
    # Terminal condition must be exact
    result[T-1, :, :] = grid_out[np.newaxis, :]
    
    return result
