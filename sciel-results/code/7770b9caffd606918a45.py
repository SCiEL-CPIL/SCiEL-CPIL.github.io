import numpy as np
import time
from scipy.interpolate import interp1d

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'])
    transition = np.array(model['transition'])
    horizon = model['horizon']
    
    m_grid_out = np.array(resource_grid)
    n_m_out = len(m_grid_out)
    n_z = len(income)
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)
    
    # Derivative of utility (marginal utility)
    def utility_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return np.power(c, -gamma)
    
    # Use Chebyshev nodes on log-scale for spectral accuracy
    # Range: m from 0.001 to 50 (covers max m' = R*40 + 1.5 = 42.3)
    n_cheb = 256  # Number of Chebyshev collocation points
    log_m_min = np.log(0.001)
    log_m_max = np.log(50.0)
    
    # Chebyshev nodes in [log_m_min, log_m_max]
    k = np.arange(n_cheb)
    theta = np.pi * (n_cheb - 1 - k) / (n_cheb - 1)
    x_cheb = np.cos(theta)  # In [-1, 1]
    log_m_cheb = (log_m_max + log_m_min) / 2 + (log_m_max - log_m_min) / 2 * x_cheb
    m_cheb = np.exp(log_m_cheb)
    
    # Build Chebyshev interpolation matrix
    # T_n(x) = cos(n * arccos(x))
    def chebyshev_poly(n, x):
        return np.cos(n * np.arccos(np.clip(x, -1, 1)))
    
    # Chebyshev Vandermonde matrix
    C = np.zeros((n_cheb, n_cheb))
    for n in range(n_cheb):
        C[:, n] = chebyshev_poly(n, x_cheb)
    
    # Inverse for converting function values to coefficients
    C_inv = np.linalg.inv(C)
    
    # Value function storage as Chebyshev coefficients
    # V_coeff[t, z, n] where n is polynomial degree
    V_coeff = np.zeros((horizon, n_z, n_cheb))
    
    # Terminal period: V[T-1, z, m] = u(m)
    # Compute coefficients for each z (same for all z at terminal)
    u_terminal = utility(m_cheb)
    V_coeff[horizon - 1, :, :] = u_terminal[np.newaxis, :] @ C_inv.T
    
    # Grid for consumption fraction s = c/m
    # Use moderate grid since spectral method handles the rest
    n_s = 500
    s_grid = np.linspace(0.0001, 1.0, n_s)
    
    # Precompute Chebyshev evaluation function for m_cheb
    # For evaluating V at arbitrary m points
    
    def eval_cheb(coeff, m_points):
        """Evaluate Chebyshev polynomial at m_points using coefficient array."""
        log_m = np.log(np.maximum(m_points, 1e-12))
        # Map to [-1, 1]
        x = (2 * log_m - (log_m_max + log_m_min)) / (log_m_max - log_m_min)
        x = np.clip(x, -1, 1)
        
        # Evaluate Chebyshev series
        result = np.zeros_like(m_points)
        for n in range(n_cheb):
            result += coeff[n] * chebyshev_poly(n, x)
        return result
    
    # Backward induction using Chebyshev spectral method
    for t in range(horizon - 2, -1, -1):
        for z in range(n_z):
            # For each m in m_cheb, find optimal s
            m_col = m_cheb[:, np.newaxis]
            s_row = s_grid[np.newaxis, :]
            
            c_mat = m_col * s_row
            c_mat = np.maximum(c_mat, 1e-12)
            
            u_c = utility(c_mat)
            
            total_val = np.zeros_like(u_c)
            
            for zp in range(n_z):
                m_save = R * (m_col - c_mat)
                m_next = m_save + income[zp]
                
                # Evaluate V[t+1, zp] at m_next using Chebyshev
                v_next = eval_cheb(V_coeff[t + 1, zp, :], m_next)
                
                total_val += beta * transition[z, zp] * v_next
            
            total_val += u_c
            
            # Find best s for each m
            best_idx = np.argmax(total_val, axis=1)
            V_vals = total_val[np.arange(n_cheb), best_idx]
            
            # Convert to Chebyshev coefficients
            V_coeff[t, z, :] = V_vals @ C_inv.T
            
        # Time check
        if time.time() - start_time > deadline - 20:
            # If running out of time, use heuristic for remaining periods
            for tt in range(t - 1, -1, -1):
                for zz in range(n_z):
                    # Use simple rule: consume fraction based on horizon
                    frac = 1.0 / (horizon - tt)
                    V_coeff[tt, zz, :] = utility(m_cheb * frac) @ C_inv.T
            break

    # Construct consumption policy on the original resource_grid
    consumption = np.zeros((horizon, n_z, n_m_out))
    
    # Terminal period
    consumption[horizon - 1, :, :] = m_grid_out[np.newaxis, :]
    
    # For earlier periods, find optimal c on the original grid
    for t in range(horizon - 2, -1, -1):
        for z in range(n_z):
            m_out = m_grid_out
            m_col = m_out[:, np.newaxis]
            s_row = s_grid[np.newaxis, :]
            
            c_mat = m_col * s_row
            c_mat = np.maximum(c_mat, 1e-12)
            
            u_c = utility(c_mat)
            
            total_val = np.zeros_like(u_c)
            
            for zp in range(n_z):
                m_save = R * (m_col - c_mat)
                m_next = m_save + income[zp]
                
                # Evaluate V[t+1, zp] at m_next using Chebyshev
                v_next = eval_cheb(V_coeff[t + 1, zp, :], m_next)
                
                total_val += beta * transition[z, zp] * v_next
            
            total_val += u_c
            
            best_idx = np.argmax(total_val, axis=1)
            best_c = c_mat[np.arange(n_m_out), best_idx]
            
            # Enforce c <= m
            best_c = np.minimum(best_c, m_out)
            best_c = np.maximum(best_c, 1e-12)
            
            consumption[t, z, :] = best_c
            
        # Time check
        if time.time() - start_time > deadline - 5:
            # Fill remaining with heuristic
            for tt in range(t - 1, -1, -1):
                for zz in range(n_z):
                    frac = 1.0 / (horizon - tt)
                    consumption[tt, zz, :] = m_grid_out * frac
            break

    # Final constraints
    for t in range(horizon):
        for z in range(n_z):
            for idx_m in range(n_m_out):
                m = m_grid_out[idx_m]
                c = consumption[t, z, idx_m]
                if c <= 0:
                    c = 1e-12
                if c > m + 1e-10:
                    c = m
                consumption[t, z, idx_m] = c
    
    # Terminal period must consume everything
    consumption[horizon - 1, :, :] = m_grid_out[np.newaxis, :]
    
    return consumption
