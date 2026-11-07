"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # Resource grid
    m_grid = resource_grid  # shape (101,)
    n_m = len(m_grid)
    n_z = len(income)
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        else:
            return c**(1 - gamma) / (1 - gamma)
    
    # We'll use Chebyshev polynomial approximation for the value function
    # First, map m_grid to Chebyshev nodes [-1, 1]
    m_min = m_grid[0]
    m_max = m_grid[-1]
    
    # Since m_grid is geomspace, let's work in log-space for better conditioning
    log_m_min = np.log(m_min)
    log_m_max = np.log(m_max)
    
    # Map to [-1, 1]
    def m_to_x(m):
        return (np.log(m) - log_m_min) / (log_m_max - log_m_min) * 2 - 1
    
    def x_to_m(x):
        log_m = log_m_min + (x + 1) / 2 * (log_m_max - log_m_min)
        return np.exp(log_m)
    
    x_grid = m_to_x(m_grid)
    
    # Number of Chebyshev coefficients
    n_cheb = 30  # degree 29 polynomial
    
    # Precompute Chebyshev basis at grid points
    # T_k(x) = cos(k * arccos(x))
    theta = np.arccos(np.clip(x_grid, -1, 1))
    cheb_basis = np.cos(np.outer(np.arange(n_cheb), theta))  # shape (n_m, n_cheb)
    
    # Initialize consumption policy: shape (horizon, n_z, n_m)
    consumption = np.zeros((horizon, n_z, n_m))
    
    # Value function coefficients for each z at each time
    # V_t(m, z) approximated as sum_k V_coeffs[t, z, k] * T_k(x(m))
    
    # Terminal condition: at t=horizon-1, no future, so V_{T}(m,z) = u(m) (consume all)
    # Actually, let's think about timing. horizon=6, so t=0,1,2,3,4,5
    # At t=5 (last period), consume all: c = m
    consumption[5, :, :] = m_grid[np.newaxis, np.newaxis, :]
    
    # For t=5, value function is just utility(m)
    # Compute value at t=5 for each z (same for all z since no future)
    V5 = utility(m_grid)  # shape (n_m,)
    
    # Fit Chebyshev coefficients for V5
    # V5[m] = sum_k V_coeffs[5, k] * T_k(x[m])
    # Least squares: V5 = cheb_basis @ V_coeffs
    V_coeffs_5 = np.linalg.lstsq(cheb_basis, V5, rcond=None)[0]  # shape (n_cheb,)
    
    # For t=4 down to 0:
    for t in range(4, -1, -1):
        # For each z, compute optimal consumption
        for z in range(n_z):
            # Continuation value for each possible next state z'
            # V_{t+1}(m', z') where m' = R*(m - c) + income[z']
            
            # We need to find c that maximizes:
            # u(c) + beta * sum_{z'} transition[z, z'] * V_{t+1}(R*(m-c) + income[z'], z')
            
            # For each m in m_grid, find optimal c
            c_opt = np.zeros(n_m)
            
            for i_m in range(n_m):
                m = m_grid[i_m]
                
                # Candidate consumptions: c from small to m
                # Use a fine grid of c values
                # c must be in (0, m]
                c_grid = np.linspace(0.001 * m, m, 200)
                
                # Compute continuation value for each c
                # m' = R * (m - c) + income[z']
                cont_vals = np.zeros(len(c_grid))
                
                for j_c in range(len(c_grid)):
                    c = c_grid[j_c]
                    m_remaining = m - c
                    total_cont = 0.0
                    for zp in range(n_z):
                        m_prime = R * m_remaining + income[zp]
                        # Clip m_prime to valid range
                        if m_prime <= m_min:
                            m_prime_clipped = m_min
                        elif m_prime >= m_max:
                            m_prime_clipped = m_max
                        else:
                            # Interpolate using Chebyshev
                            x_prime = m_to_x(m_prime)
                            x_prime = np.clip(x_prime, -1, 1)
                            theta_prime = np.arccos(x_prime)
                            cheb_prime = np.cos(np.arange(n_cheb) * theta_prime)
                            # Get V_{t+1} coefficients for z'
                            if t + 1 == 5:
                                V_next_coeffs = V_coeffs_5
                            else:
                                V_next_coeffs = V_coeffs[t + 1, zp]
                            v_next = np.dot(cheb_prime, V_next_coeffs)
                        total_cont += transition[z, zp] * v_next
                    cont_vals[j_c] = total_cont
                
                # Total value: u(c) + beta * cont
                total_vals = utility(c_grid) + beta * cont_vals
                
                # Find best c
                best_idx = np.argmax(total_vals)
                c_opt[i_m] = c_grid[best_idx]
            
            consumption[t, z, :] = c_opt
            
            # Now compute V_t(m, z) = u(c_opt) + beta * E[V_{t+1}(m', z')]
            V_t_vals = np.zeros(n_m)
            for i_m in range(n_m):
                m = m_grid[i_m]
                c = c_opt[i_m]
                m_remaining = m - c
                total_cont = 0.0
                for zp in range(n_z):
                    m_prime = R * m_remaining + income[zp]
                    if m_prime <= m_min:
                        m_prime_clipped = m_min
                    elif m_prime >= m_max:
                        m_prime_clipped = m_max
                    else:
                        x_prime = m_to_x(m_prime)
                        x_prime = np.clip(x_prime, -1, 1)
                        theta_prime = np.arccos(x_prime)
                        cheb_prime = np.cos(np.arange(n_cheb) * theta_prime)
                        if t + 1 == 5:
                            V_next_coeffs = V_coeffs_5
                        else:
                            V_next_coeffs = V_coeffs[t + 1, zp]
                        v_next = np.dot(cheb_prime, V_next_coeffs)
                    total_cont += transition[z, zp] * v_next
                V_t_vals[i_m] = utility(c) + beta * total_cont
            
            # Fit Chebyshev coefficients for V_t(z)
            V_coeffs_t = np.linalg.lstsq(cheb_basis, V_t_vals, rcond=None)[0]
            
            # Store
            if t < 5:
                if not hasattr(solve_consumption, '_V_coeffs'):
                    solve_consumption._V_coeffs = {}
                solve_consumption._V_coeffs[(t, z)] = V_coeffs_t
    
    # Wait, I need to restructure this. Let me redo with proper storage.
    # Let me restart with a cleaner implementation.
    
    # Clear any previous state
    if hasattr(solve_consumption, '_V_coeffs'):
        del solve_consumption._V_coeffs
    
    consumption = np.zeros((horizon, n_z, n_m))
    consumption[5, :, :] = m_grid[np.newaxis, np.newaxis, :]
    
    V_coeffs = {}  # (t, z) -> array of shape (n_cheb,)
    
    # Terminal: t=5
    V5 = utility(m_grid)
    V_coeffs[(5, 0)] = np.linalg.lstsq(cheb_basis, V5, rcond=None)[0]
    V_coeffs[(5, 1)] = np.linalg.lstsq(cheb_basis, V5, rcond=None)[0]
    
    for t in range(4, -1, -1):
        for z in range(n_z):
            c_opt = np.zeros(n_m)
            V_t_vals = np.zeros(n_m)
            
            for i_m in range(n_m):
                m = m_grid[i_m]
                c_grid = np.linspace(0.001 * m, m, 200)
                
                best_val = -np.inf
                best_c = m
                for j_c in range(len(c_grid)):
                    c = c_grid[j_c]
                    m_remaining = m - c
                    total_cont = 0.0
                    for zp in range(n_z):
                        m_prime = R * m_remaining + income[zp]
                        if m_prime <= m_min:
                            m_prime_clipped = m_min
                        elif m_prime >= m_max:
                            m_prime_clipped = m_max
                        else:
                            x_prime = m_to_x(m_prime)
                            x_prime = np.clip(x_prime, -1, 1)
                            theta_prime = np.arccos(x_prime)
                            cheb_prime = np.cos(np.arange(n_cheb) * theta_prime)
                            V_next_coeffs = V_coeffs[(t + 1, zp)]
                            v_next = np.dot(cheb_prime, V_next_coeffs)
                        total_cont += transition[z, zp] * v_next
                    total_val = utility(c) + beta * total_cont
                    if total_val > best_val:
                        best_val = total_val
                        best_c = c
                
                c_opt[i_m] = best_c
                V_t_vals[i_m] = best_val
            
            consumption[t, z, :] = c_opt
            V_coeffs[(t, z)] = np.linalg.lstsq(cheb_basis, V_t_vals, rcond=None)[0]
    
    return consumption
# EVOLVE-BLOCK-END
