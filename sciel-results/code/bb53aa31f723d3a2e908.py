"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def solve_consumption(model, resource_grid, budget):
    """
    Solve the finite-horizon consumption-saving problem using piecewise log-linear
    value functions and Newton's method on the Euler equation.
    """
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = len(income)
    n_grid = len(resource_grid)
    T = horizon
    
    if abs(gamma - 1.0) < 1e-10:
        def u(c):
            c_safe = np.maximum(c, 1e-12)
            return np.log(c_safe)
        
        def u_prime(c):
            c_safe = np.maximum(c, 1e-12)
            return 1.0 / c_safe
        
        def u_prime2(c):
            c_safe = np.maximum(c, 1e-12)
            return 1.0 / (c_safe * c_safe)
    else:
        def u(c):
            c_safe = np.maximum(c, 1e-12)
            return c_safe ** (1 - gamma) / (1 - gamma)
        
        def u_prime(c):
            c_safe = np.maximum(c, 1e-12)
            return c_safe ** (-gamma)
        
        def u_prime2(c):
            c_safe = np.maximum(c, 1e-12)
            return -gamma * c_safe ** (-gamma - 1)
    
    V = np.zeros((T + 1, n_states, n_grid))
    V[T, :, :] = u(resource_grid)
    
    C = np.zeros((T, n_states, n_grid))
    
    for t in range(T - 1, -1, -1):
        for z in range(n_states):
            for m_idx in range(n_grid):
                m = resource_grid[m_idx]
                
                c = min(m * 0.5, 1.0)
                
                for _newton in range(30):
                    f_val = 0.0
                    fp_val = -u_prime2(c)
                    
                    for zp in range(n_states):
                        p_zp = transition[z, zp]
                        if p_zp < 1e-15:
                            continue
                        m_prime = R * (m - c) + income[zp]
                        m_prime = max(m_prime, 1e-12)
                        
                        b = _interp_b(V[t + 1, zp, :], resource_grid, m_prime)
                        v_prime = b / m_prime
                        v_second = -b / (m_prime * m_prime)
                        
                        f_val += p_zp * v_prime
                        fp_val += beta * R * p_zp * v_second * (-R)
                    
                    f_val = u_prime(c) - beta * R * f_val
                    
                    if abs(fp_val) < 1e-15:
                        break
                    
                    c_new = c - f_val / fp_val
                    c_new = max(c_new, 1e-12)
                    c_new = min(c_new, m)
                    
                    if abs(c_new - c) < 1e-13:
                        c = c_new
                        break
                    c = c_new
                
                C[t, z, m_idx] = c
    
    return C


def _interp_b(V_row, grid, m):
    """
    Return the log-linear coefficient b for the interval containing m,
    such that V[m] = a + b*log(m) on that interval.
    """
    if len(grid) < 2:
        return 0.0
    if m <= grid[0]:
        return (V_row[1] - V_row[0]) / (np.log(grid[1]) - np.log(grid[0]))
    if m >= grid[-1]:
        return (V_row[-1] - V_row[-2]) / (np.log(grid[-1]) - np.log(grid[-2]))
    
    idx = np.searchsorted(grid, m) - 1
    idx = max(0, min(idx, len(grid) - 2))
    return (V_row[idx + 1] - V_row[idx]) / (np.log(grid[idx + 1]) - np.log(grid[idx]))
# EVOLVE-BLOCK-END
