"""Consumption-saving policy via backward induction with golden section refinement."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def solve_consumption(model, resource_grid, budget):
    """
    Solve the consumption-saving problem via backward induction with
    golden-section refinement of the consumption choice.
    
    Parameters:
    -----------
    model : dict with keys 'beta', 'gamma', 'R', 'income', 'transition', 'horizon'
    resource_grid : np.array of shape (101,)
    budget : int (number of consumption grid points per state)
    
    Returns:
    --------
    consumption : np.array of shape (6, 2, 101)
    """
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    
    m_grid = resource_grid
    n_m = len(m_grid)
    n_z = len(income)
    m_min = m_grid[0]
    m_max = m_grid[-1]
    
    # Precompute utility
    if abs(gamma - 1.0) < 1e-10:
        def utility(c):
            return np.log(np.maximum(c, 1e-12))
    else:
        one_minus_gamma = 1.0 - gamma
        def utility(c):
            return np.power(np.maximum(c, 1e-12), one_minus_gamma) / one_minus_gamma
    
    # Terminal: c[horizon-1] = m (consume all)
    consumption = np.zeros((horizon, n_z, n_m))
    consumption[horizon - 1] = m_grid[np.newaxis, :]
    
    # V_next[m_idx, z] = value at t+1
    V_next = np.tile(utility(m_grid)[np.newaxis, :], (n_z, 1))  # shape (n_z, n_m)
    
    # Golden section constants
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618
    n_golden = 25  # iterations for refinement
    
    def golden_section_max(func, a, b, n_iter):
        """Find maximum of func on [a, b] via golden section search."""
        if b - a < 1e-14:
            return (a + b) / 2.0
        c = b - inv_phi * (b - a)
        d = a + inv_phi * (b - a)
        fc = func(c)
        fd = func(d)
        for _ in range(n_iter):
            if fc < fd:
                a = c
                c = d
                d = a + inv_phi * (b - a)
                fc = fd
                fd = func(d)
            else:
                b = d
                d = c
                c = b - inv_phi * (b - a)
                fd = fc
                fc = func(c)
        return (a + b) / 2.0
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fill remaining with consume-all heuristic
            for tt in range(t, horizon - 1):
                consumption[tt] = m_grid[np.newaxis, :]
            break
        
        V_curr = np.zeros((n_z, n_m))
        C_opt = np.zeros((n_z, n_m))
        
        for z in range(n_z):
            trans_z = transition[z]
            
            for m_idx in range(n_m):
                m_val = m_grid[m_idx]
                
                if m_val <= 1e-10:
                    C_opt[z, m_idx] = m_val
                    V_curr[z, m_idx] = utility(m_val)
                    continue
                
                # Define objective: given c, compute total value
                def obj(c_val):
                    if c_val <= 0:
                        c_val = 1e-12
                    if c_val > m_val:
                        c_val = m_val
                    m_minus_c = m_val - c_val
                    if m_minus_c < 0:
                        m_minus_c = 0.0
                    m_prime = R * m_minus_c + income  # shape (n_z,)
                    m_prime = np.clip(m_prime, m_min, m_max)
                    # Interpolate V_next for each z'
                    cont = 0.0
                    for zp in range(n_z):
                        cont += trans_z[zp] * np.interp(m_prime[zp], m_grid, V_next[zp, :])
                    return utility(c_val) + beta * cont
                
                # Initial bracket: use a coarse grid to find approximate best
                # c in [eps*m, m]
                eps_frac = 0.001
                c_lo = eps_frac * m_val
                c_hi = m_val
                
                # Quick scan with fewer points to narrow bracket
                n_scan = min(budget, 50)
                c_scan = np.linspace(c_lo, c_hi, n_scan)
                vals_scan = np.array([obj(c) for c in c_scan])
                best_scan_idx = np.argmax(vals_scan)
                
                # Refine around the best scan point using golden section
                # Expand bracket slightly around best
                if best_scan_idx == 0:
                    a = c_lo
                    b = c_scan[1] if n_scan > 1 else c_hi
                elif best_scan_idx == n_scan - 1:
                    a = c_scan[-2] if n_scan > 1 else c_lo
                    b = c_hi
                else:
                    a = c_scan[best_scan_idx - 1]
                    b = c_scan[best_scan_idx + 1]
                
                # Ensure bracket is valid
                if b - a < 1e-10:
                    best_c = (a + b) / 2.0
                else:
                    best_c = golden_section_max(obj, a, b, n_golden)
                
                # Clamp
                best_c = max(best_c, 1e-12)
                best_c = min(best_c, m_val)
                
                C_opt[z, m_idx] = best_c
                V_curr[z, m_idx] = obj(best_c)
            
            # Time check within z loop
            if time.time() > deadline:
                # Fill remaining m for this z and remaining z
                for mm in range(n_m):
                    if C_opt[z, mm] == 0.0:
                        C_opt[z, mm] = m_grid[mm]
                        V_curr[z, mm] = utility(m_grid[mm])
                for zz in range(z + 1, n_z):
                    C_opt[zz] = m_grid
                    V_curr[zz] = utility(m_grid)
                break
        
        consumption[t] = C_opt
        V_next = V_curr
    
    # Ensure terminal condition
    consumption[horizon - 1] = m_grid[np.newaxis, :]
    
    # Validate: 0 < c <= m + 1e-10, finite
    for t in range(horizon):
        for z in range(n_z):
            for m_idx in range(n_m):
                m_val = m_grid[m_idx]
                c_val = consumption[t, z, m_idx]
                if not np.isfinite(c_val) or c_val <= 0:
                    consumption[t, z, m_idx] = max(1e-12, m_val)
                if c_val > m_val + 1e-10:
                    consumption[t, z, m_idx] = m_val
    
    return consumption
# EVOLVE-BLOCK-END
