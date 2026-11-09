"""Log-linear value function approximation with Newton-Kantorovich iterations."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def _utility(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(c)
    return c ** (1.0 - gamma) / (1.0 - gamma)

def _utility_prime(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return 1.0 / c
    return c ** (-gamma)

def solve_consumption(model, resource_grid, budget):
    _t0 = time.time()
    deadline = _t0 + 110.0

    beta = model['beta']
    gamma = model['gamma']
    R = model.get('R', 1.02)
    income = np.asarray(model['income'], dtype=float)
    Tmat = np.asarray(model['transition'], dtype=float)
    horizon = int(model['horizon'])
    m_grid = np.asarray(resource_grid, dtype=float)

    n_z = len(income)
    n_m = len(m_grid)
    C = np.empty((horizon, n_z, n_m), dtype=float)

    # Terminal period: consume all
    C[horizon - 1, :, :] = m_grid

    # Value function approximation: V(z, m) = A[z] + B[z] * log(m)
    # For terminal: V(z, m) = u(m)
    # For CRRA with gamma != 1: u(m) = m^(1-gamma)/(1-gamma), which is not log-linear
    # But we can still approximate. For gamma=1: u(m)=log(m) = 0 + 1*log(m), so A=0, B=1.
    # For general gamma, we'll use a log-linear approximation.
    
    # Initialize with terminal values
    # V(z, m) = u(m) for terminal
    # We'll represent as A[z] + B[z]*log(m)
    # For gamma=1: A=0, B=1
    # For gamma!=1: we need to approximate m^(1-gamma)/(1-gamma) with A + B*log(m)
    # This is not exact, but we can do a least-squares fit on the grid.
    
    A = np.zeros(n_z)
    B = np.zeros(n_z)
    
    for z in range(n_z):
        if abs(gamma - 1.0) < 1e-12:
            A[z] = 0.0
            B[z] = 1.0
        else:
            # Fit u(m) = A + B*log(m) to m^(1-gamma)/(1-gamma) on m_grid
            log_m = np.log(m_grid)
            u_vals = _utility(m_grid, gamma)
            # Least squares: [1, log_m] @ [A, B]^T = u_vals
            X = np.column_stack([np.ones(n_m), log_m])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, u_vals, rcond=None)
                A[z] = coeffs[0]
                B[z] = coeffs[1]
            except:
                A[z] = 0.0
                B[z] = 1.0

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        # Newton-Kantorovich iteration to update A, B
        # We want to find A_new[z], B_new[z] such that:
        # A_new[z] + B_new[z]*log(m) = max_c { u(c) + beta * sum_{z'} T[z,z'] * (A[z'] + B[z']*log(m_next)) }
        # where m_next = R*(m-c) + income[z']
        
        # For a fixed policy c(m), the RHS is:
        # u(c(m)) + beta * sum_{z'} T[z,z'] * A[z'] + beta * sum_{z'} T[z,z'] * B[z'] * log(R*(m-c(m)) + income[z'])
        
        # The FOC for optimal c:
        # u'(c) = beta * R * sum_{z'} T[z,z'] * B[z'] / (R*(m-c) + income[z'])
        
        # This is a nonlinear equation in c for each m.
        # We solve it numerically, then update A, B by fitting.
        
        A_new = np.zeros(n_z)
        B_new = np.zeros(n_z)
        
        for z in range(n_z):
            probs = Tmat[z]
            
            # For each m in m_grid, solve for optimal c
            c_opt = np.zeros(n_m)
            for i, m in enumerate(m_grid):
                # FOC: u'(c) = beta * R * sum_{z'} T[z,z'] * B[z'] / (R*(m-c) + income[z'])
                # Define f(c) = u'(c) - beta*R*sum_{z'} T[z,z']*B[z']/(R*(m-c)+income[z'])
                
                def f(c, m=m):
                    m_next = R * (m - c) + income
                    m_next = np.maximum(m_next, 1e-10)
                    rhs = beta * R * np.sum(probs * B / m_next)
                    return _utility_prime(c, gamma) - rhs
                
                c_low = 1e-10
                c_high = m - 1e-12
                if c_high <= c_low:
                    c_opt[i] = max(c_low, m)
                    continue
                
                f_low = f(c_low)
                f_high = f(c_high)
                
                if f_low <= 0:
                    c_opt[i] = c_low
                elif f_high >= 0:
                    c_opt[i] = c_high
                else:
                    # Bisection
                    lo, hi = c_low, c_high
                    for _ in range(50):
                        mid = 0.5 * (lo + hi)
                        if f(mid) > 0:
                            lo = mid
                        else:
                            hi = mid
                    c_opt[i] = 0.5 * (lo + hi)
                
                if time.time() > deadline:
                    # Fill remaining with consume-all heuristic
                    for zi in range(i + 1, n_m):
                        c_opt[zi] = m_grid[zi]
                    break
            
            C[t, z, :] = c_opt
            
            # Compute RHS values for fitting A_new[z] + B_new[z]*log(m)
            # RHS(m) = u(c(m)) + beta * sum_{z'} T[z,z'] * (A[z'] + B[z']*log(R*(m-c(m))+income[z']))
            rhs_vals = np.zeros(n_m)
            for i, m in enumerate(m_grid):
                c = c_opt[i]
                m_next = R * (m - c) + income
                m_next = np.maximum(m_next, 1e-10)
                cont = np.sum(probs * (A + B * np.log(m_next)))
                rhs_vals[i] = _utility(c, gamma) + beta * cont
            
            # Fit A_new[z] + B_new[z]*log(m) = rhs_vals
            log_m = np.log(m_grid)
            X = np.column_stack([np.ones(n_m), log_m])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, rhs_vals, rcond=None)
                A_new[z] = coeffs[0]
                B_new[z] = coeffs[1]
            except:
                A_new[z] = A[z]
                B_new[z] = B[z]
        
        A = A_new
        B = B_new
        
        if time.time() > deadline:
            break
    
    # Final pass: recompute optimal policies using the final A, B
    # This ensures consistency
    for t in range(horizon - 2, -1, -1):
        for z in range(n_z):
            probs = Tmat[z]
            for i, m in enumerate(m_grid):
                def f(c, m=m):
                    m_next = R * (m - c) + income
                    m_next = np.maximum(m_next, 1e-10)
                    rhs = beta * R * np.sum(probs * B / m_next)
                    return _utility_prime(c, gamma) - rhs
                
                c_low = 1e-10
                c_high = m - 1e-12
                if c_high <= c_low:
                    C[t, z, i] = max(c_low, m)
                    continue
                
                f_low = f(c_low)
                f_high = f(c_high)
                
                if f_low <= 0:
                    C[t, z, i] = c_low
                elif f_high >= 0:
                    C[t, z, i] = c_high
                else:
                    lo, hi = c_low, c_high
                    for _ in range(50):
                        mid = 0.5 * (lo + hi)
                        if f(mid) > 0:
                            lo = mid
                        else:
                            hi = mid
                    C[t, z, i] = 0.5 * (lo + hi)
                
                if time.time() > deadline:
                    for zi in range(i + 1, n_m):
                        C[t, z, zi] = m_grid[zi]
                    for zz in range(z + 1, n_z):
                        C[t, zz, :] = m_grid
                    C[horizon - 1, :, :] = m_grid
                    return C
            
            if time.time() > deadline:
                for zz in range(z + 1, n_z):
                    C[t, zz, :] = m_grid
                C[horizon - 1, :, :] = m_grid
                return C
    
    # Enforce constraints
    C = np.clip(C, 1e-12, None)
    for t in range(horizon):
        for z in range(n_z):
            C[t, z, :] = np.minimum(C[t, z, :], m_grid + 1e-10)
    C[horizon - 1, :, :] = m_grid
    
    return C
# EVOLVE-BLOCK-END
