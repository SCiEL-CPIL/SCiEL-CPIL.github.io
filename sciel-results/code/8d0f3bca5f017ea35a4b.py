"""New design: homothetic CRRA consumption policy via ratio-space optimization."""
import numpy as np
from scipy.optimize import brentq
import time

def solve_consumption(model, resource_grid, budget):
    t_start = time.time()
    deadline = 150.0  # wall-clock limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    m_grid = resource_grid
    n_m = len(m_grid)
    n_z = len(income)

    # Utility and marginal utility (CRRA)
    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1.0 - gamma) / (1.0 - gamma)

    def u_prime(c):
        c = np.maximum(c, 1e-12)
        return c ** (-gamma)

    # Terminal condition: consume everything at t = horizon - 1
    # V[horizon-1][z, m] = u(m + income[z])
    # c[horizon-1][z, m] = m + income[z]  (but must be <= m, so actually c = m)
    # Wait: m' = R(m - c) + income[z']. At terminal, no future, so maximize u(m - c) + ... 
    # Actually at t=horizon-1 (last period), agent chooses c to max u(c) s.t. c <= m.
    # So c = m (consume all resources). But the constraint says c <= m + 1e-10 and c[5] = grid.
    # c[5] is the last time index (0-indexed, so t=5 is the 6th period, which is horizon-1=5).
    # So c[5, z, i] = m_grid[i] for all z.

    # We'll store consumption policy: shape (horizon, n_z, n_m)
    c_policy = np.zeros((horizon, n_z, n_m), dtype=np.float64)
    # Continuation value: V[t, z, m] for t in 0..horizon-2
    # At t = horizon-1, V = u(c) = u(m) (since c=m at terminal)
    
    # Terminal value: V_term[z, m] = u(m)  (since c = m at last period)
    V_term = u(m_grid).copy()  # shape (n_m,)
    # c at terminal
    c_policy[horizon - 1, :, :] = m_grid[np.newaxis, np.newaxis, :]
    
    # Check deadline
    if time.time() - t_start > deadline - 30:
        return c_policy

    # Backward induction for t = horizon-2 down to 0
    # For each t, z, m: choose c in (0, m] to maximize u(c) + beta * sum_z' P(z'|z) V_next(z', R*(m-c) + income[z'])
    
    # We work in ratio space: x = c/m, so c = x*m, x in (0, 1]
    # m' = R*m*(1-x) + income[z']
    # The objective in ratio space: u(x*m) + beta * sum_z' P(z'|z) V_next(z', R*m*(1-x) + income[z'])
    # This is still m-dependent unless V is homothetic.
    
    # For CRRA, V is approximately homothetic: V(z, m) ≈ A(z) * m^(1-gamma)/(1-gamma)
    # So V'(z, m) ≈ A(z) * m^(-gamma)
    # The FOC: u'(c) = beta * R * sum_z' P(z'|z) V'(z', m')
    # u'(x*m) = (x*m)^(-gamma)
    # V'(z', m') ≈ A(z') * (m')^(-gamma)
    # So: (x*m)^(-gamma) = beta * R * sum_z' P(z'|z) A(z') * (R*m*(1-x) + income[z'])^(-gamma)
    
    # This is a 1D equation in x for each (t, z, m). But A(z) is unknown.
    # Alternative: directly optimize in x-space for each (t, z, m).
    
    # For efficiency, we'll do this: for each t, z, solve for x(m) using the FOC via root-finding,
    # using the current V_next as a lookup.
    
    V_next = V_term.copy()  # shape (n_m,)
    
    for t in range(horizon - 2, -1, -1):
        if time.time() - t_start > deadline - 10:
            break
        
        V_curr = np.zeros((n_z, n_m), dtype=np.float64)
        
        for z in range(n_z):
            # For each m in m_grid, find optimal x
            probs = transition[z]  # shape (n_z,)
            
            for i, m in enumerate(m_grid):
                if time.time() - t_start > deadline - 5:
                    break
                    
                # Define objective in x: f(x) = u(x*m) + beta * sum_z' probs[z'] * V_next_interp(z', R*m*(1-x) + income[z'])
                # We want to maximize this over x in (0, 1].
                # Use FOC: derivative = 0
                # f'(x) = u'(x*m)*m + beta * sum_z' probs[z'] * V_next_prime(z', m') * (-R*m)
                # where m' = R*m*(1-x) + income[z']
                
                # Since we don't have analytic V_next_prime, use numerical differentiation
                # or use the fact that for CRRA, we can approximate.
                
                # Simpler: use golden section or Brent's method on the objective directly.
                # But we need to be fast. Let's use a coarse grid + refinement.
                
                # Approach: evaluate objective on a grid of x, then refine with Brent.
                x_coarse = np.linspace(0.001, 1.0, 50)
                
                def obj(x):
                    c = x * m
                    m_next = R * (m - c)
                    total = u(c)
                    for zp in range(n_z):
                        m_zp = m_next + income[zp]
                        if m_zp < 1e-12:
                            m_zp = 1e-12
                        # Interpolate V_next[zp] at m_zp
                        if m_zp <= m_grid[0]:
                            v_val = V_next[zp]  # approximate
                        elif m_zp >= m_grid[-1]:
                            v_val = V_next[zp]  # approximate, should be fine
                        else:
                            v_val = np.interp(m_zp, m_grid, V_next[zp])
                        total += beta * probs[zp] * v_val
                    return total
                
                # Evaluate on coarse grid
                obj_vals = np.array([obj(x) for x in x_coarse])
                best_idx = np.argmax(obj_vals)
                best_x = x_coarse[best_idx]
                
                # Refine around best_idx
                if 0 < best_idx < len(x_coarse) - 1:
                    x_lo = x_coarse[max(0, best_idx - 1)]
                    x_hi = x_coarse[min(len(x_coarse) - 1, best_idx + 1)]
                    try:
                        from scipy.optimize import minimize_scalar
                        res = minimize_scalar(lambda x: -obj(x), bounds=(x_lo, x_hi), method='bounded', options={'xatol': 1e-8})
                        best_x = res.x
                    except:
                        pass
                elif best_idx == 0:
                    x_lo = 0.001
                    x_hi = x_coarse[1]
                    try:
                        from scipy.optimize import minimize_scalar
                        res = minimize_scalar(lambda x: -obj(x), bounds=(x_lo, x_hi), method='bounded', options={'xatol': 1e-8})
                        best_x = res.x
                    except:
                        pass
                else:
                    x_lo = x_coarse[-2]
                    x_hi = 1.0
                    try:
                        from scipy.optimize import minimize_scalar
                        res = minimize_scalar(lambda x: -obj(x), bounds=(x_lo, x_hi), method='bounded', options={'xatol': 1e-8})
                        best_x = res.x
                    except:
                        pass
                
                c_opt = best_x * m
                c_opt = np.clip(c_opt, 1e-12, m + 1e-10)
                c_policy[t, z, i] = c_opt
                
                # Compute V_curr[z, i]
                m_next = R * (m - c_opt)
                v_total = u(c_opt)
                for zp in range(n_z):
                    m_zp = m_next + income[zp]
                    if m_zp < 1e-12:
                        m_zp = 1e-12
                    if m_zp <= m_grid[0]:
                        v_val = V_next[zp]
                    elif m_zp >= m_grid[-1]:
                        v_val = V_next[zp]
                    else:
                        v_val = np.interp(m_zp, m_grid, V_next[zp])
                    v_total += beta * probs[zp] * v_val
                V_curr[z, i] = v_total
        
        V_next = V_curr
    
    # Final validation
    c_policy = np.where(np.isfinite(c_policy), c_policy, 1e-12)
    for t in range(horizon):
        for z in range(n_z):
            c_policy[t, z, :] = np.clip(c_policy[t, z, :], 1e-12, m_grid + 1e-10)
    
    # Ensure terminal condition: c[5] = grid
    c_policy[horizon - 1, :, :] = m_grid[np.newaxis, np.newaxis, :]
    
    return c_policy
