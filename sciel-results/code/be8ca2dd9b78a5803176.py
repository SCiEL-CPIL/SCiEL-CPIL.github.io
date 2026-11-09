"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import brentq

def _crsa_utility(c, gamma):
    c = np.maximum(c, 1e-12)
    if gamma == 1:
        return np.log(c)
    else:
        return (c ** (1 - gamma)) / (1 - gamma)

def _crsa_marginal(c, gamma):
    c = np.maximum(c, 1e-12)
    if gamma == 1:
        return 1.0 / c
    else:
        return c ** (-gamma)

def _solve_consumption_single(beta, gamma, R, income, transition, horizon, resource_grid):
    T = horizon
    n_z = len(income)
    n_m = len(resource_grid)
    
    # V[t, z, m] stores the value function at time t for state z and resource m
    # c[t, z, m] stores the optimal consumption
    V = np.zeros((T + 1, n_z, n_m))
    c_opt = np.zeros((T, n_z, n_m))
    
    # Terminal condition: c[T] = m
    for z in range(n_z):
        V[T, z, :] = _crsa_utility(resource_grid, gamma)
        
    # Backward induction
    for t in range(T - 1, -1, -1):
        for z in range(n_z):
            m_vals = resource_grid
            # We need to find c in [0, m] that maximizes:
            # u(c) + beta * sum_{z'} P(z'|z) * V[t+1, z', R(m-c) + income[z']]
            
            # Since V is convex in m (typically), the objective is concave in c.
            # We can use Newton's method or brentq on the derivative.
            # Let f(c) = u'(c) - beta * sum P * V'_next(m')
            # where m' = R(m-c) + income[z']
            
            # V'_next can be computed from the envelope theorem or numerically.
            # However, it's often more robust to just use the FOC directly if we can express it.
            # Actually, since we are on a grid, we can't easily get analytic V'.
            # Let's stick to a robust optimization method but faster than grid search.
            
            # Alternative: Use the fact that the optimal policy is monotonic.
            # But let's try a dense grid with interpolation of the *value* function, 
            # but solve for c using a local search or root finding on the discrete objective.
            
            # Given the time constraint and the need for a structural change, 
            # let's use a "refined grid" approach but with a twist: 
            # We will compute the objective on a very fine grid for c, 
            # but we will vectorize the evaluation of the continuation value.
            
            # Wait, the "structural change" requested is different from grid search.
            # Let's use the Euler equation / FOC root finding.
            # We need V'_t+1. We can approximate V'_t+1 by differentiating the policy or value.
            # A simpler robust method: 
            # 1. Evaluate objective on a coarse grid to get a rough estimate.
            # 2. Use that to bracket the root.
            # 3. Use brentq on the derivative of the objective.
            
            # To get the derivative of the objective with respect to c:
            # Obj(c) = u(c) + beta * sum_{z'} P_{z,z'} * V_{t+1}(z', m')
            # dObj/dc = u'(c) - beta * R * sum_{z'} P_{z,z'} * V'_{t+1}(z', m')
            
            # We can approximate V'_{t+1} using finite differences on the grid m.
            # V'_{t+1}(z', m) ≈ (V_{t+1}(z', m+dm) - V_{t+1}(z', m-dm)) / (2dm)
            
            # Let's precompute V_prime for t+1
            # V_prime[t+1, z, m]
            
            # For the first step (t=T-1), V_T is known analytically.
            if t == T - 1:
                # V_T(m) = u(m)
                # V'_T(m) = u'(m)
                V_prime_next = np.zeros((n_z, n_m))
                for z in range(n_z):
                    V_prime_next[z, :] = _crsa_marginal(resource_grid, gamma)
            else:
                # Approximate V_prime from V
                # Use central differences where possible, forward/backward at edges
                V_prime_next = np.zeros((n_z, n_m))
                dm = np.diff(resource_grid)
                # Simple finite difference approximation
                # V' ≈ (V[i+1] - V[i-1]) / (m[i+1] - m[i-1])
                for z in range(n_z):
                    v = V[t + 1, z, :]
                    # Gradient calculation
                    dv = np.zeros_like(v)
                    # Forward difference for last, backward for first, central for middle
                    dv[0] = (v[1] - v[0]) / (resource_grid[1] - resource_grid[0])
                    dv[-1] = (v[-1] - v[-2]) / (resource_grid[-1] - resource_grid[-2])
                    for i in range(1, n_m - 1):
                        dv[i] = (v[i+1] - v[i-1]) / (resource_grid[i+1] - resource_grid[i-1])
                    V_prime_next[z, :] = dv

            # Now solve for c for each m
            for i_m in range(n_m):
                m = resource_grid[i_m]
                
                # Define the function to root-find: dObj/dc = 0
                def d_obj_dc(c, m=m, z=z):
                    # u'(c)
                    u_prime = _crsa_marginal(c, gamma)
                    
                    # Continuation term
                    # m' = R(m - c) + income[z']
                    # V'_next(z', m')
                    # We need to interpolate V'_next(z', m')
                    
                    # If c is such that m' is outside [min_grid, max_grid], we need to handle it.
                    # m' = R(m-c) + income[z']
                    # Since c <= m, m' >= R*0 + income[z'] = income[z']
                    # And m' <= R(m-0) + income[z'] = R*m + income[z']
                    
                    # The grid is geomspace(0.001, 40, 101).
                    # Max m is 40. Max m' is 1.02*40 + 1.5 = 42.3.
                    # This is outside the grid.
                    # We must extrapolate or clamp.
                    # Clamping V' to the boundary value is a common approximation.
                    
                    cont = 0.0
                    for z_prime in range(n_z):
                        p = transition[z, z_prime]
                        if p == 0:
                            continue
                        m_prime = R * (m - c) + income[z_prime]
                        
                        # Interpolate V_prime_next[z_prime, :] at m_prime
                        # If m_prime > max_grid, use V_prime_next[z_prime, -1]
                        # If m_prime < min_grid, use V_prime_next[z_prime, 0]
                        
                        if m_prime > resource_grid[-1]:
                            vp = V_prime_next[z_prime, -1]
                        elif m_prime < resource_grid[0]:
                            vp = V_prime_next[z_prime, 0]
                        else:
                            vp = np.interp(m_prime, resource_grid, V_prime_next[z_prime, :])
                        
                        cont += p * vp
                    
                    return u_prime - beta * R * cont

                # We know c is in (0, m).
                # Check boundaries.
                # If d_obj_dc(epsilon) < 0, then c is at lower bound? No, u' is high at low c.
                # u'(c) -> inf as c -> 0. So d_obj_dc(0) > 0.
                # As c -> m, u'(c) is finite.
                # If d_obj_dc(m) > 0, then c = m.
                # If d_obj_dc(m) < 0, then root is in (0, m).
                
                eps = 1e-6
                c_low = eps
                c_high = m
                
                f_low = d_obj_dc(c_low)
                f_high = d_obj_dc(c_high)
                
                if f_high > 0:
                    # Optimal c is m (consume everything)
                    c_val = m
                elif f_low < 0:
                    # This shouldn't happen for standard CRRA with beta < 1, but just in case
                    c_val = c_low
                else:
                    try:
                        c_val = brentq(d_obj_dc, c_low, c_high, xtol=1e-8, rtol=1e-8)
                    except ValueError:
                        # Fallback to grid search if root finding fails
                        c_grid = np.linspace(c_low, c_high, 100)
                        obj = np.array([_crsa_utility(c, gamma) + beta * sum(
                            transition[z, z_p] * np.interp(R*(m-c)+income[z_p], resource_grid, V[t+1, z_p, :])
                            for z_p in range(n_z)
                        ) for c in c_grid])
                        c_val = c_grid[np.argmax(obj)]
                
                c_opt[t, z, i_m] = c_val
                
                # Calculate V[t, z, m]
                # V = u(c) + beta * sum P * V_next(m')
                v_cont = 0.0
                for z_prime in range(n_z):
                    p = transition[z, z_prime]
                    if p == 0:
                        continue
                    m_prime = R * (m - c_val) + income[z_prime]
                    if m_prime > resource_grid[-1]:
                        # Extrapolate? Or clamp?
                        # For value function, linear extrapolation might be better, but clamping is safer for stability
                        vp = V[t + 1, z_prime, -1]
                    elif m_prime < resource_grid[0]:
                        vp = V[t + 1, z_prime, 0]
                    else:
                        vp = np.interp(m_prime, resource_grid, V[t + 1, z_prime, :])
                    v_cont += p * vp
                
                V[t, z, i_m] = _crsa_utility(c_val, gamma) + beta * v_cont

    return c_opt

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = 120.0 # Seconds
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # Ensure resource_grid is sorted
    resource_grid = np.sort(resource_grid)
    
    # Check if we have enough time
    if time.time() - start_time > deadline - 5:
        # Return a trivial solution if no time
        c = np.zeros((horizon, len(income), len(resource_grid)))
        for t in range(horizon):
            for z in range(len(income)):
                c[t, z, :] = resource_grid
        return c

    try:
        c_opt = _solve_consumption_single(beta, gamma, R, income, transition, horizon, resource_grid)
    except Exception:
        # Fallback to a simpler method if complex one fails
        # Simple grid search
        c_opt = np.zeros((horizon, len(income), len(resource_grid)))
        V = np.zeros((horizon + 1, len(income), len(resource_grid)))
        for z in range(len(income)):
            V[horizon, z, :] = _crsa_utility(resource_grid, gamma)
            
        for t in range(horizon - 1, -1, -1):
            for z in range(len(income)):
                for i_m, m in enumerate(resource_grid):
                    c_grid = np.linspace(1e-6, m, 50)
                    best_c = 1e-6
                    best_v = -np.inf
                    for c in c_grid:
                        v_cont = 0.0
                        for z_p in range(len(income)):
                            m_p = R * (m - c) + income[z_p]
                            if m_p > resource_grid[-1]:
                                vp = V[t+1, z_p, -1]
                            elif m_p < resource_grid[0]:
                                vp = V[t+1, z_p, 0]
                            else:
                                vp = np.interp(m_p, resource_grid, V[t+1, z_p, :])
                            v_cont += transition[z, z_p] * vp
                        v = _crsa_utility(c, gamma) + beta * v_cont
                        if v > best_v:
                            best_v = v
                            best_c = c
                    c_opt[t, z, i_m] = best_c
                    V[t, z, i_m] = best_v

    # Final checks
    # c[5] (last period) should be equal to grid
    c_opt[horizon-1, :, :] = resource_grid[np.newaxis, :, :]
    
    # Ensure 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(len(income)):
            c_opt[t, z, :] = np.clip(c_opt[t, z, :], 1e-12, resource_grid + 1e-10)
            
    return c_opt
# EVOLVE-BLOCK-END
