"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    H = model['horizon']

    m_grid = resource_grid
    M = len(m_grid)

    # Utility: exact CRRA as specified
    def utility(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return c ** (1.0 - gamma) / (1.0 - gamma)

    # Value and policy arrays
    V = np.zeros((H, 2, M))
    C = np.zeros((H, 2, M))

    # Terminal period: t = H-1 = 5, c = m
    V[H - 1, :, :] = utility(m_grid)[np.newaxis, np.newaxis, :]
    C[H - 1, :, :] = m_grid[np.newaxis, np.newaxis, :]

    # Pre-compute spline for the terminal period
    # We will build splines for each t+1, z' as we go backward
    # For t=4, we need V[5, :, :]
    
    # Helper to build spline for a given value vector
    def build_spline(v_vec):
        # Use natural boundary conditions
        # Clip m_grid to avoid issues if m_prime goes out of bounds? 
        # CubicSpline extrapolates. 
        # For m_prime < min(m_grid) or > max(m_grid), it will extrapolate.
        # This is generally acceptable for smooth value functions.
        return CubicSpline(m_grid, v_vec, bc_type='natural')

    # Backward induction
    for t in range(H - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with c=m (suboptimal but valid)
            # We need to break out and handle post-loop
            break

        # Build splines for next period's value functions
        spline_next = []
        for zp in range(2):
            spline_next.append(build_spline(V[t + 1, zp, :]))

        for z in range(2):
            if time.time() > deadline:
                break

            # Pre-compute transition weights
            # transition[z, :] is the probability of next state given current state z
            p0 = transition[z, 0]
            p1 = transition[z, 1]
            
            # Income for next states
            y0 = income[0]
            y1 = income[1]

            # We will solve for c for each m in m_grid
            # To speed up, we can vectorize the objective evaluation?
            # minimize_scalar is scalar. 
            # However, 101 * 5 * 2 = 1010 optimizations.
            # Each optimization takes ~1-5ms. Total ~1-5s. Should be fine.
            
            best_c = np.zeros(M)
            best_v = np.zeros(M)

            for i in range(M):
                m = m_grid[i]
                
                # Define objective function
                # We want to maximize u(c) + beta * E[V_next]
                # minimize_scalar minimizes, so we minimize -objective
                
                def objective(c_val):
                    if c_val <= 1e-12:
                        return np.inf
                    if c_val > m + 1e-10:
                        return np.inf
                        
                    c_clipped = max(c_val, 1e-12)
                    u_c = utility(c_clipped)
                    
                    # Calculate m' for both next states
                    m_prime_0 = R * (m - c_clipped) + y0
                    m_prime_1 = R * (m - c_clipped) + y1
                    
                    # Evaluate continuation values
                    v0 = spline_next[0](m_prime_0)
                    v1 = spline_next[1](m_prime_1)
                    
                    e_v = p0 * v0 + p1 * v1
                    
                    return -(u_c + beta * e_v)

                # Bounds for c: [epsilon, m]
                # Use a small epsilon to avoid log(0)
                lower = 1e-8
                upper = m
                
                # Use bounded method
                # xatol controls the precision
                res = minimize_scalar(objective, bounds=(lower, upper), method='bounded', options={'xatol': 1e-8})
                
                best_c[i] = res.x
                best_v[i] = -res.fun

            C[t, z, :] = best_c
            V[t, z, :] = best_v

    # Post-loop validity check and fallback
    # If we broke out early, some C[t, z, :] might be 0.
    # Also, ensure c > 0 and c <= m.
    for t in range(H):
        for z in range(2):
            # Check for zeros or invalid values
            # If any value is 0 or <= 0, it's invalid.
            # If any value is > m_grid + 1e-10, it's invalid.
            
            # We can vectorize this check
            # But since we might have broken early, C[t] might be all 0.
            # Let's check if the row is all zeros.
            if np.all(C[t, z, :] == 0):
                C[t, z, :] = m_grid
            else:
                # Ensure constraints
                # Clip to [1e-12, m_grid]
                # np.clip works element-wise
                C[t, z, :] = np.clip(C[t, z, :], 1e-12, m_grid)
                
                # Re-check if any are 0 (shouldn't be with clip, but just in case)
                # And if any exceed m_grid (shouldn't be with clip)
                
                # One more check: if the clip resulted in 0 (if m_grid had 0, which it doesn't, min is 0.001)
                # So C[t, z, :] should be valid now.
                
                # However, if we broke out early, the V values for previous steps might be inconsistent 
                # with the C values if we just filled C with m_grid.
                # But the task only returns C. The scoring uses C to calculate value.
                # So V consistency is not required for the output, only C validity.

    return C
# EVOLVE-BLOCK-END
