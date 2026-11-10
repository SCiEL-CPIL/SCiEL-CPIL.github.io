import numpy as np
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    N_target = len(resource_grid)
    T = horizon
    Z = len(income)
    
    # Use a finer grid for internal calculations to improve interpolation accuracy
    M = 1000
    resource_fine = np.geomspace(0.001, 40, M)
    
    # Policy array on fine grid: c[t, z, i]
    c_policy_fine = np.zeros((T, Z, M))
    
    # Last period: consume everything
    c_policy_fine[T-1, :, :] = resource_fine[np.newaxis, np.newaxis, :]
    
    def util_prime(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return 1.0 / c
        else:
            return c ** (-gamma)
            
    def util_double_prime(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return -1.0 / (c * c)
        else:
            return -gamma * c ** (-gamma - 1)

    # Precompute interpolators for the last period (t = T-1)
    # We need c_next and dc/dm_next
    # For t = T-1, c(m) = m, so dc/dm = 1
    # But we will update interpolators as we go backward.
    
    # Initialize interpolators for t = T-1
    # c_next[z] is resource_fine for all z
    # dc_dm_next[z] is 1 for all z
    
    # Backward induction
    for t in range(T-2, -1, -1):
        c_next = c_policy_fine[t+1]
        
        # Create PCHIP interpolators for c_next and its derivative
        # We need the derivative of c_next. 
        # Instead of differentiating the interpolant, we can compute the derivative of the policy function.
        # However, the policy function is discrete. 
        # A common trick is to use the derivative of the PCHIP interpolant.
        # PchipInterpolator has a method to get the derivative.
        
        interp_c = np.zeros((Z, M))
        interp_dc = np.zeros((Z, M))
        
        for zp in range(Z):
            # PCHIP is good for monotonic data, which consumption policies usually are
            pchip_c = PchipInterpolator(resource_fine, c_next[zp])
            # Evaluate derivative
            interp_c[zp] = c_next[zp] # We are evaluating at the grid points, so no interp needed for values at grid
            # But we need derivatives at the grid points to start Newton? 
            # Actually, we need to evaluate the continuation value at m' which is NOT on the grid.
            # So we need the interpolators themselves, not just values at grid points.
            
        # Let's store the interpolators or just re-create them if slow? 
        # Creating 2 * Z interpolators per step is okay for M=1000, T=6, Z=2.
        
        # We will store the interpolators in a list
        interpolators = []
        for zp in range(Z):
            pchip_c = PchipInterpolator(resource_fine, c_next[zp])
            # Derivative of PCHIP
            pchip_dc = pchip_c.derivative()
            interpolators.append((pchip_c, pchip_dc))

        for z in range(Z):
            m_vals = resource_fine
            
            # Initial guess: use the next period's policy for this state z
            # Clip to valid range
            c_curr = np.clip(c_next[z], 1e-12, m_vals)
            
            # Newton Iterations
            for _ in range(20):
                # Calculate m' for each z'
                m_prime = R * (m_vals[np.newaxis, :] - c_curr[np.newaxis, :]) + income[:, np.newaxis]
                
                # Clip m_prime to grid range for interpolation safety
                # PCHIP can extrapolate, but it's safer to clip to [min, max] of grid
                # Actually, m' can be outside [0.001, 40]. 
                # If m' < 0.001, c' should be small. If m' > 40, c' should be large.
                # PCHIP extrapolates linearly by default? No, it uses the end slopes.
                # Let's just use the interpolators.
                
                c_prime = np.zeros((Z, M))
                dc_prime_dm = np.zeros((Z, M))
                
                for zp in range(Z):
                    pchip_c, pchip_dc = interpolators[zp]
                    c_prime[zp] = pchip_c(m_prime[zp])
                    dc_prime_dm[zp] = pchip_dc(m_prime[zp])
                
                # Calculate f(c) and f'(c)
                # f(c) = U'(c) - beta * sum_z' P(z'|z) * R * U'(c')
                up_c = util_prime(c_curr)
                
                exp_term = np.zeros(M)
                exp_term_dd = np.zeros(M)
                
                for zp in range(Z):
                    prob = transition[z, zp]
                    if prob < 1e-15:
                        continue
                    up_cp = util_prime(c_prime[zp])
                    d2up_cp = util_double_prime(c_prime[zp])
                    
                    exp_term += prob * R * up_cp
                    # Derivative term contribution:
                    # d/dc [ R * U'(c') ] = R * U''(c') * (dc'/dc)
                    # dc'/dc = -R * (dc'/dm')
                    # So term is -R^2 * U''(c') * (dc'/dm')
                    exp_term_dd += prob * (-R**2) * d2up_cp * dc_prime_dm[zp]
                
                f_val = up_c - beta * exp_term
                f_prime_val = util_double_prime(c_curr) - beta * exp_term_dd
                
                # Newton step with safeguarding
                # Ensure f_prime_val is not zero
                f_prime_val = np.where(np.abs(f_prime_val) < 1e-10, 1e-10 * np.sign(f_prime_val), f_prime_val)
                f_prime_val = np.where(f_prime_val == 0, 1e-10, f_prime_val)
                
                delta = f_val / f_prime_val
                c_new = c_curr - delta
                
                # Clip to valid range [1e-12, m]
                c_new = np.clip(c_new, 1e-12, m_vals)
                
                # Check convergence
                if np.max(np.abs(c_new - c_curr)) < 1e-10:
                    c_curr = c_new
                    break
                c_curr = c_new
                
            c_policy_fine[t, z, :] = c_curr
            
    # Interpolate back to the target grid
    c_policy_target = np.zeros((T, Z, N_target))
    for t in range(T):
        for z in range(Z):
            # Linear interpolation is sufficient for the final output
            c_policy_target[t, z, :] = np.interp(resource_grid, resource_fine, c_policy_fine[t, z, :])
            
    return c_policy_target
