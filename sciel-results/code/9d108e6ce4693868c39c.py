import numpy as np
from scipy.interpolate import CubicSpline
import time
import scipy.optimize as opt

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_grid = len(resource_grid)
    n_z = len(income)
    
    def crra(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
            
    def crra_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        else:
            return c ** (-gamma)
            
    # Fine grid for internal computation
    # 1000 points should be very accurate and fast with Newton's method
    N_fine = 1000
    fine_grid = np.geomspace(0.001, 40.0, N_fine)
    
    # V[t, z, m_idx]
    V = np.zeros((horizon, n_z, N_fine))
    C = np.zeros((horizon, n_z, N_fine))
    
    # Last period: consume all
    for z in range(n_z):
        V[horizon-1, z, :] = crra(fine_grid)
        C[horizon-1, z, :] = fine_grid
        
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        # Prepare splines for next period
        splines = []
        spline_derivs = []
        for z in range(n_z):
            s = CubicSpline(fine_grid, V[t+1, z, :])
            splines.append(s)
            spline_derivs.append(s.derivative())
            
        for z in range(n_z):
            for m_idx, m in enumerate(fine_grid):
                # Define the function to find the root of: J'(c) = 0
                # J'(c) = u'(c) - beta * R * sum_{z'} P(z'|z) * V'_{t+1}(z', R(m-c) + y_{z'})
                
                def euler_residual(c):
                    up = crra_prime(c)
                    cont_deriv = 0.0
                    for zp in range(n_z):
                        m_next = R * (m - c) + income[zp]
                        vp = spline_derivs[zp](m_next)
                        cont_deriv += transition[z, zp] * vp
                    return up - beta * R * cont_deriv
                
                # Newton's method with bisection fallback
                c_low = 1e-12
                c_high = m
                
                # Initial guess: use previous m_idx if available, else 0.5*m
                if m_idx > 0:
                    c_init = C[t, z, m_idx-1]
                    c_init = np.clip(c_init, c_low, c_high)
                else:
                    c_init = 0.5 * m
                
                # Newton iterations
                c_opt = c_init
                for _ in range(20):
                    f_val = euler_residual(c_opt)
                    
                    # Check boundaries
                    if f_val <= 0:
                        c_opt = c_low
                        break
                    elif f_val >= 0 and c_opt >= c_high - 1e-10:
                        c_opt = c_high
                        break
                    
                    # Numerical derivative
                    h = max(1e-8, abs(c_opt) * 1e-5)
                    c_plus = min(c_opt + h, c_high)
                    c_minus = max(c_opt - h, c_low)
                    if c_plus == c_minus:
                        break
                    f_plus = euler_residual(c_plus)
                    f_minus = euler_residual(c_minus)
                    f_prime = (f_plus - f_minus) / (c_plus - c_minus)
                    
                    if abs(f_prime) < 1e-15:
                        break
                    
                    c_new = c_opt - f_val / f_prime
                    c_new = np.clip(c_new, c_low, c_high)
                    
                    if abs(c_new - c_opt) < 1e-12:
                        c_opt = c_new
                        break
                    c_opt = c_new
                
                # Final check and clip
                c_opt = np.clip(c_opt, c_low, c_high)
                
                C[t, z, m_idx] = c_opt
                
                # Compute Value
                val = crra(c_opt)
                for zp in range(n_z):
                    m_next = R * (m - c_opt) + income[zp]
                    v_next = splines[zp](m_next)
                    val += beta * transition[z, zp] * v_next
                V[t, z, m_idx] = val
                
        # Check time
        if time.time() > deadline:
            # If we run out of time, we should break and interpolate what we have
            # But we need all t for the output.
            # If we break early, we don't have V for earlier t.
            # However, the loop is from t=4 down to 0.
            # If we break at t=2, we have V[4], V[3], V[2]. We are missing V[1], V[0].
            # We can't just return.
            # Given the speed, this is unlikely.
            pass

    # Interpolate C from fine_grid to resource_grid
    C_out = np.zeros((horizon, n_z, n_grid))
    for t in range(horizon):
        for z in range(n_z):
            C_out[t, z, :] = np.interp(resource_grid, fine_grid, C[t, z, :])
            
    # Ensure c[5] is exactly the grid
    C_out[horizon-1, :, :] = resource_grid
    
    # Final validity check and clipping
    # 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            C_out[t, z, :] = np.maximum(C_out[t, z, :], 1e-12)
            C_out[t, z, :] = np.minimum(C_out[t, z, :], resource_grid + 1e-10)
            
    return C_out
