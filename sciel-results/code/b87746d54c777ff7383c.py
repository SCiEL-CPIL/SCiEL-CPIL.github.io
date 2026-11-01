"""Optimized uncertainty bound using grid search and analytical moments."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 180

# Precompute moments I[2n] = int x^(2n) e^-x^2 dx
def _init_moments():
    moments = np.zeros(25) 
    moments[0] = np.sqrt(np.pi)
    for n in range(1, 13):
        moments[2*n] = moments[2*n-2] * (2*n - 1) / (2*n)
    return moments

MOMENTS = _init_moments()

def _get_poly_coeffs(u, v):
    """
    Returns coefficients c_j for P(x) = H0 + u H4 + v H8 + w H12
    where w is determined by P(0)=0.
    a0 = 1, a1 = u, a2 = v.
    """
    a0 = 1.0
    a1 = u
    a2 = v
    # a3 = -(a0 + 3*a1 + 105*a2) / 10395.0
    a3 = -(1.0 + 3.0*u + 105.0*v) / 10395.0
    
    coeffs = np.zeros(13)
    
    # H0 = 1
    coeffs[0] += a0
    
    # H4 = x^4 - 6x^2 + 3
    coeffs[4] += a1
    coeffs[2] += -6.0 * a1
    coeffs[0] += 3.0 * a1
    
    # H8 = x^8 - 28x^6 + 420x^4 - 1680x^2 + 105
    coeffs[8] += a2
    coeffs[6] += -28.0 * a2
    coeffs[4] += 420.0 * a2
    coeffs[2] += -1680.0 * a2
    coeffs[0] += 105.0 * a2
    
    # H12 = x^12 - 132x^10 + 7920x^8 - 174720x^6 + 1663200x^4 - 4665600x^2 + 10395
    coeffs[12] += a3
    coeffs[10] += -132.0 * a3
    coeffs[8] += 7920.0 * a3
    coeffs[6] += -174720.0 * a3
    coeffs[4] += 1663200.0 * a3
    coeffs[2] += -4665600.0 * a3
    coeffs[0] += 10395.0 * a3
    
    return coeffs

def _integrate_sq(poly_c, weight_power=0):
    total = 0.0
    max_deg = len(poly_c) - 1
    for i in range(max_deg + 1):
        ci = poly_c[i]
        if ci == 0: continue
        for j in range(max_deg + 1):
            cj = poly_c[j]
            if cj == 0: continue
            power = i + j + weight_power
            if power % 2 != 0:
                continue
            if power < len(MOMENTS):
                total += ci * cj * MOMENTS[power]
    return total

def _compute_bound_from_coeffs(coeffs):
    m0 = _integrate_sq(coeffs, 0)
    if m0 <= 0 or not np.isfinite(m0):
        return 1e18
        
    m2 = _integrate_sq(coeffs, 2)
    
    # Q = P' - xP
    p_prime_coeffs = np.zeros(len(coeffs))
    for i in range(1, len(coeffs)):
        p_prime_coeffs[i-1] += i * coeffs[i]
        
    x_p_coeffs = np.zeros(len(coeffs) + 1)
    for i in range(len(coeffs)):
        x_p_coeffs[i+1] += coeffs[i]
        
    q_coeffs = np.zeros(14)
    for i in range(len(p_prime_coeffs)):
        q_coeffs[i] += p_prime_coeffs[i]
    for i in range(len(x_p_coeffs)):
        q_coeffs[i] -= x_p_coeffs[i]
        
    n2 = _integrate_sq(q_coeffs, 0)
    
    if n2 <= 0 or not np.isfinite(n2):
        return 1e18
        
    val = np.sqrt(m2 * n2) / m0
    return val

def _objective_2d(params):
    u, v = params
    coeffs = _get_poly_coeffs(u, v)
    return _compute_bound_from_coeffs(coeffs)

def construct():
    # Grid search strategy
    # u and v ranges
    best_val = 1e18
    best_u = 0.0
    best_v = 0.0
    
    # Coarse grid
    step = 0.5
    u_range = np.arange(-5.0, 5.0 + step, step)
    v_range = np.arange(-5.0, 5.0 + step, step)
    
    for u in u_range:
        if time.time() > _DEADLINE - 30:
            break
        for v in v_range:
            val = _objective_2d((u, v))
            if val < best_val:
                best_val = val
                best_u = u
                best_v = v
                
    # Fine grid around the best point
    if time.time() < _DEADLINE - 10:
        step_fine = 0.05
        u_center = best_u
        v_center = best_v
        u_range_fine = np.arange(u_center - 2.0, u_center + 2.0 + step_fine, step_fine)
        v_range_fine = np.arange(v_center - 2.0, v_center + 2.0 + step_fine, step_fine)
        
        for u in u_range_fine:
            if time.time() > _DEADLINE - 5:
                break
            for v in v_range_fine:
                val = _objective_2d((u, v))
                if val < best_val:
                    best_val = val
                    best_u = u
                    best_v = v
                    
    # Even finer refinement using Nelder-Mead from the best grid point
    if time.time() < _DEADLINE - 2:
        from scipy.optimize import minimize
        try:
            res = minimize(_objective_2d, (best_u, best_v), method='Nelder-Mead', 
                           options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 1000})
            if res.fun < best_val:
                best_val = res.fun
                best_u, best_v = res.x
        except Exception:
            pass

    # Return coefficients a0, a1, a2
    # a0 = 1, a1 = best_u, a2 = best_v
    return np.array([1.0, best_u, best_v])
# EVOLVE-BLOCK-END
