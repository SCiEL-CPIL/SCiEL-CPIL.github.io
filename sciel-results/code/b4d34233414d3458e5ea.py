"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Coefficients for Hermite polynomials H0, H4, H8, H12
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 4608x^6 + 20736x^4 - 20736x^2 + 3024
# H12(x) = 4096x^12 - 147456x^10 + 1658880x^8 - 6635520x^6 + 8028192x^4 - 2677104x^2 + 1032192

def compute_value(c0, c1, c2):
    # c3 is determined by g(0)=0
    c3 = -(c0 * 1.0 + c1 * 12.0 + c2 * 3024.0) / 1032192.0
    
    # If leading coefficient is negative, flip signs
    if c3 < 0:
        c0, c1, c2, c3 = -c0, -c1, -c2, -c3

    # Coefficients of P(y) where y = x^2
    # P(y) = a5 y^5 + a4 y^4 + a3 y^3 + a2 y^2 + a1 y + a0
    a5 = 4096.0 * c3
    a4 = -147456.0 * c3
    a3 = 256.0 * c2 + 1658880.0 * c3
    a2 = -4608.0 * c2 - 6635520.0 * c3
    a1 = 16.0 * c1 + 20736.0 * c2 + 8028192.0 * c3
    a0 = -48.0 * c1 - 20736.0 * c2 - 2677104.0 * c3

    coeffs = [a5, a4, a3, a2, a1, a0]
    
    # Find roots of P(y)
    try:
        roots = np.roots(coeffs)
    except Exception:
        return 1e100

    # Filter real positive roots
    real_roots = roots[np.isreal(roots)].real
    pos_roots = real_roots[real_roots > 1e-9]
    
    if len(pos_roots) == 0:
        return 1e100

    # Sort in descending order to find the largest
    pos_roots_sorted = np.sort(pos_roots)[::-1]
    
    for y in pos_roots_sorted:
        # Check for sign change
        # Use a small epsilon relative to y
        eps = 1e-5 * max(1.0, abs(y))
        y_minus = y - eps
        y_plus = y + eps
        
        # Evaluate P at y_minus and y_plus
        p_minus = np.polyval(coeffs, y_minus)
        p_plus = np.polyval(coeffs, y_plus)
        
        if p_minus * p_plus < 0:
            r = np.sqrt(y)
            return r**2 / (2 * np.pi)

    return 1e100

def construct():
    start_time = time.time()
    deadline = start_time + 180.0
    
    # Seed values
    c0_seed = -113.0 / 100.0
    c1_seed = 1.0 / 25.0
    c2_seed = 1.0 / 3240.0
    
    best_params = np.array([c0_seed, c1_seed, c2_seed])
    best_val = compute_value(best_params[0], best_params[1], best_params[2])
    
    # Multi-start local optimization
    # Generate starting points around the seed
    np.random.seed(42)
    starts = [best_params]
    
    # Add perturbed points
    for _ in range(20):
        noise = np.random.normal(0, 0.1, 3)
        start = best_params + noise
        # Clip to reasonable bounds to avoid extreme values
        start[0] = np.clip(start[0], -10, 10)
        start[1] = np.clip(start[1], -10, 10)
        start[2] = np.clip(start[2], -1, 1)
        starts.append(start)
        
        # Check time
        if time.time() > deadline - 30.0:
            break

    # Optimize from each start
    for start in starts:
        if time.time() > deadline - 10.0:
            break
            
        try:
            result = minimize(
                lambda x: compute_value(x[0], x[1], x[2]),
                start,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass

    return best_params
# EVOLVE-BLOCK-END
