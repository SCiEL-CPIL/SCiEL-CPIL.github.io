import numpy as np
import time
from scipy.optimize import minimize

# Precompute grid
# Use a sufficiently fine grid to capture the min of |hat_g|
# The functions are smooth, so 4096 points should be enough for high accuracy
N_GRID = 4096
x = np.linspace(-10, 10, N_GRID)
dx = x[1] - x[0]
gauss = np.exp(-x**2 / 2)

# Probabilists' Hermite polynomials
# He_0 = 1
# He_4 = x^4 - 6x^2 + 3
# He_8 = x^8 - 28x^6 + 420x^4 - 840x^2 + 105
# He_12 = x^12 - 66x^10 + 2310x^8 - 39900x^6 + 332640x^4 - 798336x^2 + 10395

# Vectorized evaluation
x2 = x**2
x4 = x2**2
x6 = x4 * x2
x8 = x4**2
x10 = x8 * x2
x12 = x6**2

H0 = np.ones_like(x)
H4 = x4 - 6*x2 + 3
H8 = x8 - 28*x6 + 420*x4 - 840*x2 + 105
H12 = x12 - 66*x10 + 2310*x8 - 39900*x6 + 332640*x4 - 798336*x2 + 10395

# Values at 0 for constraint
H0_0 = 1.0
H4_0 = 3.0
H8_0 = 105.0
H12_0 = 10395.0

def c3_from(c0, c1, c2):
    return -(c0 + H4_0*c1 + H8_0*c2) / H12_0

def soft_min(arr, temperature=1e-3):
    """Smooth approximation of min using log-sum-exp trick"""
    neg_arr = -arr
    max_val = np.max(neg_arr)
    # Handle case where all are very large negative (unlikely here)
    if max_val < -100: 
        return 0.0
    return -max_val - temperature * np.log(np.sum(np.exp((neg_arr - max_val) / temperature)))

def objective_smooth(coeffs, temperature=1e-3):
    """
    coeffs: [c0, c4, c8]
    Returns smooth approximation of the bound.
    """
    c0, c1, c2 = coeffs
    c3 = c3_from(c0, c1, c2)
    
    # g(x) = P(x) * gauss
    P = c0*H0 + c1*H4 + c2*H8 + c3*H12
    g = P * gauss
    
    # L1 norm
    l1 = np.sum(np.abs(g)) * dx
    
    # Fourier transform
    # For probabilists' Hermite polynomials, the FT of He_n(x)e^{-x^2/2} is 
    # sqrt(2pi) * (i)^n * He_n(x) e^{-x^2/2}.
    # Since n=0,4,8,12 are multiples of 4, (i)^n = 1.
    # So hat_g = sqrt(2pi) * P * gauss.
    hat_g = np.sqrt(2*np.pi) * P * gauss
    
    abs_hat = np.abs(hat_g)
    min_abs_hat = soft_min(abs_hat, temperature)
    
    if min_abs_hat < 1e-10:
        return 1e10
        
    return l1 / min_abs_hat

def objective_exact(coeffs):
    """Exact objective for final evaluation"""
    c0, c1, c2 = coeffs
    c3 = c3_from(c0, c1, c2)
    
    P = c0*H0 + c1*H4 + c2*H8 + c3*H12
    g = P * gauss
    l1 = np.sum(np.abs(g)) * dx
    
    hat_g = np.sqrt(2*np.pi) * P * gauss
    abs_hat = np.abs(hat_g)
    min_abs_hat = np.min(abs_hat)
    
    if min_abs_hat < 1e-10:
        return 1e10
        
    return l1 / min_abs_hat

def fibonacci_sphere(N):
    """Generate N points on the unit sphere using Fibonacci lattice."""
    points = np.zeros((N, 3))
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(N):
        y = 1 - 2 * (i + 0.5) / N
        radius = np.sqrt(1 - y**2)
        phi = i * golden_angle
        points[i, 0] = radius * np.cos(phi)
        points[i, 1] = y
        points[i, 2] = radius * np.sin(phi)
    return points

def construct():
    t0 = time.time()
    deadline = t0 + 195  # Safety margin
    
    # 1. Global Search on the Sphere
    # The objective is scale invariant, so we can fix ||(c0, c4, c8)|| = 1
    # This maps to a point on the sphere S^2.
    N_POINTS = 2000
    sphere_points = fibonacci_sphere(N_POINTS)
    
    best_val = float('inf')
    best_coeffs = np.array([1.0, 0.0, 0.0]) # Default
    
    # Evaluate all points
    # We can vectorize the L1 part, but the min part is per-point.
    # However, N_POINTS=2000 and N_GRID=4096 is small enough to loop or use advanced indexing.
    # Let's loop for simplicity and safety, it's fast.
    
    start_search = time.time()
    
    for i in range(N_POINTS):
        if time.time() > deadline - 10:
            break
        c = sphere_points[i]
        val = objective_smooth(c)
        if val < best_val:
            best_val = val
            best_coeffs = c.copy()
            
    # 2. Local Refinement from the best global point
    # Also try a few other good points to be safe
    # Sort the points by objective value? We didn't store them. Let's just take the best one.
    # To be more robust, let's re-evaluate a subset or just proceed with the best.
    
    # Let's add a few random starts as well, in case the lattice missed the basin
    rng = np.random.default_rng(42)
    extra_starts = [best_coeffs]
    for _ in range(10):
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v)
        extra_starts.append(v)
        
    # Also include the seed from the original program, normalized
    seed = np.array([-113/100, 1/25, 1/3240])
    seed_norm = seed / np.linalg.norm(seed)
    extra_starts.append(seed_norm)
    
    best_exact_val = float('inf')
    best_exact_coeffs = best_coeffs
    
    for start in extra_starts:
        if time.time() > deadline - 15:
            break
        try:
            # Use BFGS for better convergence
            res = minimize(objective_smooth, start, method='BFGS', 
                           options={'maxiter': 500, 'gtol': 1e-8})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
                # Update best exact immediately to track
                val_exact = objective_exact(best_coeffs)
                if val_exact < best_exact_val:
                    best_exact_val = val_exact
                    best_exact_coeffs = best_coeffs.copy()
        except Exception:
            pass
            
    # 3. Final Polish with Exact Objective
    # The smooth objective might drift slightly from the true minimum due to temperature.
    # Use the best candidate from the smooth search to polish with the exact objective.
    # Since exact objective is non-smooth, use Nelder-Mead or just accept the smooth result if it's good.
    # Nelder-Mead is robust to non-smoothness.
    
    if time.time() < deadline - 5:
        try:
            # Start from the best smooth solution
            res = minimize(objective_exact, best_coeffs, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_exact_val:
                best_exact_val = res.fun
                best_exact_coeffs = res.x.copy()
        except Exception:
            pass
            
    # Return the coefficients
    # Note: The scale doesn't matter for the score, but we should return a normalized or reasonable set.
    # The task asks for "three coefficients". Any scalar multiple is valid.
    # We will return the ones from the best_exact_coeffs.
    
    return np.array(best_exact_coeffs)
