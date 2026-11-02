"""Chebyshev collocation approach for Hermite-Gaussian uncertainty bound optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """
    Find optimal coefficients (c0, c4, c8) for the Hermite-Gaussian test function
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is determined by g(0)=0.
    
    Uses Chebyshev collocation for accurate supremum computation and
    multi-start Powell optimization for the coefficient search.
    """
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s limit

    # Pre-compute constants for Hermite polynomials at x=0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 47900160.0  # H12(0) = (-1)^6 * 12! / 6! = 47900160

    # Chebyshev collocation grid on [-L, L]
    # The function decays as exp(-x^2/2), so L=8 captures >99.9999% of the mass
    L = 8.0
    N = 2048  # Number of Chebyshev points
    
    # Chebyshev-Gauss-Lobatto points mapped to [-L, L]
    k = np.arange(N)
    theta = np.pi * k / (N - 1)
    x = -L * np.cos(theta)
    
    # Pre-compute exp(-x^2/2)
    exp_factor = np.exp(-x**2 / 2.0)
    
    # Pre-compute Hermite polynomials on the Chebyshev grid
    x2 = x * x
    H0 = np.ones_like(x)
    H4 = 16.0*x2*x2 - 48.0*x2 + 12.0
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    H8 = 256.0*x8 - 4480.0*x6 + 26880.0*x4 - 6720.0*x2 + 1680.0
    x10 = x8 * x2
    x12 = x6 * x6
    # Physicists' Hermite H12(x)
    H12 = (4096.0*x12 - 129024.0*x10 + 1761280.0*x8 - 11942400.0*x6 
           + 37158400.0*x4 - 54432000.0*x2 + 47900160.0)
    
    # Pre-compute basis functions
    # g(x) = exp(-x^2/2) * [ c0*(H0 - H0_0/H12_0*H12) + c4*(H4 - H4_0/H12_0*H12) + c8*(H8 - H8_0/H12_0*H12) ]
    
    basis0 = (H0 - (H0_0/H12_0)*H12) * exp_factor
    basis4 = (H4 - (H4_0/H12_0)*H12) * exp_factor
    basis8 = (H8 - (H8_0/H12_0)*H12) * exp_factor
    
    # We want to minimize (sup |x * g(x)|)^2
    xb0 = x * basis0
    xb4 = x * basis4
    xb8 = x * basis8

    def objective(coeffs):
        """
        Compute the squared supremum of |x * g(x)|.
        coeffs: [c0, c4, c8]
        """
        c0, c4, c8 = coeffs
        gx = c0*xb0 + c4*xb4 + c8*xb8
        max_val = np.max(np.abs(gx))
        return max_val * max_val

    # Initial guess based on known good values
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = objective(x0)
    best_x = x0.copy()
    
    # Multi-start Powell optimization
    # Generate multiple starting points around the initial guess
    np.random.seed(42)
    
    # Starting points: initial guess + perturbations
    start_points = [x0.copy()]
    
    # Add perturbations at various scales
    for scale in [0.1, 0.05, 0.01, 0.001]:
        for _ in range(3):
            perturbation = np.random.randn(3) * scale
            start_points.append(x0 + perturbation)
    
    # Add some structured starts
    start_points.append(np.array([-1.2, 0.03, 0.0002]))
    start_points.append(np.array([-1.1, 0.04, 0.0004]))
    start_points.append(np.array([-1.15, 0.035, 0.0003]))
    
    # Run Powell from each starting point
    for i, sp in enumerate(start_points):
        if time.time() > deadline - 5:
            break
        
        try:
            res = minimize(objective, sp, method='Powell', 
                          options={'maxiter': 300, 'ftol': 1e-14, 'xtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish with Nelder-Mead
    try:
        if time.time() < deadline - 3:
            res_nm = minimize(objective, best_x, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 1e-15, 'fatol': 1e-15})
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_x = res_nm.x.copy()
    except Exception:
        pass
    
    return best_x

# EVOLVE-BLOCK-END
