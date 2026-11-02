"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

def _hermite_polynomials(x):
    """
    Computes H0, H4, H8, H12 for input x.
    Uses explicit formulas for speed and accuracy.
    H0(x) = 1
    H4(x) = x^4 - 6x^2 + 3
    H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    H12(x) = x^12 - 54x^10 + 1080x^8 - 7560x^6 + 15120x^4 - 7560x^2 + 945
    """
    x2 = x**2
    x4 = x2**2
    x6 = x4 * x2
    x8 = x4**2
    x10 = x8 * x2
    x12 = x8 * x4
    
    H0 = np.ones_like(x)
    H4 = x4 - 6*x2 + 3
    H8 = x8 - 28*x6 + 210*x4 - 420*x2 + 105
    H12 = x12 - 54*x10 + 1080*x8 - 7560*x6 + 15120*x4 - 7560*x2 + 945
    
    return H0, H4, H8, H12

def _compute_objective(coeffs):
    """
    Computes the uncertainty product bound for the given coefficients.
    coeffs: [a0, a4, a8]
    Returns the value of J = L4_4 / L2_sq^3
    """
    a0, a4, a8 = coeffs
    
    # Enforce g(0) = 0 to find a12
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=945
    # a0*1 + a4*3 + a8*105 + a12*945 = 0
    a12 = -(a0 + 3*a4 + 105*a8) / 945.0
    
    # Setup Gauss-Hermite Quadrature with high precision
    # For degree-48 polynomial in L4, we need N >= 25, use N=80 for safety
    N = 80
    nodes, weights = np.polynomial.hermite.hermgauss(N)
    
    # Evaluate P(x) at nodes for L2 (x = nodes)
    H0, H4, H8, H12 = _hermite_polynomials(nodes)
    P_vals = a0*H0 + a4*H4 + a8*H8 + a12*H12
    
    L2_sq = np.sum(weights * P_vals**2)
    
    # Evaluate P(x) at nodes for L4 (x = nodes/sqrt(2))
    x_scaled = nodes / np.sqrt(2.0)
    H0_s, H4_s, H8_s, H12_s = _hermite_polynomials(x_scaled)
    P_vals_s = a0*H0_s + a4*H4_s + a8*H8_s + a12*H12_s
    
    # Integral of |P|^4 e^{-2x^2} dx
    # = 1/sqrt(2) * sum w_i |P(x_i)|^4
    L4_4 = (1.0/np.sqrt(2.0)) * np.sum(weights * P_vals_s**4)
    
    # Avoid division by zero
    if L2_sq < 1e-15:
        return 1e15
        
    # Objective: minimize L4_4 / L2_sq^3
    return L4_4 / (L2_sq**3)

def _compute_objective_high_precision(coeffs):
    """
    High-precision version using more quadrature nodes.
    """
    a0, a4, a8 = coeffs
    
    # Enforce g(0) = 0 to find a12
    a12 = -(a0 + 3*a4 + 105*a8) / 945.0
    
    # Use higher precision quadrature
    N = 150
    nodes, weights = np.polynomial.hermite.hermgauss(N)
    
    # Evaluate P(x) at nodes for L2 (x = nodes)
    H0, H4, H8, H12 = _hermite_polynomials(nodes)
    P_vals = a0*H0 + a4*H4 + a8*H8 + a12*H12
    
    L2_sq = np.sum(weights * P_vals**2)
    
    # Evaluate P(x) at nodes for L4 (x = nodes/sqrt(2))
    x_scaled = nodes / np.sqrt(2.0)
    H0_s, H4_s, H8_s, H12_s = _hermite_polynomials(x_scaled)
    P_vals_s = a0*H0_s + a4*H4_s + a8*H8_s + a12*H12_s
    
    # Integral of |P|^4 e^{-2x^2} dx
    L4_4 = (1.0/np.sqrt(2.0)) * np.sum(weights * P_vals_s**4)
    
    # Avoid division by zero
    if L2_sq < 1e-15:
        return 1e15
        
    # Objective: minimize L4_4 / L2_sq^3
    return L4_4 / (L2_sq**3)

def construct():
    start_time = time.time()
    deadline = start_time + 180.0 # 180s limit, keep margin
    
    # Seed solution
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _compute_objective(seed)
    best_x = seed.copy()
    
    # Multi-start optimization with diverse starting points
    rng = np.random.default_rng(42)
    
    # Create diverse starting points
    starts = [seed]
    
    # Random starts around seed with various scales
    for scale in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
        for _ in range(3):
            x0 = seed + rng.normal(0, scale, 3)
            starts.append(x0)
    
    # Pure random starts in larger range
    for _ in range(15):
        x0 = rng.uniform(-10, 10, 3)
        starts.append(x0)
    
    # Additional structured starts
    for _ in range(10):
        # Try combinations that emphasize different Hermite components
        a0 = rng.uniform(-5, 5)
        a4 = rng.uniform(-1, 1)
        a8 = rng.uniform(-0.1, 0.1)
        x0 = np.array([a0, a4, a8])
        starts.append(x0)
    
    # Run optimization from each starting point
    for x0 in starts:
        if time.time() > deadline - 15.0:
            break
            
        try:
            # Use Nelder-Mead for robustness
            res = opt.minimize(_compute_objective, x0, method='Nelder-Mead', 
                               options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 300, 'adaptive': True})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            continue
    
    # Refine the best solution with high precision and tighter tolerances
    if time.time() < deadline - 10.0:
        try:
            # First refinement with standard precision
            res = opt.minimize(_compute_objective, best_x, method='Nelder-Mead',
                               options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 500, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            res = opt.minimize(_compute_objective_high_precision, best_x, method='Nelder-Mead',
                               options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 200, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x

# EVOLVE-BLOCK-END
