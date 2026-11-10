import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import lpmv, factorial

def rudin_problem10() -> np.ndarray:
    """
    Rudin problem for polynomials on the 2-sphere. Find the real coefficients
    (c_m) for m = -k..k of a real degree-k spherical harmonic

        f = sum_{m=-k}^{k} c_m Y_k^m

    on the round unit sphere S^2 that MINIMIZE the norm ratio

        ||f||_{L^4(S^2)} / ||f||_{L^2(S^2)}

    (norms taken with the standard surface measure dOmega of total mass 4*pi).
    Here the degree is fixed to k = 10, so the returned vector has length 2*k + 1 = 21.

    Strategy:
    1. Precompute the spherical harmonic matrix Y at all quadrature points.
    2. Define the objective function R = ||f||_4 / ||f||_2 using the precomputed matrix.
    3. Use multi-start L-BFGS-B optimization with diverse initializations:
       - Single modes (m=0, m=1, ..., m=10)
       - Sparse random vectors
       - Random dense vectors
    4. Keep track of the best result found within a 600s time budget.
    5. Return the coefficients of the best solution found.
    """
    k = 10
    n_coeffs = 2 * k + 1
    
    # Time management
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin
    
    # --- 1. Precompute Spherical Harmonics at Quadrature Points ---
    # 70-point Gauss-Legendre in cos(theta), 130 equally spaced phi
    # Gauss-Legendre nodes and weights
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    # x_gl is in [-1, 1], which is cos(theta)
    # Weights w_gl are for the integral over [-1, 1]
    # Surface measure dOmega = sin(theta) dtheta dphi = -d(cos(theta)) dphi
    # So the integral over theta becomes integral over x_gl with weights w_gl
    # Total integral over sphere: sum(w_gl) * sum_phi(1) * (1/2) ? No.
    # The Gauss-Legendre quadrature approximates integral_{-1}^{1} f(x) dx.
    # Here x = cos(theta). dOmega = dphi dx.
    # So integral over sphere = sum_i w_i sum_j f(x_i, phi_j)
    # Wait, the standard Gauss-Legendre weights sum to 2.
    # The integral of 1 over the sphere is 4*pi.
    # Integral over phi is 2*pi.
    # Integral over x from -1 to 1 of 1 dx is 2.
    # So sum(w_gl) * 2*pi should be 4*pi. sum(w_gl) is 2. Correct.
    
    # Phi points
    n_phi = 130
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    # Create meshgrid
    x_grid, phi_grid = np.meshgrid(x_gl, phi_vals, indexing='ij')
    # x_grid shape: (70, 130)
    # phi_grid shape: (70, 130)
    
    # Flatten for vectorization
    x_flat = x_grid.flatten()
    phi_flat = phi_grid.flatten()
    w_flat = np.repeat(w_gl, n_phi)  # Each x_i has n_phi points
    
    # Compute Spherical Harmonics Y_m for m = -10..10
    # Y_0 = N_0 P_10^0(cos theta)
    # Y_m = sqrt(2) N_m P_10^m(cos theta) cos(m phi) for m > 0
    # Y_m = sqrt(2) N_|m| P_10^{|m|}(cos theta) sin(|m| phi) for m < 0
    
    N = np.zeros(n_coeffs)
    P = np.zeros((n_coeffs, len(x_flat)))
    
    for m in range(0, k + 1):
        # Normalization N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
        # 21 = (2*10+1)/2
        norm_factor = 21.0 / (4.0 * np.pi) * factorial(k - m) / factorial(k + m)
        N[m] = np.sqrt(norm_factor)
        
        # Associated Legendre polynomial P_10^m(x) with Condon-Shortley phase
        # scipy.special.lpmv(m, l, x) includes the Condon-Shortley phase (-1)^m
        P[m] = lpmv(m, k, x_flat)
        
        if m > 0:
            N[-m] = N[m]  # Symmetric normalization
            P[-m] = P[m]
            
    # Build the Y matrix: shape (n_points, n_coeffs)
    # n_points = 70 * 130 = 9100
    n_points = len(x_flat)
    Y_matrix = np.zeros((n_points, n_coeffs))
    
    for m_idx, m in enumerate(range(-k, k + 1)):
        if m == 0:
            Y_matrix[:, m_idx] = N[0] * P[0]
        elif m > 0:
            Y_matrix[:, m_idx] = np.sqrt(2) * N[m] * P[m] * np.cos(m * phi_flat)
        else:
            abs_m = -m
            Y_matrix[:, m_idx] = np.sqrt(2) * N[abs_m] * P[abs_m] * np.sin(abs_m * phi_flat)
            
    # Weights for integration: w_flat has shape (n_points,)
    # The integral of f^p over the sphere is sum(w_flat * f^p)
    # Note: w_flat sums to 2 * 130 = 260? No.
    # sum(w_gl) = 2. sum(w_flat) = 2 * 130 = 260.
    # The total area is 4*pi.
    # So the discrete integral is sum(w_flat * f) * (2*pi / 130) ? No.
    # The quadrature approximates integral_{-1}^1 integral_0^{2pi} f(x, phi) dphi dx.
    # The phi integral is approximated by (2*pi / 130) * sum_j f(x, phi_j).
    # The x integral is approximated by sum_i w_i f(x_i, ...).
    # So the total integral is sum_i w_i * (2*pi / 130) * sum_j f(x_i, phi_j).
    # In our flattened array, this is sum_p w_flat[p] * (2*pi / 130) * f[p].
    # Let's define effective weights: W_eff = w_flat * (2 * np.pi / n_phi)
    W_eff = w_flat * (2.0 * np.pi / n_phi)
    
    # --- 2. Define Objective Function ---
    def objective(coeffs):
        # f = Y_matrix @ coeffs
        f = Y_matrix @ coeffs
        f2 = f * f
        f4 = f2 * f2
        
        # L2 norm squared: integral f^2
        l2_sq = np.sum(W_eff * f2)
        if l2_sq < 1e-12:
            return 1e10  # Avoid division by zero
        
        # L4 norm: integral f^4
        l4 = np.sum(W_eff * f4)
        
        # Ratio R = ||f||_4 / ||f||_2 = (integral f^4) / (integral f^2)
        # Wait, ||f||_4 = (integral |f|^4)^(1/4). ||f||_2 = (integral |f|^2)^(1/2).
        # R = (integral f^4)^(1/4) / (integral f^2)^(1/2).
        # R^4 = (integral f^4)^1 / (integral f^2)^2.
        # Minimizing R is equivalent to minimizing R^4.
        # Let's minimize R^4 for numerical stability and smoothness.
        return l4 / (l2_sq * l2_sq)
        
    # --- 3. Multi-Start Optimization ---
    best_score = np.inf
    best_coeffs = np.ones(n_coeffs)
    
    # Generate initializations
    inits = []
    
    # Single modes
    for m in range(-k, k + 1):
        init = np.zeros(n_coeffs)
        init[m + k] = 1.0
        inits.append(init)
        
    # Sparse random
    for _ in range(20):
        init = np.random.randn(n_coeffs)
        # Make sparse
        mask = np.random.rand(n_coeffs) < 0.5
        init[~mask] = 0
        if np.linalg.norm(init) > 1e-6:
            init /= np.linalg.norm(init)
        inits.append(init)
        
    # Dense random
    for _ in range(30):
        init = np.random.randn(n_coeffs)
        init /= np.linalg.norm(init)
        inits.append(init)
        
    # All ones (baseline)
    inits.append(np.ones(n_coeffs) / np.sqrt(n_coeffs))
    
    # Shuffle inits
    np.random.shuffle(inits)
    
    # Optimization loop
    for i, x0 in enumerate(inits):
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.status == 0:
                score = res.fun
                if score < best_score:
                    best_score = score
                    best_coeffs = res.x.copy()
                    
        except Exception:
            continue
            
    # --- 4. Validate and Return ---
    # Check validity
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        return np.ones(n_coeffs)
        
    max_c = np.max(np.abs(best_coeffs))
    if max_c < 1e-3 or max_c > 1e5:
        # Scale to be in range, e.g., max|c| = 1
        if max_c > 1e-3:
            best_coeffs /= max_c
        else:
            best_coeffs = best_coeffs / 1e-3 * 1.0
            
    return best_coeffs
