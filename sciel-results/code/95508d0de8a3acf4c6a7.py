import numpy as np
import scipy.optimize as opt
import time

def construct():
    start_time = time.time()
    deadline = 205.0  # Safety margin before 210s limit
    
    # Precompute Gauss-Hermite quadrature nodes and weights
    # For physicists' Hermite polynomials, the weight is exp(-x^2)
    # n points integrate exactly polynomials up to degree 2n-1
    # Our integrands have max degree 26 (x^2 * P^2 where deg P = 12)
    # So we need 2n-1 >= 26 => n >= 13.5 => n = 14
    n_gauss = 15  # Use 15 for safety
    nodes, weights = np.polynomial.hermite.hermgauss(n_gauss)
    
    # Precompute polynomial basis values at nodes
    # H0, H4, H8, H12 at each node
    x_nodes = nodes
    
    def H0_vals(x):
        return np.ones_like(x, dtype=float)
    
    def H4_vals(x):
        return x**4 - 6*x**2 + 3
    
    def H8_vals(x):
        return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    
    def H12_vals(x):
        return x**12 - 66*x**10 + 1260*x**8 - 9240*x**6 + 23100*x**4 - 27720*x**2 + 10395
    
    # Precompute basis values and derivatives at Gauss nodes
    H0_at_nodes = H0_vals(x_nodes)
    H4_at_nodes = H4_vals(x_nodes)
    H8_at_nodes = H8_vals(x_nodes)
    H12_at_nodes = H12_vals(x_nodes)
    
    # Derivatives
    dH0_at_nodes = np.zeros_like(x_nodes)
    dH4_at_nodes = 4*x_nodes**3 - 12*x_nodes
    dH8_at_nodes = 8*x_nodes**7 - 168*x_nodes**5 + 840*x_nodes**3 - 840*x_nodes
    dH12_at_nodes = 12*x_nodes**11 - 660*x_nodes**9 + 10080*x_nodes**7 - 55440*x_nodes**5 + 92400*x_nodes**3 - 55440*x_nodes
    
    # Precompute x^2 at nodes
    x2_at_nodes = x_nodes**2
    
    # Precompute (x * H) values needed for (P' - x*P)
    # We need: sum c_i * (dH_i - x * H_i) at each node
    # Let's precompute (dH_i - x*H_i) for each basis polynomial
    # For H0: dH0 - x*H0 = 0 - x = -x
    # For H4: dH4 - x*H4
    # For H8: dH8 - x*H8
    # For H12: dH12 - x*H12
    
    dH0_minus_xH0 = dH0_at_nodes - x_nodes * H0_at_nodes  # = -x
    dH4_minus_xH4 = dH4_at_nodes - x_nodes * H4_at_nodes
    dH8_minus_xH8 = dH8_at_nodes - x_nodes * H8_at_nodes
    dH12_minus_xH12 = dH12_at_nodes - x_nodes * H12_at_nodes
    
    def compute_uncertainty_product(coeffs):
        c0, c4, c8 = coeffs
        
        # Check time
        if time.time() > deadline - 3:
            return np.inf
        
        try:
            # c12 from constraint P(0) = 0
            c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
            
            # P(x) at nodes: c0*H0 + c4*H4 + c8*H8 + c12*H12
            P_at_nodes = c0 * H0_at_nodes + c4 * H4_at_nodes + c8 * H8_at_nodes + c12 * H12_at_nodes
            
            # (P' - x*P) at nodes: c0*(dH0-xH0) + c4*(dH4-xH4) + c8*(dH8-xH8) + c12*(dH12-xH12)
            Q_at_nodes = c0 * dH0_minus_xH0 + c4 * dH4_minus_xH4 + c8 * dH8_minus_xH8 + c12 * dH12_minus_xH12
            
            # I0 = sum w_i * P(x_i)^2
            I0 = np.sum(weights * P_at_nodes**2)
            if I0 <= 1e-30 or not np.isfinite(I0):
                return np.inf
            
            # Ix2 = sum w_i * x_i^2 * P(x_i)^2
            Ix2 = np.sum(weights * x2_at_nodes * P_at_nodes**2)
            if not np.isfinite(Ix2) or Ix2 < 0:
                return np.inf
            
            # Ideriv = sum w_i * Q(x_i)^2
            Ideriv = np.sum(weights * Q_at_nodes**2)
            if not np.isfinite(Ideriv) or Ideriv < 0:
                return np.inf
            
            # A(f)^2 = Ix2 / I0
            # A(hat{f})^2 = (1/(2*pi)^2) * Ideriv / I0
            # Product = sqrt( (Ix2/I0) * (Ideriv/I0) / (4*pi^2) )
            
            A_f_sq = Ix2 / I0
            A_gf_sq = (1.0 / (4.0 * np.pi**2)) * (Ideriv / I0)
            
            return np.sqrt(A_f_sq * A_gf_sq)
        except Exception:
            return np.inf
    
    # Initial guess from seed (Goncalves et al. 2017)
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_score = compute_uncertainty_product(x0)
    best_coeffs = x0.copy()
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Local optimization from seed with tight tolerances
    if time.time() < deadline - 15:
        try:
            res = opt.minimize(compute_uncertainty_product, x0, method='Nelder-Mead', 
                              options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 2000, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Random restarts - can do more since evaluation is now fast
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # Generate a starting point
        if i < 10:
            # Perturb around best
            scale = 0.05 * (1 + i/10.0)
            x_start = best_coeffs + rng.normal(0, scale, 3)
        elif i < 30:
            # Perturb around seed
            scale = 0.1
            x_start = x0 + rng.normal(0, scale, 3)
        else:
            # Larger exploration
            scale = 0.3
            x_start = best_coeffs + rng.normal(0, scale, 3)
            
        try:
            res = opt.minimize(compute_uncertainty_product, x_start, method='Nelder-Mead', 
                              options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 1000, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(compute_uncertainty_product, best_coeffs, method='Nelder-Mead', 
                              options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 5000, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: One more polish if time remains
    if time.time() < deadline - 2:
        try:
            res = opt.minimize(compute_uncertainty_product, best_coeffs, method='Nelder-Mead', 
                              options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 2000, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_coeffs)
