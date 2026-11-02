import numpy as np
import scipy.optimize as opt
import time

def construct():
    start_time = time.time()
    deadline = 190.0  # Safety margin before 210s limit
    
    # Pre-compute Gauss-Hermite nodes and weights
    # Degree of g is 12, so g^2 is degree 24.
    # Gauss-Hermite with N points integrates polynomials up to degree 2N-1 exactly.
    # We need 2N-1 >= 24 => N >= 12.5. 
    # Let's use N=64 for high precision and speed.
    N_points = 64
    x_nodes, w_nodes = np.polynomial.hermite.hermgauss(N_points)
    
    # Hermite polynomials (physicists' convention)
    # H_0(x) = 1
    # H_4(x) = x^4 - 6x^2 + 3
    # H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H_12(x) = x^12 - 66x^10 + 1260x^8 - 9240x^6 + 23100x^4 - 27720x^2 + 10395
    
    # We can evaluate these efficiently using vectorized operations on x_nodes
    # But it's better to define the polynomial evaluation function that takes coeffs and returns values at nodes
    
    def eval_g_and_deriv(coeffs):
        """
        Evaluates g(x) and g'(x) at the Gauss-Hermite nodes.
        coeffs: (c0, c4, c8)
        Returns: g_vals, gp_vals
        """
        c0, c4, c8 = coeffs
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        
        # Compute powers of x_nodes
        x = x_nodes
        x2 = x**2
        x4 = x2**2
        x6 = x4 * x2
        x8 = x4**2
        x10 = x8 * x2
        x12 = x6**2
        
        # H0
        H0 = np.ones_like(x)
        # H4
        H4 = x4 - 6*x2 + 3
        # H8
        H8 = x8 - 28*x6 + 210*x4 - 420*x2 + 105
        # H12
        H12 = x12 - 66*x10 + 1260*x8 - 9240*x6 + 23100*x4 - 27720*x2 + 10395
        
        g = c0*H0 + c4*H4 + c8*H8 + c12*H12
        
        # Derivatives
        # H0' = 0
        # H4' = 4x^3 - 12x
        # H8' = 8x^7 - 168x^5 + 840x^3 - 840x
        # H12' = 12x^11 - 660x^9 + 10080x^7 - 55440x^5 + 92400x^3 - 55440x
        
        # We need x^3, x^5, x^7, x^9, x^11
        x3 = x2 * x
        x5 = x4 * x
        x7 = x6 * x
        x9 = x8 * x
        x11 = x10 * x
        
        dH0 = np.zeros_like(x)
        dH4 = 4*x3 - 12*x
        dH8 = 8*x7 - 168*x5 + 840*x3 - 840*x
        dH12 = 12*x11 - 660*x9 + 10080*x7 - 55440*x5 + 92400*x3 - 55440*x
        
        gp = c0*dH0 + c4*dH4 + c8*dH8 + c12*dH12
        
        return g, gp

    def compute_score(coeffs):
        if time.time() > deadline - 2:
            return np.inf
            
        c0, c4, c8 = coeffs
        try:
            g, gp = eval_g_and_deriv(coeffs)
            
            # Integrals: sum(w_i * f(x_i))
            # I0 = sum(w * g^2)
            I0 = np.sum(w_nodes * g**2)
            if I0 <= 1e-30 or not np.isfinite(I0):
                return np.inf
                
            # Ix2 = sum(w * x^2 * g^2)
            Ix2 = np.sum(w_nodes * x_nodes**2 * g**2)
            if not np.isfinite(Ix2) or Ix2 < 0:
                return np.inf
                
            # Ideriv = sum(w * (gp - x*g)^2)
            deriv_term = gp - x_nodes * g
            Ideriv = np.sum(w_nodes * deriv_term**2)
            if not np.isfinite(Ideriv) or Ideriv < 0:
                return np.inf
                
            # Product
            A_f_sq = Ix2 / I0
            A_gf_sq = (1.0 / (4.0 * np.pi**2)) * Ideriv / I0
            
            return np.sqrt(A_f_sq * A_gf_sq)
        except Exception:
            return np.inf

    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_score = compute_score(x0)
    best_coeffs = x0.copy()
    
    # Strategy: 
    # 1. Coarse search with Nelder-Mead (robust)
    # 2. Fine polish with L-BFGS-B (fast, uses gradients approx)
    
    # Nelder-Mead
    try:
        res = opt.minimize(compute_score, x0, method='Nelder-Mead', 
                          options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 1000, 'adaptive': True})
        if res.fun < best_score and np.isfinite(res.fun):
            best_score = res.fun
            best_coeffs = res.x.copy()
    except Exception:
        pass

    # L-BFGS-B Polish
    try:
        res2 = opt.minimize(compute_score, best_coeffs, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        if res2.fun < best_score and np.isfinite(res2.fun):
            best_score = res2.fun
            best_coeffs = res2.x.copy()
    except Exception:
        pass
        
    # Random restarts if time permits
    np.random.seed(42)
    for i in range(50):
        if time.time() > deadline - 10:
            break
        
        # Perturb around best found
        # Scale noise based on iteration
        noise_scale = 0.5 / (1 + i * 0.2)
        noise = np.random.normal(0, noise_scale, 3)
        x_try = best_coeffs + noise
        
        try:
            # Quick local optimization from this point
            res3 = opt.minimize(compute_score, x_try, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-12})
            if res3.fun < best_score and np.isfinite(res3.fun):
                best_score = res3.fun
                best_coeffs = res3.x.copy()
                
                # Optional: Polish with Nelder-Mead if L-BFGS-B got stuck? 
                # Or just keep L-BFGS-B as it's fast.
        except Exception:
            pass

    return np.array(best_coeffs)
