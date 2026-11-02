import numpy as np
import scipy.optimize as opt
import time

def construct():
    start_time = time.time()
    deadline = 205.0  # Safety margin before 210s limit
    
    # Precompute Gauss-Hermite quadrature nodes and weights
    n_gauss = 15
    nodes, weights = np.polynomial.hermite.hermgauss(n_gauss)
    
    x_nodes = nodes
    
    def H0_vals(x):
        return np.ones_like(x, dtype=float)
    
    def H4_vals(x):
        return x**4 - 6*x**2 + 3
    
    def H8_vals(x):
        return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    
    def H12_vals(x):
        return x**12 - 66*x**10 + 1260*x**8 - 9240*x**6 + 23100*x**4 - 27720*x**2 + 10395
    
    H0_at_nodes = H0_vals(x_nodes)
    H4_at_nodes = H4_vals(x_nodes)
    H8_at_nodes = H8_vals(x_nodes)
    H12_at_nodes = H12_vals(x_nodes)
    
    dH0_at_nodes = np.zeros_like(x_nodes)
    dH4_at_nodes = 4*x_nodes**3 - 12*x_nodes
    dH8_at_nodes = 8*x_nodes**7 - 168*x_nodes**5 + 840*x_nodes**3 - 840*x_nodes
    dH12_at_nodes = 12*x_nodes**11 - 660*x_nodes**9 + 10080*x_nodes**7 - 55440*x_nodes**5 + 92400*x_nodes**3 - 55440*x_nodes
    
    x2_at_nodes = x_nodes**2
    
    dH0_minus_xH0 = dH0_at_nodes - x_nodes * H0_at_nodes
    dH4_minus_xH4 = dH4_at_nodes - x_nodes * H4_at_nodes
    dH8_minus_xH8 = dH8_at_nodes - x_nodes * H8_at_nodes
    dH12_minus_xH12 = dH12_at_nodes - x_nodes * H12_at_nodes
    
    # Precompute for gradient: we need dP/dc0, dP/dc4, dP/dc8, dP/dc12
    # and dQ/dc0, dQ/dc4, dQ/dc8, dQ/dc12 at each node
    # P = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # Q = c0*dH0xH0 + c4*dH4xH4 + c8*dH8xH8 + c12*dH12xH12
    
    # c12 = -(c0 + 3*c4 + 105*c8)/10395
    # dc12/dc0 = -1/10395
    # dc12/dc4 = -3/10395
    # dc12/dc8 = -105/10395
    
    c12_dc0 = -1.0/10395.0
    c12_dc4 = -3.0/10395.0
    c12_dc8 = -105.0/10395.0
    
    # Effective basis for P in terms of c0,c4,c8:
    # P = c0*(H0 + c12_dc0*H12) + c4*(H4 + c12_dc4*H12) + c8*(H8 + c12_dc8*H12)
    # Q = c0*(dH0xH0 + c12_dc0*dH12xH12) + c4*(dH4xH4 + c12_dc4*dH12xH12) + c8*(dH8xH8 + c12_dc8*dH12xH12)
    
    P_basis0 = H0_at_nodes + c12_dc0 * H12_at_nodes
    P_basis4 = H4_at_nodes + c12_dc4 * H12_at_nodes
    P_basis8 = H8_at_nodes + c12_dc8 * H12_at_nodes
    
    Q_basis0 = dH0_minus_xH0 + c12_dc0 * dH12_minus_xH12
    Q_basis4 = dH4_minus_xH4 + c12_dc4 * dH12_minus_xH12
    Q_basis8 = dH8_minus_xH8 + c12_dc8 * dH12_minus_xH12
    
    def compute_uncertainty_product(coeffs):
        c0, c4, c8 = coeffs
        
        if time.time() > deadline - 3:
            return np.inf
        
        try:
            P_at_nodes = c0 * P_basis0 + c4 * P_basis4 + c8 * P_basis8
            Q_at_nodes = c0 * Q_basis0 + c4 * Q_basis4 + c8 * Q_basis8
            
            I0 = np.sum(weights * P_at_nodes**2)
            if I0 <= 1e-30 or not np.isfinite(I0):
                return np.inf
            
            Ix2 = np.sum(weights * x2_at_nodes * P_at_nodes**2)
            if not np.isfinite(Ix2) or Ix2 < 0:
                return np.inf
            
            Ideriv = np.sum(weights * Q_at_nodes**2)
            if not np.isfinite(Ideriv) or Ideriv < 0:
                return np.inf
            
            A_f_sq = Ix2 / I0
            A_gf_sq = (1.0 / (4.0 * np.pi**2)) * (Ideriv / I0)
            
            return np.sqrt(A_f_sq * A_gf_sq)
        except Exception:
            return np.inf
    
    def compute_gradient(coeffs):
        """Compute gradient of the objective function."""
        c0, c4, c8 = coeffs
        
        try:
            P_at_nodes = c0 * P_basis0 + c4 * P_basis4 + c8 * P_basis8
            Q_at_nodes = c0 * Q_basis0 + c4 * Q_basis4 + c8 * Q_basis8
            
            I0 = np.sum(weights * P_at_nodes**2)
            if I0 <= 1e-30 or not np.isfinite(I0):
                return np.array([0.0, 0.0, 0.0])
            
            Ix2 = np.sum(weights * x2_at_nodes * P_at_nodes**2)
            Ideriv = np.sum(weights * Q_at_nodes**2)
            
            if not np.isfinite(Ix2) or not np.isfinite(Ideriv):
                return np.array([0.0, 0.0, 0.0])
            
            # f = sqrt( (Ix2/I0) * (Ideriv/I0) / (4*pi^2) )
            # Let A = Ix2/I0, B = Ideriv/I0
            # f = sqrt(A*B/(4*pi^2))
            # df/dc = (1/(2*f)) * (B/(4*pi^2) * dA/dc + A/(4*pi^2) * dB/dc)
            
            A = Ix2 / I0
            B = Ideriv / I0
            f = np.sqrt(A * B / (4.0 * np.pi**2))
            
            if f <= 1e-30 or not np.isfinite(f):
                return np.array([0.0, 0.0, 0.0])
            
            # dA/dc = (dIx2/dc * I0 - Ix2 * dI0/dc) / I0^2
            # dB/dc = (dIderiv/dc * I0 - Ideriv * dI0/dc) / I0^2
            
            # dI0/dc_i = 2 * sum(w * P * dP/dc_i)
            # dIx2/dc_i = 2 * sum(w * x^2 * P * dP/dc_i)
            # dIderiv/dc_i = 2 * sum(w * Q * dQ/dc_i)
            
            grad = np.zeros(3)
            for i in range(3):
                if i == 0:
                    dP_dc = P_basis0
                    dQ_dc = Q_basis0
                elif i == 1:
                    dP_dc = P_basis4
                    dQ_dc = Q_basis4
                else:
                    dP_dc = P_basis8
                    dQ_dc = Q_basis8
                
                dI0_dc = 2.0 * np.sum(weights * P_at_nodes * dP_dc)
                dIx2_dc = 2.0 * np.sum(weights * x2_at_nodes * P_at_nodes * dP_dc)
                dIderiv_dc = 2.0 * np.sum(weights * Q_at_nodes * dQ_dc)
                
                dA_dc = (dIx2_dc * I0 - Ix2 * dI0_dc) / (I0 * I0)
                dB_dc = (dIderiv_dc * I0 - Ideriv * dI0_dc) / (I0 * I0)
                
                # df/dc = (1/(2*f)) * (B/(4*pi^2) * dA_dc + A/(4*pi^2) * dB_dc)
                grad[i] = (1.0 / (2.0 * f)) * (B / (4.0 * np.pi**2) * dA_dc + A / (4.0 * np.pi**2) * dB_dc)
            
            if not np.all(np.isfinite(grad)):
                return np.array([0.0, 0.0, 0.0])
            
            return grad
        except Exception:
            return np.array([0.0, 0.0, 0.0])
    
    # Initial guess from seed (Goncalves et al. 2017)
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_score = compute_uncertainty_product(x0)
    best_coeffs = x0.copy()
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Local optimization from seed with Nelder-Mead
    if time.time() < deadline - 15:
        try:
            res = opt.minimize(compute_uncertainty_product, x0, method='Nelder-Mead', 
                              options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 2000, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Random restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        if i < 10:
            scale = 0.05 * (1 + i/10.0)
            x_start = best_coeffs + rng.normal(0, scale, 3)
        elif i < 30:
            scale = 0.1
            x_start = x0 + rng.normal(0, scale, 3)
        else:
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
    
    # Phase 3: Gradient-based polish using BFGS
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(compute_uncertainty_product, best_coeffs, jac=compute_gradient, 
                              method='BFGS', options={'gtol': 1e-12, 'maxiter': 500})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final Nelder-Mead polish
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
