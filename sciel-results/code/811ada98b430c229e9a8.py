import numpy as np
from scipy.special import roots_hermite
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 170.0
    
    # Gauss-Hermite quadrature nodes and weights
    # ∫ f(x) e^{-x²} dx ≈ Σ w_i f(x_i)
    n_quad = 200
    x_nodes, w_nodes = roots_hermite(n_quad)
    
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 560.0
    H12_0 = 425280.0
    
    def compute_a12(a0, a4, a8):
        return -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    def P_and_dP(x, a0, a4, a8):
        a12 = compute_a12(a0, a4, a8)
        x2 = x**2
        x4 = x2**2
        x6 = x4*x2
        x8 = x4**2
        x10 = x8*x2
        x12 = x6**2
        
        H0 = 1.0
        H4 = 16*x4 - 48*x2 + 12
        H8 = 256*x8 - 4480*x6 + 26880*x4 - 60480*x2 + 560
        H12 = 4096*x12 - 147456*x10 + 2190240*x8 - 14745600*x6 + 46137600*x4 - 59978880*x2 + 425280
        
        dH0 = 0.0
        dH4 = 64*x**3 - 96*x
        dH8 = 2048*x**7 - 26880*x**5 + 107520*x**3 - 120960*x
        dH12 = 49152*x**11 - 1474560*x**9 + 17521920*x**7 - 88473600*x**5 + 184550400*x**3 - 119957760*x
        
        p = a0*H0 + a4*H4 + a8*H8 + a12*H12
        dp = a0*dH0 + a4*dH4 + a8*dH8 + a12*dH12
        return p, dp
    
    def objective(params):
        a0, a4, a8 = params
        
        p_vals, dp_vals = P_and_dP(x_nodes, a0, a4, a8)
        
        # I0 = ∫ P(x)^2 e^{-x^2} dx
        I0 = np.sum(w_nodes * p_vals**2)
        
        # I2 = ∫ x^2 P(x)^2 e^{-x^2} dx
        I2 = np.sum(w_nodes * x_nodes**2 * p_vals**2)
        
        # I2_fourier = ∫ [P'(x) - xP(x)]^2 e^{-x^2} dx
        I2_fourier = np.sum(w_nodes * (dp_vals - x_nodes * p_vals)**2)
        
        if I0 <= 1e-15:
            return 1e10
        
        C4 = I2 * I2_fourier / (I0 * I0)
        return C4
    
    best_params = np.array([-113/100, 1/25, 1/3240])
    best_val = objective(best_params)
    
    start_points = [
        best_params.copy(),
        np.array([1.0, 0.1, 0.01]),
        np.array([-1.0, 0.1, 0.01]),
        np.array([0.5, -0.1, 0.01]),
        np.array([-0.5, -0.1, 0.01]),
        np.array([1.0, -0.1, -0.01]),
        np.array([-1.0, -0.1, -0.01]),
        np.array([0.1, 0.1, 0.1]),
        np.array([-0.1, 0.1, 0.1]),
    ]
    
    for sp in start_points:
        if time.time() > deadline:
            break
        try:
            result = minimize(objective, sp, method='Nelder-Mead', 
                            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-12, 'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    return best_params
