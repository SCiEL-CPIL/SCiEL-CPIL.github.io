import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_hermite
from numpy.polynomial.hermite import hermgauss

START_TIME = time.time()
TIME_LIMIT = 200.0

def construct():
    # GH Quadrature setup
    # The integral is over -inf to inf with weight e^{-x^2/2}.
    # Transform to standard GH weight e^{-z^2}.
    # x = sqrt(2) z, dx = sqrt(2) dz.
    # Integral = sqrt(2) * sum(w_i * f(sqrt(2) z_i))
    # f(x) = x^4 * |P(x)|
    # f(sqrt(2) z) = (sqrt(2) z)^4 * |P(sqrt(2) z)| = 4 z^4 * |P(sqrt(2) z)|
    # So Integral = 4 * sqrt(2) * sum(w_i * z_i^4 * |P(sqrt(2) z_i)|)
    
    n_gh = 100 # 100 points is very accurate for degree 16 polynomials
    z, w = hermgauss(n_gh)
    x_gh = np.sqrt(2) * z
    w_gh = np.sqrt(2) * w
    
    # Precompute Hermite polynomials at x_gh
    # H0, H4, H8, H12
    H0 = np.ones_like(x_gh)
    H4 = eval_hermite(4, x_gh)
    H8 = eval_hermite(8, x_gh)
    H12 = eval_hermite(12, x_gh)
    
    # Coefficients for P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
    # c12 = -(c0 + 3 c4 + 105 c8) / 945
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=945
    
    def get_c12(c0, c4, c8):
        return -(c0 + 3.0*c4 + 105.0*c8) / 945.0

    def objective_and_grad(coeffs, eps=1e-4):
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        # P(x_gh)
        P_vals = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
        
        # Smooth absolute value: sqrt(P^2 + eps^2)
        P_abs_smooth = np.sqrt(P_vals**2 + eps**2)
        
        # Integrand for objective: 4 * x_gh^4 * P_abs_smooth * w_gh
        # Wait, the factor 4 comes from (sqrt(2)z)^4 = 4z^4.
        # And w_gh includes sqrt(2).
        # Let's re-verify.
        # I = int_{-inf}^{inf} x^4 |P(x)| e^{-x^2/2} dx
        # x = sqrt(2) z
        # I = int_{-inf}^{inf} (sqrt(2)z)^4 |P(sqrt(2)z)| e^{-z^2} sqrt(2) dz
        # I = 4 * sqrt(2) * int_{-inf}^{inf} z^4 |P(sqrt(2)z)| e^{-z^2} dz
        # GH approx: 4 * sqrt(2) * sum(w_i * z_i^4 * |P(sqrt(2) z_i)|)
        
        obj_val = 4.0 * np.sqrt(2.0) * np.sum(w * (z**4) * P_abs_smooth)
        
        # Gradient
        # d/dc_k int ... = int x^4 * (P / sqrt(P^2+eps^2)) * H_k(x) e^{-x^2/2} dx
        # = 4 * sqrt(2) * sum(w_i * z_i^4 * (P_i / P_abs_smooth_i) * H_k(x_gh_i))
        
        grad_factor = P_vals / P_abs_smooth
        common_part = 4.0 * np.sqrt(2.0) * w * (z**4) * grad_factor
        
        grad0 = np.sum(common_part * H0)
        grad4 = np.sum(common_part * H4)
        grad8 = np.sum(common_part * H8)
        
        # Note: c12 depends on c0, c4, c8.
        # d/dc0 = d/dc0_direct + d/dc12 * dc12/dc0
        # dc12/dc0 = -1/945
        # dc12/dc4 = -3/945
        # dc12/dc8 = -105/945
        # The gradient calculated above is for the direct terms c0, c4, c8.
        # We need to add the contribution from c12.
        # d/dck (c12 * H12) = dc12/dck * H12
        # So grad_total_k = grad_direct_k + dc12/dck * grad_H12_part
        # grad_H12_part = sum(common_part * H12)
        
        grad12_part = np.sum(common_part * H12)
        
        dc12_dc0 = -1.0 / 945.0
        dc12_dc4 = -3.0 / 945.0
        dc12_dc8 = -105.0 / 945.0
        
        grad0_total = grad0 + dc12_dc0 * grad12_part
        grad4_total = grad4 + dc12_dc4 * grad12_part
        grad8_total = grad8 + dc12_dc8 * grad12_part
        
        return obj_val, np.array([grad0_total, grad4_total, grad8_total])

    def objective_wrapper(coeffs):
        val, _ = objective_and_grad(coeffs)
        return val

    def grad_wrapper(coeffs):
        _, grad = objective_and_grad(coeffs)
        return grad

    # Initial guess
    best_coeffs = np.array([-1.13, 0.04, 1.0/3240.0])
    best_score = objective_wrapper(best_coeffs)
    
    np.random.seed(42)
    
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.1, 0.1)]
    
    # Multi-start with L-BFGS-B
    for i in range(20):
        if time.time() > TIME_LIMIT - 10:
            break
            
        x0 = np.array([
            np.random.uniform(bounds[0][0], bounds[0][1]),
            np.random.uniform(bounds[1][0], bounds[1][1]),
            np.random.uniform(bounds[2][0], bounds[2][1])
        ])
        
        try:
            res = minimize(objective_wrapper, x0, jac=grad_wrapper, method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
        except Exception:
            pass
            
    return best_coeffs
