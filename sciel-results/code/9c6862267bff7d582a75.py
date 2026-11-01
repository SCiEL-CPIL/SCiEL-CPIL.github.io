import numpy as np
from scipy.optimize import minimize
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 180.0
    
    # Hermite Polynomials H0, H4, H8, H12
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12(x) = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 63000x^4 - 126000x^2 + 10395
    
    def H0(x): return 1.0
    def H4(x): return x**4 - 6*x**2 + 3.0
    def H8(x): return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105.0
    def H12(x): return x**12 - 60*x**10 + 1260*x**8 - 12600*x**6 + 63000*x**4 - 126000*x**2 + 10395.0

    def g(x, c0, c4, c8):
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        return (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * np.exp(-x**2/2.0)

    def g_prime(x, c0, c4, c8):
        # Derivative of (P(x) e^{-x^2/2}) is (P'(x) - x P(x)) e^{-x^2/2}
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        P = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
        P_prime = c0*0 + c4*(4*x**3 - 12*x) + c8*(8*x**7 - 168*x**5 + 840*x**3 - 840*x) + c12*(12*x**11 - 600*x**9 + 10080*x**7 - 75600*x**5 + 252000*x**3 - 252000*x)
        return (P_prime - x*P) * np.exp(-x**2/2.0)

    # Numerical integration using Gauss-Hermite quadrature? 
    # The weight is e^{-x^2}. Gauss-Hermite is for e^{-x^2}.
    # But our function has e^{-x^2/2}. 
    # Let y = x/sqrt(2). dx = sqrt(2) dy.
    # g(x) = P(x) e^{-x^2/2}.
    # It is easier to use standard numerical integration on a large grid or scipy.integrate.quad.
    # quad is robust and accurate enough.
    
    from scipy.integrate import quad

    def compute_product(c0, c4, c8):
        # Integrate g^2, x^2 g^2, (g')^2
        # g is even, so integrate 0 to inf and multiply by 2
        try:
            f_g2 = lambda x: g(x, c0, c4, c8)**2
            f_x2g2 = lambda x: x**2 * g(x, c0, c4, c8)**2
            f_gp2 = lambda x: g_prime(x, c0, c4, c8)**2
            
            I0, _ = quad(f_g2, 0, np.inf, limit=200)
            I2, _ = quad(f_x2g2, 0, np.inf, limit=200)
            Igp, _ = quad(f_gp2, 0, np.inf, limit=200)
            
            I0 *= 2.0
            I2 *= 2.0
            Igp *= 2.0
            
            if I0 < 1e-10:
                return np.inf
            
            var_x = I2 / I0
            var_xi = Igp / I0
            
            if var_x < 0 or var_xi < 0:
                return np.inf
                
            return np.sqrt(var_x * var_xi)
        except:
            return np.inf

    # Optimization
    best_coeffs = np.array([-1.13, 0.04, 0.000308]) # Seed
    best_val = compute_product(*best_coeffs)
    
    # Multi-start
    # Generate random starting points around the seed and some broad range
    rng = np.random.default_rng(42)
    starts = [best_coeffs.copy()]
    
    # Local neighborhood
    for _ in range(20):
        starts.append(best_coeffs + rng.normal(0, 0.1, 3))
        
    # Broader search
    for _ in range(20):
        starts.append(rng.uniform(-2, 2, 3))
        
    # Specific known good regions if any? 
    # Let's just run minimize on these starts
    
    for c0, c4, c8 in starts:
        if time.time() > deadline - 10:
            break
            
        res = minimize(compute_product, [c0, c4, c8], method='Nelder-Mead', 
                       options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500})
        
        if res.fun < best_val:
            best_val = res.fun
            best_coeffs = res.x
            
    return np.array(best_coeffs)
