import numpy as np
import scipy.optimize as so
from scipy.special import roots_hermite
import time

def construct():
    start_time = time.time()
    deadline = start_time + 180  # 180s limit, use 175s for safety
    
    def compute_value(c0, c1, c2):
        # Compute c3 so that g(0) = 0
        H0_0 = 1
        H4_0 = 12
        H8_0 = 112
        H12_0 = 924
        c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
        
        # Check sign at infinity: leading coefficient of g
        # g ~ c3 * 4096 * x^12
        # If c3 * 4096 < 0, g -> -inf, so we flip sign
        # But flipping sign doesn't change roots, just sign
        # So we just need to find roots of q = g/x^2
        
        # q coefficients (from x^10 down to x^0)
        # g = sum of terms, let's build polynomial coefficients
        # H0 = 1
        # H4 = 16x^4 - 48x^2 + 12
        # H8 = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 112
        # H12 = 4096x^12 - 32256x^10 + 84672x^8 - 103680x^6 + 60480x^4 - 12096x^2 + 924
        
        # g coefficients (x^12, x^11, ..., x^0)
        # Only even powers
        g_coeffs = np.zeros(13)  # index 12 down to 0
        g_coeffs[12] = c3 * 4096
        g_coeffs[10] = c3 * (-32256)
        g_coeffs[8] = c2 * 256 + c3 * 84672
        g_coeffs[6] = c2 * (-1344) + c3 * (-103680)
        g_coeffs[4] = c1 * 16 + c2 * 2688 + c3 * 60480
        g_coeffs[2] = c1 * (-48) + c2 * (-1344) + c3 * (-12096)
        g_coeffs[0] = c0 + c1 * 12 + c2 * 112 + c3 * 924  # should be ~0
        
        # q = g / x^2, so q_coeffs[i] = g_coeffs[i+2]
        q_coeffs = np.zeros(11)  # degree 10
        for i in range(11):
            q_coeffs[i] = g_coeffs[i + 2]
        
        # Find roots of q
        roots = np.roots(q_coeffs)
        
        # Filter real roots (imaginary part close to 0)
        real_roots = []
        for r in roots:
            if abs(r.imag) < 1e-8:
                real_roots.append(r.real)
        
        if not real_roots:
            return np.inf
        
        # Find sign-changing roots
        # For each real root, check if q changes sign
        # q is a polynomial, so we can evaluate q at root +/- epsilon
        eps = 1e-6
        sign_changing_roots = []
        for r in real_roots:
            # Evaluate q at r - eps and r + eps
            q_left = np.polyval(q_coeffs, r - eps)
            q_right = np.polyval(q_coeffs, r + eps)
            if q_left * q_right < 0:
                sign_changing_roots.append(r)
        
        if not sign_changing_roots:
            return np.inf
        
        r_max = max(sign_changing_roots)
        if r_max <= 0:
            return np.inf
        
        value = r_max**2 / (2 * np.pi)
        return value
    
    # Seed
    seed = np.array([-113/100, 1/25, 1/3240])
    best_val = compute_value(*seed)
    best_c = seed.copy()
    
    # Try local optimization from seed
    try:
        def objective(c):
            return compute_value(c[0], c[1], c[2])
        
        result = so.minimize(objective, seed, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
        if result.fun < best_val:
            best_val = result.fun
            best_c = result.x.copy()
    except:
        pass
    
    # Try some random restarts if time allows
    np.random.seed(42)
    for i in range(50):
        if time.time() > deadline - 30:
            break
        # Random perturbation around seed or best
        if np.random.random() < 0.5:
            c0, c1, c2 = seed + np.random.randn(3) * 0.1
        else:
            c0, c1, c2 = best_c + np.random.randn(3) * 0.01
        
        val = compute_value(c0, c1, c2)
        if val < best_val:
            best_val = val
            best_c = np.array([c0, c1, c2])
            
            # Local refinement
            try:
                def objective2(c):
                    return compute_value(c[0], c[1], c[2])
                result2 = so.minimize(objective2, best_c, method='Nelder-Mead',
                                     options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12})
                if result2.fun < best_val:
                    best_val = result2.fun
                    best_c = result2.x.copy()
            except:
                pass
    
    return np.array(best_c)
