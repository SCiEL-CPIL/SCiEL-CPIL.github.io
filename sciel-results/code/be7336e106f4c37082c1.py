"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_q_coefficients(c0, c1, c2):
    """Compute coefficients of q(y) = b1 + b2*y + ... + b6*y^5 where y=x^2."""
    c3 = -(c0 + 12.0 * c1 + 168.0 * c2) / 18480.0
    b1 = -48.0 * c1 - 1344.0 * c2 - 921600.0 * c3
    b2 = 16.0 * c1 + 2688.0 * c2 + 1424640.0 * c3
    b3 = -1344.0 * c2 - 1003520.0 * c3
    b4 = 256.0 * c2 + 332160.0 * c3
    b5 = -46080.0 * c3
    b6 = 4096.0 * c3
    return np.array([b1, b2, b3, b4, b5, b6])

def _evaluate_value(c0, c1, c2):
    """Compute value = r^2 / (2*pi) where r is largest sign-changing root of q."""
    coeffs = _compute_q_coefficients(c0, c1, c2)
    # q(y) = b1 + b2*y + b3*y^2 + b4*y^3 + b5*y^4 + b6*y^5
    # numpy.poly1d expects highest degree first
    poly = np.poly1d(coeffs[::-1])
    roots = np.roots(coeffs[::-1])
    # Filter real positive roots
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    real_roots = real_roots[real_roots > 1e-10]
    if len(real_roots) == 0:
        return np.inf
    # Check sign change at each root
    valid_roots = []
    for r in real_roots:
        eps = 1e-8 * max(1.0, abs(r))
        q_left = poly(r - eps)
        q_right = poly(r + eps)
        if q_left * q_right < 0:
            valid_roots.append(r)
    if len(valid_roots) == 0:
        return np.inf
    r_max = max(valid_roots)
    return r_max / (2.0 * np.pi)

def construct():
    deadline = time.time() + 180.0
    best_c = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_val = _evaluate_value(best_c[0], best_c[1], best_c[2])
    
    # Local Nelder-Mead optimization from seed
    def objective(x):
        if time.time() > deadline - 5.0:
            return best_val
        try:
            return _evaluate_value(x[0], x[1], x[2])
        except:
            return 1e10
    
    try:
        res = minimize(objective, best_c, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        if time.time() < deadline - 5.0 and res.fun < best_val:
            best_c = res.x
            best_val = res.fun
    except:
        pass
    
    # Additional local refinement with tighter tolerances
    if time.time() < deadline - 10.0:
        try:
            def objective2(x):
                if time.time() > deadline - 5.0:
                    return best_val
                try:
                    return _evaluate_value(x[0], x[1], x[2])
                except:
                    return 1e10
            res2 = minimize(objective2, best_c, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
            if res2.fun < best_val:
                best_c = res2.x
                best_val = res2.fun
        except:
            pass
    
    return best_c
# EVOLVE-BLOCK-END
