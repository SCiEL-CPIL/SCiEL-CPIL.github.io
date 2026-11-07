"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_value(c0, c1, c2):
    """Compute the value = r^2 / (2*pi) for given coefficients using exact analytical formulas."""
    # Exact coefficients of H_n in terms of y = x^2
    # H0(y) = 1
    # H4(y) = 16y^2 - 48y + 12
    # H8(y) = 256y^4 - 3072y^3 + 11520y^2 - 13440y + 1680
    # H12(y) = 4096y^6 - 73728y^5 + 524280y^4 - 1468800y^3 + 1728960y^2 - 725760y + 103950
    
    # g(y) = c0*H0(y) + c1*H4(y) + c2*H8(y) + c3*H12(y)
    # g(0) = c0 + 12*c1 + 1680*c2 + 103950*c3 = 0
    # c3 = -(c0 + 12*c1 + 1680*c2) / 103950
    
    c3 = -(c0 + 12.0 * c1 + 1680.0 * c2) / 103950.0
    
    # q(y) = g(y)/y, degree 5 in y
    # g(y) = (c0*1 + c1*12 + c2*1680 + c3*103950) + 
    #        (c1*(-48) + c2*(-13440) + c3*(-725760))*y +
    #        (c1*16 + c2*11520 + c3*1728960)*y^2 +
    #        (c2*(-3072) + c3*(-1468800))*y^3 +
    #        (c2*256 + c3*524280)*y^4 +
    #        (c3*(-73728))*y^5 +
    #        (c3*4096)*y^6
    # 
    # Since g(0)=0, the constant term is 0.
    # q(y) = g(y)/y = a1 + a2*y + a3*y^2 + a4*y^3 + a5*y^4 + a6*y^5
    
    a1 = c1 * (-48.0) + c2 * (-13440.0) + c3 * (-725760.0)
    a2 = c1 * 16.0 + c2 * 11520.0 + c3 * 1728960.0
    a3 = c2 * (-3072.0) + c3 * (-1468800.0)
    a4 = c2 * 256.0 + c3 * 524280.0
    a5 = c3 * (-73728.0)
    a6 = c3 * 4096.0
    
    q_coeffs = [a1, a2, a3, a4, a5, a6]  # degree 5
    
    # Check if all coefficients are zero
    if all(abs(c) < 1e-15 for c in q_coeffs):
        return float('inf')
    
    # Find real roots of q(y) = 0
    roots = np.roots(q_coeffs)
    
    # Filter real roots with y >= 0
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            ry = r.real
            if ry >= -1e-10:
                real_roots.append(max(0.0, ry))
    
    if not real_roots:
        return float('inf')
    
    # For each real root y_r, check if q changes sign at y_r
    sign_change_roots = []
    eps = 1e-8
    
    def q_eval(y):
        """Evaluate q(y) using Horner's method"""
        result = a6
        result = result * y + a5
        result = result * y + a4
        result = result * y + a3
        result = result * y + a2
        result = result * y + a1
        return result
    
    for y_r in real_roots:
        if y_r < 1e-10:
            # At y=0, check sign change as y increases
            q_at_0 = a1
            q_at_eps = q_eval(eps)
            if q_at_0 * q_at_eps < 0:
                sign_change_roots.append(y_r)
        else:
            y_left = max(0.0, y_r - eps)
            y_right = y_r + eps
            q_left = q_eval(y_left)
            q_right = q_eval(y_right)
            if q_left * q_right < 0:
                sign_change_roots.append(y_r)
    
    if not sign_change_roots:
        return float('inf')
    
    max_y = max(sign_change_roots)
    value = max_y / (2.0 * np.pi)
    return value

def construct():
    start_time = time.time()
    deadline = start_time + 180.0
    
    # Seed coefficients
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_coeffs = seed.copy()
    best_value = _compute_value(*seed)
    
    # Nelder-Mead style local search with adaptive step sizes
    np.random.seed(42)
    
    # Phase 1: Coarse search
    step_size = 0.5
    for iteration in range(3000):
        if time.time() > deadline - 30.0:
            break
        
        # Generate perturbation
        noise = np.random.normal(0, step_size, 3)
        candidate = best_coeffs + noise
        
        val = _compute_value(*candidate)
        
        if val < best_value:
            best_value = val
            best_coeffs = candidate.copy()
            step_size *= 0.97
        else:
            step_size *= 1.01
            if step_size > 2.0:
                step_size = 2.0
    
    # Phase 2: Fine search
    step_size = 0.01
    for iteration in range(5000):
        if time.time() > deadline - 10.0:
            break
        
        noise = np.random.normal(0, step_size, 3)
        candidate = best_coeffs + noise
        val = _compute_value(*candidate)
        
        if val < best_value:
            best_value = val
            best_coeffs = candidate.copy()
            step_size *= 0.98
        else:
            step_size *= 1.005
            if step_size > 0.1:
                step_size = 0.1
    
    # Phase 3: Very fine polish
    step_size = 0.001
    for iteration in range(3000):
        if time.time() > deadline - 5.0:
            break
        
        noise = np.random.normal(0, step_size, 3)
        candidate = best_coeffs + noise
        val = _compute_value(*candidate)
        
        if val < best_value:
            best_value = val
            best_coeffs = candidate.copy()
            step_size *= 0.97
        else:
            step_size *= 1.002
            if step_size > 0.01:
                step_size = 0.01
    
    return best_coeffs
# EVOLVE-BLOCK-END
