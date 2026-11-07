import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import factorial
import time

# Precompute Hermite polynomial coefficients
# H_n(x) = sum_{k=0}^{n//2} (-1)^k * n! / (k! (n-2k)!) * (2x)^(n-2k)
def _hermite_coeffs(n):
    coeffs = {}
    for k in range(n // 2 + 1):
        coeff = (-1)**k * float(factorial(n)) / (float(factorial(k)) * float(factorial(n - 2*k))) * (2**(n - 2*k))
        coeffs[n - 2*k] = coeff
    result = [0.0] * (n + 1)
    for deg, c in coeffs.items():
        result[n - deg] = c
    return result

_H0 = _hermite_coeffs(0)
_H4 = _hermite_coeffs(4)
_H8 = _hermite_coeffs(8)
_H12 = _hermite_coeffs(12)

# Convert to Q(y) basis
# g(x) = G(x^2)
# G(y) = c0 * H0(y) + c1 * H4(y) + c2 * H8(y) + c3 * H12(y)
# H0(y) = 1
# H4(y) = y^2 - 6y + 3
# H8(y) = y^4 - 28y^3 + 210y^2 - 420y + 105
# H12(y) = y^6 - 84y^5 + 2520y^4 - 30240y^3 + 151200y^2 - 221760y + 10395

# Q(y) = G(y)/y
# Coeffs of G(y) (degree 6 down to 0)
# deg 6: c3
# deg 5: -84 c3
# deg 4: 2520 c3 + c2
# deg 3: -30240 c3 - 28 c2
# deg 2: 151200 c3 + 210 c2 - 6 c1
# deg 1: -221760 c3 - 420 c2 - 6 c1
# deg 0: 10395 c3 + 105 c2 + 3 c1 + c0 = 0

# Q(y) coeffs (degree 5 down to 0)
# deg 5: c3
# deg 4: -84 c3
# deg 3: 2520 c3 + c2
# deg 2: -30240 c3 - 28 c2
# deg 1: 151200 c3 + 210 c2 - 6 c1
# deg 0: -221760 c3 - 420 c2 - 6 c1

def _get_q_coeffs(c0, c1, c2):
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0
    
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    q5 = c3
    q4 = -84.0 * c3
    q3 = 2520.0 * c3 + c2
    q2 = -30240.0 * c3 - 28.0 * c2
    q1 = 151200.0 * c3 + 210.0 * c2 - 6.0 * c1
    q0 = -221760.0 * c3 - 420.0 * c2 - 6.0 * c1
    
    return [q5, q4, q3, q2, q1, q0]

def _compute_value(c0, c1, c2):
    q_coeffs = _get_q_coeffs(c0, c1, c2)
    # Remove leading zeros
    while len(q_coeffs) > 1 and abs(q_coeffs[0]) < 1e-10:
        q_coeffs.pop(0)
        
    if len(q_coeffs) == 1:
        return float('inf')
        
    roots = np.roots(q_coeffs)
    real_roots = roots[np.abs(roots.imag) < 1e-9].real
    
    best_r = None
    eps = 1e-7
    
    for r in real_roots:
        if r <= 0:
            continue
        q_minus = np.polyval(q_coeffs, r - eps)
        q_plus = np.polyval(q_coeffs, r + eps)
        if q_minus * q_plus < 0:
            if best_r is None or r > best_r:
                best_r = r
                
    if best_r is None:
        return float('inf')
        
    return best_r**2 / (2 * np.pi)

def construct():
    start_time = time.time()
    deadline = start_time + 180.0
    
    best_coeffs = None
    best_value = float('inf')
    
    def update_best(c0, c1, c2):
        nonlocal best_coeffs, best_value
        try:
            v = _compute_value(c0, c1, c2)
            if v < best_value:
                best_value = v
                best_coeffs = [c0, c1, c2]
        except:
            pass

    # Try fixing c0=1
    # Optimize c1, c2
    def obj1(x):
        c1, c2 = x
        return _compute_value(1.0, c1, c2)
        
    # Try fixing c1=1
    def obj2(x):
        c0, c2 = x
        return _compute_value(c0, 1.0, c2)
        
    # Try fixing c2=1
    def obj3(x):
        c0, c1 = x
        return _compute_value(c0, c1, 1.0)

    bounds = [(-100, 100), (-100, 100)]
    
    # Differential Evolution for each case
    for i, obj in enumerate([obj1, obj2, obj3]):
        if time.time() > deadline - 60:
            break
        try:
            result = differential_evolution(obj, bounds, maxiter=100, seed=42+i, polish=True, tol=1e-8)
            if result.fun < best_value:
                if i == 0:
                    update_best(1.0, result.x[0], result.x[1])
                elif i == 1:
                    update_best(result.x[0], 1.0, result.x[1])
                else:
                    update_best(result.x[0], result.x[1], 1.0)
        except:
            pass

    # Local refinement from best found
    if best_coeffs is not None:
        c0, c1, c2 = best_coeffs
        
        # Use Nelder-Mead on 3D space
        def obj_full(x):
            return _compute_value(x[0], x[1], x[2])
            
        try:
            result = minimize(obj_full, [c0, c1, c2], method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_value:
                update_best(result.x[0], result.x[1], result.x[2])
        except:
            pass

    # If still inf, try random search
    if best_value == float('inf'):
        rng = np.random.RandomState(42)
        for _ in range(1000):
            if time.time() > deadline - 10:
                break
            c = rng.uniform(-10, 10, 3)
            update_best(c[0], c[1], c[2])

    if best_coeffs is None:
        return np.array([-113.0/100, 1.0/25, 1.0/3240.0])
        
    return np.array(best_coeffs)
