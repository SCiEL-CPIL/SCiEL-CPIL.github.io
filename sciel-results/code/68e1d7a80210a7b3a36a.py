"""Search-based program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.optimize import brentq
import sys

# Precompute the Hermite polynomial coefficients (physicists' convention)
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
# H12(x) = 4096x^12 - 46080x^10 + 332160x^8 - 1003520x^6 + 1424640x^4 - 921600x^2 + 18480

# Coefficients for each Hermite polynomial (constant term, x^2 term, x^4 term, ...)
# H0: [1]
# H4: [12, -48, 16]  (coeffs of x^0, x^2, x^4)
# H8: [168, -1344, 2688, -1344, 256]  (coeffs of x^0, x^2, x^4, x^6, x^8)
# H12: [18480, -921600, 1424640, -1003520, 332160, -46080, 4096]  (coeffs of x^0, x^2, x^4, x^6, x^8, x^10, x^12)

H0_coeffs = [1.0]
H4_coeffs = [12.0, -48.0, 16.0]
H8_coeffs = [168.0, -1344.0, 2688.0, -1344.0, 256.0]
H12_coeffs = [18480.0, -921600.0, 1424640.0, -1003520.0, 332160.0, -46080.0, 4096.0]

# H12(0) = 18480
H12_0 = 18480.0

def compute_q_coefficients(c0, c1, c2):
    """
    Compute the coefficients of q(x) = g(x)/x^2 where
    g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    c3 = -(c0*H0(0) + c1*H4(0) + c2*H8(0)) / H12(0)
    
    q(x) is an even polynomial of degree 10, so we work with y = x^2.
    q(x) = sum_{k=0}^{5} q_k * y^k where y = x^2
    
    Returns: list of coefficients [q_0, q_1, ..., q_5] for powers y^0 to y^5
    """
    # c3 is determined by g(0) = 0
    H0_0 = H0_coeffs[0]  # 1
    H4_0 = H4_coeffs[0]  # 12
    H8_0 = H8_coeffs[0]  # 168
    
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # Now compute g(x) coefficients. g(x) is even, degree 12.
    # g(x) = sum_{k=0}^{6} g_k * x^{2k}
    # g(x)/x^2 = sum_{k=0}^{5} g_{k+1} * x^{2k} = sum_{k=0}^{5} q_k * y^k
    
    # Initialize g coefficients (length 7, for x^0 to x^12)
    g_coeffs = [0.0] * 7
    
    # Add c0 * H0
    for i, coeff in enumerate(H0_coeffs):
        g_coeffs[i] += c0 * coeff
    
    # Add c1 * H4
    for i, coeff in enumerate(H4_coeffs):
        g_coeffs[i] += c1 * coeff
    
    # Add c2 * H8
    for i, coeff in enumerate(H8_coeffs):
        g_coeffs[i] += c2 * coeff
    
    # Add c3 * H12
    for i, coeff in enumerate(H12_coeffs):
        g_coeffs[i] += c3 * coeff
    
    # q(x) = g(x)/x^2, so q_k = g_{k+1} for k = 0..5
    q_coeffs = [g_coeffs[k+1] for k in range(6)]
    
    return q_coeffs

def evaluate_value(c0, c1, c2):
    """
    Compute the value = r^2 / (2*pi) where r is the largest real root of q
    at which q changes sign.
    
    Returns the value, or float('inf') if no valid root exists.
    """
    q_coeffs = compute_q_coefficients(c0, c1, c2)
    
    # q(y) = sum_{k=0}^{5} q_coeffs[k] * y^k where y = x^2 >= 0
    # We need to find roots of q(y) = 0 for y >= 0 where q changes sign.
    
    # Use numpy to find roots of the polynomial in y
    # q_coeffs is [q_0, q_1, ..., q_5] for y^0 to y^5
    # numpy.roots expects coefficients in descending order: [q_5, q_4, ..., q_0]
    poly_coeffs_desc = list(reversed(q_coeffs))
    
    try:
        roots_y = np.roots(poly_coeffs_desc)
    except Exception:
        return float('inf')
    
    # Filter for real, non-negative roots
    valid_roots = []
    for r in roots_y:
        if np.isreal(r):
            r_real = np.real(r)
            if r_real >= -1e-10:  # allow small negative due to numerical error
                r_real = max(0.0, r_real)
                valid_roots.append(r_real)
    
    if not valid_roots:
        return float('inf')
    
    # For each valid root, check if q changes sign there.
    # A sign change means the root has odd multiplicity.
    # We can check by evaluating q at points slightly below and above the root.
    
    def q_eval(y):
        """Evaluate q(y) = sum q_coeffs[k] * y^k"""
        result = 0.0
        for k, coeff in enumerate(q_coeffs):
            result += coeff * (y ** k)
        return result
    
    sign_change_roots = []
    for r in valid_roots:
        if r < 1e-12:
            # Root at y=0, check sign change from positive side
            eps = 1e-8
            val_before = q_eval(0.0)  # q(0) = q_coeffs[0]
            val_after = q_eval(eps)
            if val_before * val_after < 0:
                sign_change_roots.append(r)
        else:
            eps = 1e-8 * max(1.0, r)
            val_before = q_eval(r - eps)
            val_after = q_eval(r + eps)
            if val_before * val_after < 0:
                sign_change_roots.append(r)
    
    if not sign_change_roots:
        return float('inf')
    
    # r is the largest real root at which q changes sign
    # Note: r in the problem is x, and y = x^2, so the root in x is sqrt(y)
    # Wait, let me re-read the problem.
    # "Let q = g / x^2, which must divide exactly. Let r be the largest real root of q at which q changes sign."
    # q is a function of x. q(x) = g(x)/x^2. So q is an even function of x.
    # The roots of q are the same as the roots of g except possibly at x=0.
    # Since g(0)=0 and g is even, g(x) = x^2 * q(x).
    # The roots of q(x) = 0 are the roots of g(x) = 0 excluding x=0 (if it's a simple root of g).
    # q(x) is even, so its roots come in pairs ±r.
    # "r be the largest real root" - this likely means the largest positive root.
    
    # So r is the largest positive x such that q(x) = 0 and q changes sign at x.
    # In terms of y = x^2, the root is y = r^2.
    
    # So we need the largest y in sign_change_roots, and r = sqrt(y).
    
    max_y = max(sign_change_roots)
    r = np.sqrt(max_y)
    
    value = r**2 / (2 * np.pi)
    
    return value

def construct():
    """
    Search for coefficients [c0, c1, c2] that minimize the value.
    """
    start_time = time.time()
    deadline = start_time + 200.0  # 200 second deadline (10s safety margin)
    
    # Seed value
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_value = evaluate_value(seed[0], seed[1], seed[2])
    best_coeffs = seed.copy()
    
    if best_value == float('inf'):
        # Seed is invalid, need to find a valid point
        # Try some random points
        for _ in range(100):
            c0 = np.random.uniform(-5, 5)
            c1 = np.random.uniform(-5, 5)
            c2 = np.random.uniform(-5, 5)
            val = evaluate_value(c0, c1, c2)
            if val < float('inf'):
                best_value = val
                best_coeffs = np.array([c0, c1, c2])
                break
    
    # Define the objective function for optimization
    def objective(x):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            return float('inf')
        try:
            return evaluate_value(x[0], x[1], x[2])
        except Exception:
            return float('inf')
    
    # Use differential evolution for global search
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=50,
            tol=1e-8,
            seed=42,
            popsize=15,
            polish=True
        )
        
        if result.fun < best_value:
            best_value = result.fun
            best_coeffs = result.x.copy()
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 20.0:
        return best_coeffs
    
    # Refine with local optimization
    try:
        result2 = minimize(
            objective,
            best_coeffs,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        
        if result2.fun < best_value:
            best_value = result2.fun
            best_coeffs = result2.x.copy()
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 10.0:
        return best_coeffs
    
    # Try more local optimizations from different starting points
    for _ in range(20):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            break
        
        # Perturb the best solution
        perturb = np.random.normal(0, 0.5, 3)
        start_point = best_coeffs + perturb
        
        try:
            result3 = minimize(
                objective,
                start_point,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if result3.fun < best_value:
                best_value = result3.fun
                best_coeffs = result3.x.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
