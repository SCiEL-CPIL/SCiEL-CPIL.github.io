"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _hermite_poly_coeffs(n):
    """Return coefficients of physicists' Hermite polynomial H_n(x) as list [a_0, a_1, ..., a_n]
    where H_n(x) = sum a_k * x^k."""
    if n == 0:
        return [1]
    if n == 1:
        return [0, 2]
    h_prev = [1]  # H_0
    h_curr = [0, 2]  # H_1
    for k in range(1, n):
        # H_{k+1} = 2x * H_k - 2k * H_{k-1}
        new_len = len(h_curr) + 1
        h_next = [0] * new_len
        for i in range(len(h_curr)):
            h_next[i+1] = 2 * h_curr[i]
        for i in range(len(h_prev)):
            h_next[i] -= 2 * k * h_prev[i]
        h_prev = h_curr
        h_curr = h_next
    return h_curr

# Precompute Hermite polynomial coefficients
H0_coeffs = _hermite_poly_coeffs(0)
H4_coeffs = _hermite_poly_coeffs(4)
H8_coeffs = _hermite_poly_coeffs(8)
H12_coeffs = _hermite_poly_coeffs(12)

def _even_coeffs(h_coeffs):
    """From polynomial in x, extract coefficients of even powers (i.e., polynomial in y=x^2)."""
    result = []
    for i in range(0, len(h_coeffs), 2):
        result.append(h_coeffs[i])
    return result

h0_y = _even_coeffs(H0_coeffs)    # [1]
h4_y = _even_coeffs(H4_coeffs)    # [12, -48, 16]
h8_y = _even_coeffs(H8_coeffs)    # [168, -1344, 2688, -1344, 256]
h12_y = _even_coeffs(H12_coeffs)  # [18480, -921600, 1424640, -1003520, 332160, -46080, 4096]

def compute_q_coeffs(c0, c1, c2):
    """
    Compute coefficients of q(y) = g(x)/x^2 as a polynomial in y = x^2.
    Returns q_coeffs as list [q_0, ..., q_5] and c3.
    """
    c3 = -(c0 + 12*c1 + 168*c2) / 18480.0
    
    max_deg = 6
    b = [0.0] * (max_deg + 1)
    for k in range(len(h0_y)):
        b[k] += c0 * h0_y[k]
    for k in range(len(h4_y)):
        b[k] += c1 * h4_y[k]
    for k in range(len(h8_y)):
        b[k] += c2 * h8_y[k]
    for k in range(len(h12_y)):
        b[k] += c3 * h12_y[k]
    
    # q(y) = g(y)/y
    q_coeffs = b[1:7]
    return q_coeffs, c3

def evaluate_value(c0, c1, c2):
    """
    Compute value = r^2/(2*pi) where r is the largest real root of q(x) at which q changes sign.
    """
    q_coeffs, c3 = compute_q_coeffs(c0, c1, c2)
    
    lead_coeff = c3 * 4096.0
    if lead_coeff < 0:
        q_coeffs = [-c for c in q_coeffs]
    
    try:
        roots_y = np.roots(q_coeffs)
    except Exception:
        return float('inf')
    
    real_positive_roots = []
    for r in roots_y:
        if abs(r.imag) < 1e-10 and r.real > 1e-12:
            real_positive_roots.append(r.real)
    
    if not real_positive_roots:
        return float('inf')
    
    def q_eval(y):
        result = 0.0
        for coeff in q_coeffs:
            result = result * y + coeff
        return result
    
    best_r = 0.0
    for y_root in real_positive_roots:
        eps = max(1e-10, y_root * 1e-10)
        val_minus = q_eval(y_root - eps) if y_root > eps else q_eval(0)
        val_plus = q_eval(y_root + eps)
        
        if val_minus * val_plus < 0:
            r = math.sqrt(y_root)
            if r > best_r:
                best_r = r
    
    if best_r == 0.0:
        return float('inf')
    
    value = best_r**2 / (2.0 * math.pi)
    return value

def construct():
    """
    Find coefficients [c0, c1, c2] that minimize value = r^2/(2*pi).
    Uses Nelder-Mead optimization with multi-start strategy.
    """
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin

    # Seed
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_coeffs = seed.copy()
    best_value = evaluate_value(seed[0], seed[1], seed[2])

    try:
        from scipy.optimize import minimize
        have_scipy = True
    except ImportError:
        have_scipy = False

    def objective(coeffs):
        val = evaluate_value(coeffs[0], coeffs[1], coeffs[2])
        if not math.isfinite(val):
            return 1e10
        return val

    if have_scipy:
        # Multi-start Nelder-Mead
        # Define starting points around the seed and some random exploratory points
        start_points = [seed.copy()]
        
        # Add some perturbed points around the seed
        rng = np.random.default_rng(42)
        for _ in range(10):
            perturb = rng.normal(0, 0.1, size=3)
            perturb[2] *= 0.01  # c2 is much smaller in scale
            start_points.append(seed + perturb)
        
        # Add some broader exploratory points
        for _ in range(10):
            p0 = rng.uniform(-2.0, 2.0)
            p1 = rng.uniform(-0.5, 0.5)
            p2 = rng.uniform(-0.005, 0.005)
            start_points.append(np.array([p0, p1, p2]))
        
        for sp in start_points:
            if time.time() > deadline - 10:
                break
            try:
                result = minimize(
                    objective, sp,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                if result.fun < best_value:
                    best_value = result.fun
                    best_coeffs = result.x.copy()
            except Exception:
                pass
    else:
        # Fallback: simple local search
        step_sizes = [0.1, 0.01, 0.001, 0.0001]
        current = seed.copy()
        for ss in step_sizes:
            for _ in range(1000):
                if time.time() > deadline - 10:
                    break
                direction = np.random.randn(3)
                direction[2] *= 0.01
                new_point = current + ss * direction
                new_val = evaluate_value(new_point[0], new_point[1], new_point[2])
                if new_val < best_value:
                    best_value = new_val
                    best_coeffs = new_point.copy()
                if new_val < evaluate_value(current[0], current[1], current[2]):
                    current = new_point

    return best_coeffs
# EVOLVE-BLOCK-END
