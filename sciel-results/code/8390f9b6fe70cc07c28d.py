"""Optimized program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Precompute Hermite polynomial coefficients in terms of y = x^2
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12
# H_8(x) = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
# H_12(x) = 4096x^12 - 46080x^10 + 332160x^8 - 1003520x^6 + 1424640x^4 - 921600x^2 + 18480

# Coefficients in ascending power of y
H0_y = np.array([1.0])
H4_y = np.array([12.0, -48.0, 16.0])
H8_y = np.array([168.0, -1344.0, 2688.0, -1344.0, 256.0])
H12_y = np.array([18480.0, -921600.0, 1424640.0, -1003520.0, 332160.0, -46080.0, 4096.0])


def compute_q_coeffs(c0, c1, c2):
    """
    Compute coefficients of q(y) = g(x)/x^2 as a polynomial in y.
    Returns coefficients in ascending power of y: [q0, q1, ..., q5]
    """
    c3 = -(c0 * 1.0 + c1 * 12.0 + c2 * 168.0) / 18480.0
    
    # Build G(y) coefficients (degree 6, ascending)
    G = np.zeros(7)
    G[0:1] += c0 * H0_y
    G[0:3] += c1 * H4_y
    G[0:5] += c2 * H8_y
    G[0:7] += c3 * H12_y
    
    # q(y) = G(y) / y, so q coefficients are G[1:7]
    q = G[1:7]
    return q


def find_largest_sign_changing_root(q_coeffs):
    """
    Find the largest real positive root of q(y) where q changes sign.
    q_coeffs are in ascending power of y.
    Returns the root value (in y-space), or None if no such root exists.
    """
    # numpy.roots expects descending order
    q_desc = q_coeffs[::-1].copy()
    
    # Remove leading zeros
    while len(q_desc) > 1 and abs(q_desc[0]) < 1e-15:
        q_desc = q_desc[1:]
    
    if len(q_desc) <= 1:
        return None
    
    roots = np.roots(q_desc)
    
    # Filter real positive roots
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-9 * max(1.0, abs(r.real)) and r.real > 0:
            real_roots.append(r.real)
    
    if not real_roots:
        return None
    
    # For each real positive root, check if q changes sign there
    # Evaluate q at r-eps and r+eps
    eps = 1e-7
    sign_changing = []
    for r in real_roots:
        q_left = np.polyval(q_desc, r - eps)
        q_right = np.polyval(q_desc, r + eps)
        if q_left * q_right < 0:
            sign_changing.append(r)
    
    if not sign_changing:
        return None
    
    return max(sign_changing)


def evaluate_value(c0, c1, c2):
    """
    Compute the value = r^2 / (2*pi) where r is the largest sign-changing root of q.
    Since r^2 = y_root (the root in y-space), value = y_root / (2*pi).
    Returns a large number if no valid root is found.
    """
    q = compute_q_coeffs(c0, c1, c2)
    y_root = find_largest_sign_changing_root(q)
    if y_root is None:
        return 1e10
    return y_root / (2 * np.pi)


def objective(params):
    """Objective function for optimization."""
    c0, c1, c2 = params
    return evaluate_value(c0, c1, c2)


def construct():
    """
    Find optimal coefficients [c0, c1, c2] to minimize the value.
    """
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin before 210s kill
    
    # Seed values
    seed = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_params = seed.copy()
    best_value = evaluate_value(*seed)
    
    # Generate diverse starting points
    start_points = [seed]
    
    rng = np.random.default_rng(12345)
    
    # Random perturbations around seed
    for i in range(100):
        scale0 = 0.5 * (1 + i * 0.05)
        scale1 = 0.5 * (1 + i * 0.05)
        scale2 = 0.01 * (1 + i * 0.05)
        pt = np.array([
            seed[0] + rng.uniform(-scale0, scale0),
            seed[1] + rng.uniform(-scale1, scale1),
            seed[2] + rng.uniform(-scale2, scale2)
        ])
        start_points.append(pt)
    
    # Grid search
    c0_vals = np.linspace(-3.0, 3.0, 13)
    c1_vals = np.linspace(-3.0, 3.0, 13)
    c2_vals = np.linspace(-0.05, 0.05, 11)
    for c0v in c0_vals:
        for c1v in c1_vals:
            for c2v in c2_vals:
                start_points.append(np.array([c0v, c1v, c2v]))
    
    # Evaluate all starting points and keep top candidates
    scored = []
    for pt in start_points:
        val = evaluate_value(*pt)
        scored.append((val, pt))
        if val < best_value:
            best_value = val
            best_params = pt.copy()
    
    scored.sort(key=lambda x: x[0])
    
    # Local optimization from top candidates
    top_n = min(15, len(scored))
    for i in range(top_n):
        if time.time() > deadline - 20:
            break
        
        val, pt = scored[i]
        try:
            result = minimize(
                objective,
                pt,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'maxiter': 1000,
                    'maxfev': 2000
                }
            )
            if result.fun < best_value:
                best_value = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Further refinement
    if time.time() < deadline - 10:
        try:
            result = minimize(
                objective,
                best_params,
                method='Powell',
                options={
                    'xtol': 1e-12,
                    'ftol': 1e-14,
                    'maxiter': 1000,
                    'maxfev': 10000
                }
            )
            if result.fun < best_value:
                best_value = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Final Nelder-Mead refinement
    if time.time() < deadline - 3:
        try:
            result = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'maxiter': 3000,
                    'maxfev': 10000
                }
            )
            if result.fun < best_value:
                best_value = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)

# EVOLVE-BLOCK-END
