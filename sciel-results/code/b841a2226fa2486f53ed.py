"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Hermite polynomial constants
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 168.0
H12_0 = 18480.0

def compute_q_y_coeffs(c0, c1, c2):
    """Compute coefficients of q(y) where y=x^2, degree 5.
    Returns [a0, a1, a2, a3, a4, a5] for q(y) = a0 + a1*y + ... + a5*y^5."""
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    a0 = c1 * (-48) + c2 * (-1344) + c3 * (-921600)
    a1 = c1 * 16 + c2 * 2688 + c3 * 1424640
    a2 = c2 * (-1344) + c3 * (-1003520)
    a3 = c2 * 256 + c3 * 332160
    a4 = c3 * (-46080)
    a5 = c3 * 4096
    
    return np.array([a0, a1, a2, a3, a4, a5])

def evaluate_value(c0, c1, c2):
    """Compute r^2/(2*pi) where r is the largest sign-changing root of q(x)=g(x)/x^2.
    In y-space, this is y_max/(2*pi) where y_max is the largest positive sign-changing root."""
    coeffs = compute_q_y_coeffs(c0, c1, c2)
    
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # If c3 < 0, g -> -inf, so we replace g by -g, which negates all coefficients
    if c3 < 0:
        coeffs = -coeffs
        c3 = -c3
    
    if c3 <= 0:
        return 1e10
    
    # Find real roots of q(y) = 0 for y > 0
    # Use np.roots on the polynomial in standard form [a5, a4, a3, a2, a1, a0]
    poly_coeffs = coeffs[::-1]  # [a5, a4, a3, a2, a1, a0]
    roots = np.roots(poly_coeffs)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    
    # Filter for positive real roots
    positive_roots = real_roots[real_roots > 1e-10]
    
    if len(positive_roots) == 0:
        return 1e10
    
    # For each positive root, check if q changes sign (odd multiplicity)
    sign_change_roots = []
    for r in positive_roots:
        eps = 1e-8 * max(1.0, abs(r))
        q_left = np.polyval(poly_coeffs, r - eps)
        q_right = np.polyval(poly_coeffs, r + eps)
        if q_left * q_right < 0:
            sign_change_roots.append(r)
    
    if len(sign_change_roots) == 0:
        return 1e10
    
    y_max = max(sign_change_roots)
    return y_max / (2.0 * np.pi)

def objective(params):
    """Objective: minimize the value."""
    try:
        val = evaluate_value(params[0], params[1], params[2])
        if np.isnan(val) or np.isinf(val):
            return 1e10
        return val
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 170  # Safety margin before 210s kill
    
    # Seed
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best = seed.copy()
    best_val = evaluate_value(*seed)
    
    # Strategy 1: Differential evolution with wider bounds
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        try:
            bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.5, 0.5)]
            result = differential_evolution(
                objective, bounds,
                maxiter=80,
                popsize=20,
                tol=1e-8,
                seed=42,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=False
            )
            if result.fun < best_val:
                best = result.x
                best_val = result.fun
        except Exception:
            pass
    
    # Strategy 2: Local refinement from best point
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        try:
            nm_result = minimize(
                objective, best,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            if nm_result.fun < best_val:
                best = nm_result.x
                best_val = nm_result.fun
        except Exception:
            pass
    
    # Strategy 3: Multiple random restarts with local optimization
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        rng = np.random.RandomState(123)
        for i in range(40):
            if time.time() > deadline - 20:
                break
            # Generate diverse starting points
            scale = 0.1 + 0.5 * rng.rand()
            x0 = seed + rng.normal(0, scale, 3)
            try:
                local_result = minimize(
                    objective, x0,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True}
                )
                if local_result.fun < best_val:
                    best = local_result.x
                    best_val = local_result.fun
            except Exception:
                pass
    
    # Strategy 4: Final fine-tuning
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        try:
            final_result = minimize(
                objective, best,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True}
            )
            if final_result.fun < best_val:
                best = final_result.x
                best_val = final_result.fun
        except Exception:
            pass
    
    return best
# EVOLVE-BLOCK-END
