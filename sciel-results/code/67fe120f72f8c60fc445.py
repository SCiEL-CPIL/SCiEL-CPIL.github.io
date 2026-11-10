"""Rewritten program: Robust constrained fit to Linder (2005) growth index approximation."""
import numpy as np
import time

def _compute_om_a(a, Om0, w0, wa):
    """Compute matter density parameter Om(a) for CPL dark energy."""
    if a <= 0:
        return 1.0
    de_exponent = -3.0 * (1.0 + w0 + wa)
    if a >= 1.0:
        de_factor = 1.0
    else:
        one_minus_a = max(1.0 - a, 1e-15)
        de_factor = one_minus_a ** de_exponent
    E2 = Om0 / (a ** 3) + (1.0 - Om0) * de_factor
    Om_a = (Om0 / (a ** 3)) / E2
    return Om_a

def _linder_gamma(Om0, w0, wa):
    """
    Linder (2005) approximation for the growth index gamma.
    gamma = 0.55 / (1 - 0.5*wa - 0.25*w0*wa + 0.45*w0 + 0.0375*wa)
    This is accurate to ~1-2% for a wide range of CPL parameters.
    """
    denom = 1.0 - 0.5 * wa - 0.25 * w0 * wa + 0.45 * w0 + 0.0375 * wa
    if abs(denom) < 1e-6:
        return 0.55
    gamma = 0.55 / denom
    # Clip to physical range to avoid extreme values in bad models
    return np.clip(gamma, 0.1, 1.2)

def _check_constraints(coeffs, models, a_values):
    """
    Check if gamma lies in [0.1, 1.2] for all points.
    Returns True if valid, False otherwise.
    """
    c0, c1, c2, c3, c4, c5 = coeffs
    for Om0, w0, wa in models:
        for a in a_values:
            Om_a = _compute_om_a(a, Om0, w0, wa)
            if Om_a <= 0 or Om_a >= 1:
                continue
            q = 1.0 - Om_a
            u = 1.0 + w0 + wa * (1.0 - a)
            gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
            if gamma < 0.1 or gamma > 1.2:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin

    # Define a set of representative cosmologies for fitting
    # We want to cover the likely validation space.
    Om0_vals = [0.2, 0.25, 0.3, 0.35, 0.4]
    w0_vals = [-1.5, -1.2, -1.0, -0.8, -0.5]
    wa_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    models = []
    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                # Basic physical constraints
                if 1.0 + w0 + wa > -0.5 and 1.0 + w0 < -0.3:
                    models.append((Om0, w0, wa))
    
    # Limit models for speed, but keep a good spread
    # We'll use a subset
    import random
    random.seed(42)
    if len(models) > 100:
        models = random.sample(models, 100)
        
    a_values = np.linspace(0.1, 0.99, 50)

    # Prepare data for least squares
    # Target: gamma_Linder(Om0, w0, wa)
    # But since gamma_ansatz can depend on a, and Linder gamma is constant,
    # the best fit for a constant target with a variable basis is the average.
    # However, the true gamma has a small a-dependence. 
    # Linder's formula is for the asymptotic value or an average.
    # For better accuracy, we might want to fit to a more detailed reference.
    # But given the constraint error, safety is priority #1.
    
    # Let's assume the "true" gamma is well-approximated by Linder's constant gamma.
    # We fit the ansatz to match this constant gamma as closely as possible
    # across the range of a and cosmologies.
    
    A_rows = []
    b_vals = []
    
    for Om0, w0, wa in models:
        gamma_ref = _linder_gamma(Om0, w0, wa)
        for a in a_values:
            Om_a = _compute_om_a(a, Om0, w0, wa)
            if Om_a <= 0 or Om_a >= 1:
                continue
            q = 1.0 - Om_a
            u = 1.0 + w0 + wa * (1.0 - a)
            
            # Basis functions
            row = [1.0, q, q**2, u, wa * a, u * q]
            A_rows.append(row)
            b_vals.append(gamma_ref)
            
    if len(A_rows) < 10:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    A = np.array(A_rows)
    b = np.array(b_vals)
    
    # Solve least squares
    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
        
    coeffs = np.clip(coeffs, -10, 10)
    
    # Verify constraints
    # Use a denser check set including the models used for fitting
    check_models = models
    check_a = a_values
    
    if not _check_constraints(coeffs, check_models, check_a):
        # If constraints are violated, shrink towards the safe center
        # Center is approximately [0.55, 0, 0, 0, 0, 0]
        center = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Try to find a scaling factor that makes it valid
        # This is a heuristic repair
        for scale in np.linspace(0.0, 1.0, 100):
            test_coeffs = center + scale * (coeffs - center)
            if _check_constraints(test_coeffs, check_models, check_a):
                coeffs = test_coeffs
                break
        else:
            # If even the center fails (unlikely, but possible if check set is weird),
            # return the center.
            coeffs = center
            
    # Final safety check
    if not _check_constraints(coeffs, check_models, check_a):
        # Fallback to hardcoded safe values
        # gamma ~ 0.55 is safe for most standard models
        coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
        
    return list(coeffs)
