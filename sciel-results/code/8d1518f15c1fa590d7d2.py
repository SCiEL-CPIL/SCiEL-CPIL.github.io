"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_value(c0, c1, c2):
    """Compute value = r^2/(2*pi) for given coefficients c0, c1, c2."""
    # Physicists' Hermite polynomial values at x=0:
    # H0(0) = 1, H4(0) = 12, H8(0) = 1344, H12(0) = 1032192
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1344.0
    H12_0 = 1032192.0
    
    # c3 from g(0) = 0
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # Q(t) = g(x)/x^2 where t = x^2
    # H0 = 1, H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1344
    # H12 = 4096x^12 - 122880x^10 + 1290240x^8 - 5806080x^6 + 9953280x^4 - 5505024x^2 + 1032192
    #
    # g = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # g/x^2 = Q(t) where t=x^2:
    # Constant term (x^0 in g -> x^-2 in q... wait, g(0)=0 so g has no constant term)
    # Actually g(0) = c0*1 + c1*12 + c2*1344 + c3*1032192 = 0 by construction
    # So g = (terms with x^2 and higher)
    # g/x^2 = Q(t) where t = x^2
    #
    # H0/x^2: H0 = 1, but we need g(0)=0 so this is handled by c3
    # Let's compute Q(t) directly:
    # g(x) = c0*(1) + c1*(16x^4-48x^2+12) + c2*(256x^8-3072x^6+11520x^4-13440x^2+1344) + c3*(4096x^12-122880x^10+1290240x^8-5806080x^6+9953280x^4-5505024x^2+1032192)
    #
    # g(0) = c0 + 12*c1 + 1344*c2 + 1032192*c3 = 0 (by construction)
    #
    # g/x^2 in terms of t=x^2:
    # The x^2 coefficient of g: -48*c1 - 13440*c2 - 5505024*c3
    # The x^4 coefficient: 16*c1 + 11520*c2 + 9953280*c3
    # The x^6 coefficient: -3072*c2 - 5806080*c3
    # The x^8 coefficient: 256*c2 + 1290240*c3
    # The x^10 coefficient: -122880*c3
    # The x^12 coefficient: 4096*c3
    #
    # So Q(t) = 4096*c3*t^5 - 122880*c3*t^4 + (256*c2+1290240*c3)*t^3 + (-3072*c2-5806080*c3)*t^2 + (16*c1+11520*c2+9953280*c3)*t + (-48*c1-13440*c2-5505024*c3)
    
    a5 = 4096.0 * c3
    a4 = -122880.0 * c3
    a3 = 256.0 * c2 + 1290240.0 * c3
    a2 = -3072.0 * c2 - 5806080.0 * c3
    a1 = 16.0 * c1 + 11520.0 * c2 + 9953280.0 * c3
    a0 = -48.0 * c1 - 13440.0 * c2 - 5505024.0 * c3
    
    # Check: g tends to negative limit at infinity?
    # Leading coefficient of g is a5 (coefficient of x^12 = t^6... wait)
    # Actually g is degree 12 in x, so leading coeff is 4096*c3
    # If 4096*c3 < 0, g -> -inf, so we replace g by -g, which flips Q to -Q
    # This doesn't change roots but flips sign changes... actually it doesn't matter for root locations
    # The sign change condition: q changes sign at r means Q changes sign at t=r^2
    # If we flip g to -g, Q becomes -Q, sign changes remain the same
    
    # Find roots of Q(t) = a5*t^5 + a4*t^4 + a3*t^3 + a2*t^2 + a1*t + a0
    coeffs_Q = np.array([a5, a4, a3, a2, a1, a0], dtype=np.float64)
    
    # Remove leading zeros
    while len(coeffs_Q) > 1 and abs(coeffs_Q[0]) < 1e-15:
        coeffs_Q = coeffs_Q[1:]
    
    if len(coeffs_Q) <= 1:
        return float('inf')
    
    try:
        roots_t = np.roots(coeffs_Q)
    except Exception:
        return float('inf')
    
    # Filter for real positive roots
    real_positive_roots = []
    for rt in roots_t:
        if abs(rt.imag) < 1e-10:
            tr = float(rt.real)
            if tr > 1e-12:
                real_positive_roots.append(tr)
    
    if not real_positive_roots:
        return float('inf')
    
    # For each positive real root t_i, check if Q changes sign at t_i
    # Q changes sign at t_i iff the root has odd multiplicity
    # We can check by evaluating Q at t_i - eps and t_i + eps
    
    best_t = None
    for ti in sorted(real_positive_roots):
        # Check sign change
        eps = max(1e-8, ti * 1e-10)
        val_left = np.polyval(coeffs_Q, ti - eps)
        val_right = np.polyval(coeffs_Q, ti + eps)
        if val_left * val_right < 0:
            # Sign change at ti
            if best_t is None or ti > best_t:
                best_t = ti
    
    if best_t is None:
        return float('inf')
    
    return best_t / (2.0 * np.pi)


def construct():
    """Return three coefficients [c0, c1, c2] that minimize the value."""
    start_time = time.time()
    deadline = start_time + 170  # Safety margin before 210s kill
    
    # Seed values
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    seed_value = _compute_value(seed[0], seed[1], seed[2])
    best_coeffs = seed.copy()
    best_value = seed_value
    
    def objective(x):
        return _compute_value(x[0], x[1], x[2])
    
    # Multi-start Nelder-Mead optimization
    rng = np.random.RandomState(42)
    
    # Generate diverse starting points
    initial_points = [seed.copy()]
    
    # Perturbed seeds
    for scale in [0.05, 0.1, 0.2, 0.5]:
        for _ in range(3):
            perturbation = rng.normal(0, scale, size=3)
            initial_points.append(seed + perturbation)
    
    # Some random points in a broader range
    for _ in range(5):
        x = rng.uniform(-1, 1, size=3)
        initial_points.append(x)
    
    # Try each starting point with Nelder-Mead
    for x0 in initial_points:
        if time.time() > deadline - 10:
            break
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Polish the best result with tighter tolerances
    if time.time() < deadline - 5:
        try:
            result = minimize(objective, best_coeffs, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
