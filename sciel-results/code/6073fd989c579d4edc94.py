import numpy as np
import time
from scipy.optimize import minimize

# Precompute Hermite polynomial coefficients for H0, H4, H8, H12
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
# H12(x) = 4096x^12 - 122880x^10 + 1397760x^8 - 6812160x^6 + 13977600x^4 - 11827200x^2 + 4084080

def compute_value(coeffs):
    c0, c1, c2 = coeffs
    c3 = -(c0 * 1.0 + c1 * 12.0 + c2 * 1680.0) / 4084080.0
    
    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # g(x)/x^2 = q(x^2), so let y = x^2, q(y) = g(sqrt(y))/y
    # Coefficients of g(x) in powers of x^2:
    # x^0: c0*1 + c1*12 + c2*1680 + c3*4084080 = 0 (by construction)
    # x^2: c1*(-48) + c2*(-13440) + c3*(-11827200)
    # x^4: c1*16 + c2*11520 + c3*13977600
    # x^6: c2*(-3072) + c3*(-6812160)
    # x^8: c2*256 + c3*1397760
    # x^10: c3*(-122880)
    # x^12: c3*4096
    
    A = -48.0 * c1 - 13440.0 * c2 - 11827200.0 * c3
    B = 16.0 * c1 + 11520.0 * c2 + 13977600.0 * c3
    C = -3072.0 * c2 - 6812160.0 * c3
    D = 256.0 * c2 + 1397760.0 * c3
    E = -122880.0 * c3
    F = 4096.0 * c3
    
    coeffs_q = [F, E, D, C, B, A]
    
    # Find roots of q(y)
    roots = np.roots(coeffs_q)
    
    valid_roots = []
    for root in roots:
        if abs(root.imag) < 1e-10:
            r = root.real
            if r > 1e-12:
                eps = max(1e-8, abs(r) * 1e-8)
                q_minus = np.polyval(coeffs_q, r - eps)
                q_plus = np.polyval(coeffs_q, r + eps)
                if q_minus * q_plus < 0:
                    valid_roots.append(r)
            
    if not valid_roots:
        return 1e9
        
    r_max = max(valid_roots)
    return r_max ** 2 / (2.0 * np.pi)

def _g_values(coeffs, x_grid):
    """Evaluate g(x) at points in x_grid. Returns array."""
    c0, c1, c2 = coeffs
    c3 = -(c0 * 1.0 + c1 * 12.0 + c2 * 1680.0) / 4084080.0
    
    x2 = x_grid ** 2
    x4 = x2 ** 2
    x6 = x4 * x2
    x8 = x4 ** 2
    x10 = x8 * x2
    x12 = x8 * x4
    
    g = (c0 * 1.0 +
         c1 * (16.0 * x4 - 48.0 * x2 + 12.0) +
         c2 * (256.0 * x8 - 3072.0 * x6 + 11520.0 * x4 - 13440.0 * x2 + 1680.0) +
         c3 * (4096.0 * x12 - 122880.0 * x10 + 1397760.0 * x8 - 6812160.0 * x6 + 13977600.0 * x4 - 11827200.0 * x2 + 4084080.0))
    return g

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # safety margin before 210s kill
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    seed_val = compute_value(seed)
    best_val = seed_val
    best_coeffs = seed.copy()
    
    # Dense grid for positivity check
    # g(x) >= 0 for all x >= 0. Since g(0)=0 and g is a polynomial in x^2,
    # we need g(x) >= 0 for x >= 0.
    # Use a grid that covers the relevant range. The root r gives x_max = sqrt(r_max).
    # We'll use a grid up to a generous bound.
    x_max_grid = 20.0
    n_grid = 2000
    x_grid = np.linspace(0.0, x_max_grid, n_grid)
    
    beta = 100.0 # Sharpness of softplus penalty
    
    def objective(params):
        """Smooth penalty-based objective: minimize r^2 subject to g(x) >= 0."""
        c0, c1, c2 = params
        coeffs = [c0, c1, c2]
        
        # Compute the actual value
        val = compute_value(coeffs)
        
        # Smooth positivity penalty: sum(softplus(-beta * g(x)))
        g_vals = _g_values(coeffs, x_grid)
        # g(0) = 0 by construction, so skip x=0 to avoid log(1) = 0 which is fine, 
        # but let's be consistent. softplus(-beta*0) = log(2)/beta ~ 0.0069, small.
        # Actually, we want to penalize negative values.
        # softplus(t) = log(1+exp(t)). For t -> -inf, softplus(t) -> 0. For t -> inf, softplus(t) -> t.
        # We want penalty for g(x) < 0. Let t = -beta * g(x).
        # If g(x) < 0, t > 0, penalty grows.
        # If g(x) > 0, t < 0, penalty is small.
        
        # To avoid overflow in exp for large positive t, we can use a stable softplus
        # or just clip. But beta=100 and g can be large.
        # Stable softplus: log(1+exp(x)) = max(x,0) + log(1+exp(-abs(x)))
        t = -beta * g_vals
        # Clip t to avoid overflow in exp
        t_clipped = np.clip(t, -50, 50)
        penalty = np.sum(np.log1p(np.exp(t_clipped)))
        
        return val + penalty
    
    # Multi-start Nelder-Mead optimization
    rng = np.random.default_rng(42)
    n_starts = 100
    
    # Create a set of initial points: seed, random, and grid-based
    start_points = [seed.copy()]
    
    # Add some random points
    for i in range(50):
        if time.time() > deadline - 20.0:
            break
        start_points.append(rng.uniform(-2.0, 2.0, size=3))
        
    # Add points near the seed
    for i in range(50):
        if time.time() > deadline - 20.0:
            break
        start_points.append(seed + rng.normal(0, 0.5, size=3))

    for start in start_points:
        if time.time() > deadline - 10.0:
            break
        
        try:
            # Use Nelder-Mead with adaptive step size
            res = minimize(objective, start, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8,
                                    'adaptive': True})
            
            if res.fun < best_val:
                # Verify the result is valid (g >= 0) with a tighter check
                g_check = _g_values(res.x, x_grid)
                if np.min(g_check[1:]) >= -1e-5:  # allow tiny numerical error
                    # Polish: run a very tight optimization from this point
                    res_polish = minimize(objective, res.x, method='Nelder-Mead',
                                          options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10,
                                                   'adaptive': True})
                    if res_polish.fun < best_val:
                        g_polish = _g_values(res_polish.x, x_grid)
                        if np.min(g_polish[1:]) >= -1e-5:
                            best_val = res_polish.fun
                            best_coeffs = res_polish.x.copy()
        except Exception:
            pass
    
    # Final check: if best is still seed, return seed
    if best_val >= seed_val:
        return seed
    
    return best_coeffs
