"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_value(c0, c1, c2):
    """
    Compute the value r^2/(2*pi) for given coefficients c0, c1, c2.
    
    The evaluator builds g = c0*H0 + c1*H4 + c2*H8 + c3*H12 with physicists' Hermite polynomials,
    where c3 is fixed so that g(0)=0. If g tends to a negative limit at infinity, g is replaced by -g.
    Then q = g/x^2 (must divide exactly). We find the largest real root r of q at which q changes sign,
    and return r^2/(2*pi).
    """
    # Physicists' Hermite polynomials:
    # H0(x) = 1
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1344
    # H12(x) = 4096x^12 - 122880x^10 + 1290240x^8 - 5806080x^6 + 9953280x^4 - 5505024x^2 + 1032192
    
    # Values at x=0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1344.0
    H12_0 = 1032192.0
    
    # c3 is determined by g(0) = 0
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # Coefficients of g(x) as a polynomial in x:
    # g(x) = sum_{k=0}^{6} a_k * x^{2k}
    
    a0 = 0.0  # x^0 coefficient (zero by construction)
    a1 = c1 * (-48) + c2 * (-13440) + c3 * (-5505024)  # x^2
    a2 = c1 * 16 + c2 * 11520 + c3 * 9953280  # x^4
    a3 = c2 * (-3072) + c3 * (-5806080)  # x^6
    a4 = c2 * 256 + c3 * 1290240  # x^8
    a5 = c3 * (-122880)  # x^10
    a6 = c3 * 4096  # x^12
    
    # Check if g tends to negative limit at infinity: look at leading coefficient
    # The leading term is a6 * x^12
    if a6 < 0:
        # g tends to -infinity, so replace g by -g
        a1 = -a1
        a2 = -a2
        a3 = -a3
        a4 = -a4
        a5 = -a5
        a6 = -a6
    
    # q(x) = g(x) / x^2 = a1 + a2*x^2 + a3*x^4 + a4*x^6 + a5*x^8 + a6*x^10
    # Let t = x^2, then Q(t) = a1 + a2*t + a3*t^2 + a4*t^3 + a5*t^4 + a6*t^5
    # Roots of q(x) correspond to roots of Q(t) with t = r^2 > 0
    # q changes sign at x=r iff Q changes sign at t=r^2
    
    # Find all real roots of Q(t)
    coeffs_Q = [a6, a5, a4, a3, a2, a1]  # descending order for np.roots
    
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
    real_roots = []
    for rt in roots_t:
        if abs(rt.imag) < 1e-10:
            tr = rt.real
            if tr > 1e-12:  # positive
                real_roots.append(tr)
    
    if not real_roots:
        return float('inf')
    
    # For each positive real root t_i, check if Q changes sign there
    # Q(t) = a1 + a2*t + a3*t^2 + a4*t^3 + a5*t^4 + a6*t^5
    def Q(t):
        return a1 + a2*t + a3*t**2 + a4*t**3 + a5*t**4 + a6*t**5
    
    best_r2 = float('inf')
    for ti in real_roots:
        eps = 1e-8
        val_left = Q(ti - eps)
        val_right = Q(ti + eps)
        if val_left * val_right < 0:
            # Sign change at ti, so r = sqrt(ti)
            r2 = ti
            if r2 < best_r2:
                best_r2 = r2
    
    if best_r2 == float('inf'):
        return float('inf')
    
    return best_r2 / (2 * np.pi)


def _g_values(c0, c1, c2, x_vals):
    """Evaluate g(x) at array of x values. Returns g(x) array."""
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1344.0
    H12_0 = 1032192.0
    
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    x2 = x_vals ** 2
    x4 = x2 ** 2
    x6 = x2 * x4
    x8 = x4 ** 2
    x10 = x2 * x8
    x12 = x4 * x8
    
    g = (c0 * 1.0 +
         c1 * (16*x4 - 48*x2 + 12) +
         c2 * (256*x8 - 3072*x6 + 11520*x4 - 13440*x2 + 1344) +
         c3 * (4096*x12 - 122880*x10 + 1290240*x8 - 5806080*x6 + 9953280*x4 - 5505024*x2 + 1032192))
    
    return g


def construct():
    """Return three coefficients [c0, c1, c2] that minimize the value."""
    start_time = time.time()
    deadline = start_time + 170  # Safety margin before 210s kill
    
    # Seed values
    seed = np.array([-113/100, 1/25, 1/3240])
    seed_value = _compute_value(seed[0], seed[1], seed[2])
    best_coeffs = seed.copy()
    best_value = seed_value
    
    # Precompute grid for penalty evaluation
    x_grid = np.linspace(0.01, 10.0, 500)
    
    def objective(x):
        try:
            val = _compute_value(x[0], x[1], x[2])
            # Add penalty for g(x) < 0 on the grid
            g_vals = _g_values(x[0], x[1], x[2], x_grid)
            # Check leading coefficient for sign
            c3 = -(x[0] * 1.0 + x[1] * 12.0 + x[2] * 1344.0) / 1032192.0
            a6 = c3 * 4096
            if a6 < 0:
                g_vals = -g_vals
            penalty = np.sum(np.maximum(0, -g_vals)) ** 2
            return val + 1000.0 * penalty
        except Exception:
            return 1e10
    
    # Strategy 1: Nelder-Mead from seed with penalty
    if time.time() < deadline - 60:
        try:
            result = minimize(objective, seed, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if result.fun < best_value + 100:  # Allow some penalty slack
                val = _compute_value(result.x[0], result.x[1], result.x[2])
                if val < best_value:
                    best_value = val
                    best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Strategy 2: Try random perturbations around seed
    rng = np.random.RandomState(42)
    if time.time() < deadline - 40:
        for _ in range(50):
            if time.time() > deadline - 35:
                break
            perturbation = rng.normal(0, 0.03, size=3)
            x0 = seed + perturbation
            val = _compute_value(x0[0], x0[1], x0[2])
            if val < best_value:
                best_value = val
                best_coeffs = x0.copy()
    
    # Strategy 3: Local refinement from best found with penalty
    if time.time() < deadline - 20:
        try:
            result = minimize(objective, best_coeffs, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            val = _compute_value(result.x[0], result.x[1], result.x[2])
            if val < best_value:
                best_value = val
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Strategy 4: Fine local search around best
    if time.time() < deadline - 10:
        for scale in [0.01, 0.005, 0.001]:
            if time.time() > deadline - 5:
                break
            for i in range(-3, 4):
                for j in range(-3, 4):
                    for k in range(-3, 4):
                        if time.time() > deadline - 5:
                            break
                        x0 = best_coeffs + scale * np.array([i, j, k])
                        val = _compute_value(x0[0], x0[1], x0[2])
                        if val < best_value:
                            best_value = val
                            best_coeffs = x0.copy()
    
    # Final polish
    if time.time() < deadline - 3:
        try:
            result = minimize(objective, best_coeffs, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
            val = _compute_value(result.x[0], result.x[1], result.x[2])
            if val < best_value:
                best_value = val
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
