"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195  # 15s margin before 210s kill

# Precompute Hermite polynomial coefficients (Physicist's convention)
# H_n(x) = sum_k c[n][k] * x^(2k) for even n
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
_H_COEFFS = {
    0: [1.0],
    4: [3.0, -6.0, 1.0],
    8: [105.0, -420.0, 210.0, -28.0, 1.0],
    12: [945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0],
}
# H_n(0) values
_H0_0, _H4_0, _H8_0, _H12_0 = 1.0, 3.0, 105.0, 945.0


def _poly_coeffs(a0, a4, a8):
    """
    Given a12 = -(a0*H0_0 + a4*H4_0 + a8*H8_0)/H12_0,
    return the coefficient array c[k] for P(x) = sum_k c[k]*x^(2k).
    """
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    # P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    # Each H_n contributes to powers x^0, x^2, ..., x^n
    # Max degree is 12, so powers 0,2,4,6,8,10,12 -> 7 coefficients
    c = np.zeros(7)
    for n, a in [(0, a0), (4, a4), (8, a8), (12, a12)]:
        coeffs = _H_COEFFS[n]
        for k, val in enumerate(coeffs):
            c[k] += a * val
    return c


def _integrate_moment(c, m):
    """
    Compute ∫_{-∞}^{∞} x^(2m) * P(x)^2 * exp(-x^2) dx
    where P(x) = sum_{k=0}^{6} c[k] * x^(2k).
    
    P(x)^2 = sum_{i,j} c[i]*c[j] * x^(2(i+j))
    x^(2m) * P(x)^2 = sum_{i,j} c[i]*c[j] * x^(2(i+j+m))
    ∫ x^(2n) * exp(-x^2) dx = (2n-1)!! * sqrt(pi) / 2^n = Gamma(n+1/2)
    
    So the integral = sum_{i,j} c[i]*c[j] * Gamma(i+j+m+1/2)
    """
    from math import gamma, sqrt, pi
    result = 0.0
    n = len(c)
    for i in range(n):
        if c[i] == 0.0:
            continue
        for j in range(i, n):  # symmetry
            if c[j] == 0.0:
                continue
            power = i + j + m  # x^(2*power)
            integral = gamma(power + 0.5)
            if i == j:
                result += c[i] * c[j] * integral
            else:
                result += 2.0 * c[i] * c[j] * integral
    return result


def _compute_delta_x_squared_fast(a0, a4, a8):
    """
    Compute (Δx)² = <x²>/||g||² analytically using exact polynomial-Gaussian integrals.
    """
    c = _poly_coeffs(a0, a4, a8)
    norm_sq = _integrate_moment(c, 0)
    if norm_sq <= 0 or not np.isfinite(norm_sq):
        return None
    x2_norm = _integrate_moment(c, 1)
    if not np.isfinite(x2_norm):
        return None
    delta_x_sq = x2_norm / norm_sq
    if delta_x_sq < 0 or not np.isfinite(delta_x_sq):
        return None
    return delta_x_sq


def _objective(x):
    """Objective function for optimization."""
    if time.time() > _DEADLINE:
        return np.inf
    v = _compute_delta_x_squared_fast(x[0], x[1], x[2])
    if v is None:
        return np.inf
    return v


def construct():
    """Find optimal coefficients to minimize the uncertainty product bound."""
    start_time = time.time()

    # Seed from Goncalves et al. 2017
    seed_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_coeffs = seed_coeffs.copy()
    best_val = _compute_delta_x_squared_fast(*best_coeffs)
    if best_val is None:
        best_val = np.inf

    def time_ok(margin=20):
        return (time.time() - start_time) < (195 - margin)

    # Strategy 1: Broad multi-start with diverse scales
    rng = np.random.default_rng(42)
    
    # Define a set of diverse starting points
    starts = [best_coeffs.copy()]
    
    # Log-uniform spread around the seed
    for _ in range(80):
        if not time_ok(30):
            break
        # Mix: some near seed, some far
        if rng.random() < 0.3:
            # Close to seed
            coeffs = best_coeffs + rng.normal(0, 0.02, 3)
        elif rng.random() < 0.6:
            # Medium range
            coeffs = rng.normal(0, 0.5, 3)
        else:
            # Wide range - log-scale exploration
            scales = 10 ** rng.uniform(-3, 2, 3)
            signs = rng.choice([-1, 1], 3)
            coeffs = signs * scales * rng.uniform(0.1, 1.0, 3)
        
        val = _compute_delta_x_squared_fast(*coeffs)
        if val is not None and val < best_val:
            best_val = val
            best_coeffs = coeffs.copy()
        starts.append(coeffs.copy())

    # Strategy 2: Nelder-Mead from top diverse starts
    if time_ok(50):
        # Sort starts by objective value, take top 10
        scored_starts = []
        for s in starts:
            v = _compute_delta_x_squared_fast(*s)
            if v is not None:
                scored_starts.append((v, s))
        scored_starts.sort(key=lambda x: x[0])
        
        for _, s in scored_starts[:10]:
            if not time_ok(20):
                break
            try:
                result = minimize(_objective, s, method='Nelder-Mead',
                                  options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14,
                                           'adaptive': True})
                if np.isfinite(result.fun) and result.fun < best_val:
                    best_val = result.fun
                    best_coeffs = result.x.copy()
            except Exception:
                pass

    # Strategy 3: L-BFGS-B from best point (smooth landscape, gradient-based)
    if time_ok(30):
        try:
            result = minimize(_objective, best_coeffs, method='L-BFGS-B',
                              options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12})
            if np.isfinite(result.fun) and result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    # Strategy 4: Powell refinement
    if time_ok(25):
        try:
            result = minimize(_objective, best_coeffs, method='Powell',
                              options={'maxiter': 3000, 'xtol': 1e-14, 'ftol': 1e-16})
            if np.isfinite(result.fun) and result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    # Strategy 5: Final tight Nelder-Mead polish
    if time_ok(15):
        try:
            result = minimize(_objective, best_coeffs, method='Nelder-Mead',
                              options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-17,
                                       'adaptive': True})
            if np.isfinite(result.fun) and result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    return best_coeffs
# EVOLVE-BLOCK-END
