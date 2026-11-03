"""Refined Cohn-Goncalves Laguerre construction: optimized root search."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = 600.0
_START = time.time()


def _time_left():
    return _DEADLINE - time.time()


def _laguerre_coeffs(n, alpha=1):
    """
    Compute coefficients of L_n^(alpha)(x) in ascending power of x.
    L_n^(alpha)(x) = sum_{k=0}^n (-1)^k * C(n+alpha, n-k) * x^k / k!
    """
    coeff = np.zeros(n + 1)
    for k in range(n + 1):
        # C(n+alpha, n-k) = C(n+alpha, k+alpha)
        # For integer alpha=1: C(n+1, n-k) = C(n+1, k+1)
        # Use log-gamma for general case
        # C(n+alpha, n-k) = Gamma(n+alpha+1) / (Gamma(n-k+1) * Gamma(k+alpha+1))
        try:
            from math import lgamma
            log_val = (lgamma(n + alpha + 1) - lgamma(n - k + 1) - lgamma(k + alpha + 1))
            binom_val = math.exp(log_val)
        except Exception:
            binom_val = 1.0
        coeff[k] = ((-1) ** k) * binom_val / math.factorial(k)
    return coeff


def _compute_norm_sq(poly):
    """
    Compute ||psi||^2 = integral_0^inf psi(x)^2 x^1 e^{-x} dx
    = sum_{i,j} poly[i] poly[j] Gamma(i+j+1+1)
    = sum_k (sum_i poly[i] poly[k-i]) * (k+1)!
    """
    sq = np.convolve(poly, poly)
    max_k = len(sq) - 1
    # Precompute (k+1)! for k = 0..max_k
    # (k+1)! = (k+1) * k * ... * 1
    # fact[j] = j!
    fact = np.ones(max_k + 2)
    for j in range(1, max_k + 2):
        fact[j] = fact[j - 1] * j
    # weights[k] = (k+1)! = fact[k+1]
    weights = fact[1:max_k + 2]
    norm_sq = np.dot(sq, weights)
    return norm_sq


def _c4_bound(roots, m=12):
    """
    Compute upper bound on C4 for given roots.
    psi(x) = L_m^(1)(x) * prod_i (x - r_i)^2
    C4 <= 1 / ||psi||^2
    """
    if _time_left() < 5:
        return float('inf')
    
    # Get L_m^(1)(x) coefficients
    coeff_L = _laguerre_coeffs(m, 1)
    
    # Multiply by (x - r)^2 for each root
    poly = coeff_L.copy()
    for r in roots:
        if r <= 0:
            return float('inf')
        q = np.array([r * r, -2.0 * r, 1.0])
        poly = np.convolve(poly, q)
    
    # Check for numerical issues
    if not np.all(np.isfinite(poly)):
        return float('inf')
    
    norm_sq = _compute_norm_sq(poly)
    if not np.isfinite(norm_sq) or norm_sq <= 0:
        return float('inf')
    
    return 1.0 / norm_sq


def _make_roots(scale, alpha, n_roots=12):
    """Generate roots as scale * (i+1)^alpha for i=0..n_roots-1"""
    return [scale * ((i + 1) ** alpha) for i in range(n_roots)]


def construct():
    m = 12
    n_roots = 12
    
    # Baseline: the seed roots
    seed_roots = [4.0 * (i + 1) for i in range(n_roots)]
    seed_val = _c4_bound(seed_roots, m)
    if not np.isfinite(seed_val):
        seed_val = 1.0  # fallback
    
    best_roots = seed_roots[:]
    best_val = seed_val
    
    # Phase 1: Coarse grid over (scale, alpha) parameterization
    # The seed has scale=4, alpha=1 (linear spacing)
    scales = np.linspace(0.5, 30.0, 80)
    alphas = np.linspace(0.3, 1.5, 60)
    
    for s in scales:
        if _time_left() < 120:
            break
        for a in alphas:
            if _time_left() < 60:
                break
            roots = _make_roots(s, a, n_roots)
            if min(roots) <= 0:
                continue
            # Check distinctness
            if len(set(round(r, 10) for r in roots)) != n_roots:
                continue
            val = _c4_bound(roots, m)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_roots = roots[:]
    
    # Phase 2: Local refinement around best point
    if _time_left() > 90 and best_val < seed_val:
        # Estimate (s, a) from best_roots
        r0 = best_roots[0]
        if len(best_roots) >= 2 and best_roots[0] > 0:
            ratio = best_roots[1] / best_roots[0]
            if ratio > 1.001:
                alpha_est = math.log(ratio) / math.log(2.0)
            else:
                alpha_est = 1.0
            scale_est = r0
        else:
            scale_est = 4.0
            alpha_est = 1.0
        
        # Multi-scale refinement rings
        for ring in range(4):
            if _time_left() < 30:
                break
            step_s = 2.0 / (ring + 1)
            step_a = 0.15 / (ring + 1)
            ds_range = np.linspace(-step_s * 2, step_s * 2, 20)
            da_range = np.linspace(-step_a * 2, step_a * 2, 20)
            for ds in ds_range:
                if _time_left() < 15:
                    break
                for da in da_range:
                    if _time_left() < 5:
                        break
                    s = scale_est * (1.0 + ds)
                    a = alpha_est + da
                    if s <= 0 or a <= 0:
                        continue
                    roots = _make_roots(s, a, n_roots)
                    if min(roots) <= 0:
                        continue
                    if len(set(round(r, 10) for r in roots)) != n_roots:
                        continue
                    val = _c4_bound(roots, m)
                    if np.isfinite(val) and val < best_val:
                        best_val = val
                        best_roots = roots[:]
                        # Update center for next ring
                        ratio = best_roots[1] / best_roots[0] if best_roots[0] > 0 else 2.0
                        if ratio > 1.001:
                            scale_est = best_roots[0]
                            alpha_est = math.log(ratio) / math.log(2.0)
    
    # Phase 3: Try alternative root structures (geometric spacing, etc.)
    if _time_left() > 60:
        # Geometric: r_i = scale * ratio^i
        for ratio in np.linspace(1.05, 2.5, 40):
            if _time_left() < 30:
                break
            for scale in np.linspace(0.5, 15.0, 30):
                if _time_left() < 10:
                    break
                roots = [scale * (ratio ** i) for i in range(n_roots)]
                if min(roots) <= 0 or max(roots) > 1e6:
                    continue
                if len(set(round(r, 10) for r in roots)) != n_roots:
                    continue
                val = _c4_bound(roots, m)
                if np.isfinite(val) and val < best_val:
                    best_val = val
                    best_roots = roots[:]
    
    # Phase 4: Try clustered roots (some close together)
    if _time_left() > 45:
        # Try: first k roots clustered, rest spread
        for k in range(2, 8):
            if _time_left() < 20:
                break
            for cluster_scale in [1.0, 2.0, 4.0, 8.0]:
                if _time_left() < 10:
                    break
                for spread in [3.0, 5.0, 8.0]:
                    if _time_left() < 5:
                        break
                    roots = []
                    for i in range(n_roots):
                        if i < k:
                            roots.append(cluster_scale * (1.0 + 0.1 * i))
                        else:
                            roots.append(cluster_scale * k + spread * (i - k))
                    if min(roots) <= 0:
                        continue
                    if len(set(round(r, 10) for r in roots)) != n_roots:
                        continue
                    val = _c4_bound(roots, m)
                    if np.isfinite(val) and val < best_val:
                        best_val = val
                        best_roots = roots[:]
    
    # Ensure we return valid positive distinct roots
    if len(best_roots) != n_roots or any(r <= 0 for r in best_roots):
        best_roots = seed_roots[:]
    
    return best_roots
# EVOLVE-BLOCK-END
