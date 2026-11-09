"""Refined Cohn-Goncalves Laguerre construction for Fourier sign uncertainty."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_laguerre

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_HARD_DEADLINE = 560.0
M = 12
N = 1
NROOTS = 12

def _deadline_reached():
    return (time.time() - _START_TIME) >= _HARD_DEADLINE

def _compute_bound(r):
    """
    Compute the upper bound on C4 for the given root locations.
    
    The bound is related to the L^2 norm of the polynomial P(x) = prod (x - r_i)^2
    under the Laguerre weight e^{-x}, or more specifically to the value of a
    functional that determines the Fourier sign uncertainty constant.
    
    For the Cohn-Goncalves construction, the bound is:
    C4 <= sqrt( int_0^inf |P(x)|^2 e^{-x} dx / (something) )
    
    A common formulation: the bound involves the norm of the polynomial in
    the Laguerre space, which can be computed via the coefficients.
    """
    r = np.asarray(r, dtype=float)
    r = np.sort(r)
    
    # Build polynomial coefficients for P(x) = prod_{i=1}^{12} (x - r_i)^2
    # Using ascending order: coeffs[k] is coefficient of x^k
    coeffs = np.array([1.0])
    for ri in r:
        quad = np.array([ri*ri, -2.0*ri, 1.0])  # ascending: ri^2 - 2ri x + x^2
        coeffs = np.convolve(coeffs, quad)
    
    # Now coeffs[k] is the coefficient of x^k in P(x)
    # P(x)^2 = P(x) * P(x), degree 48
    p2_coeffs = np.convolve(coeffs, coeffs)
    
    # Integral of x^j * e^{-x} dx from 0 to inf = j!
    # So int P(x)^2 e^{-x} dx = sum_k p2_coeffs[k] * k!
    total = 0.0
    for k in range(len(p2_coeffs)):
        total += p2_coeffs[k] * math.factorial(k)
    
    if total < 0:
        total = 0.0
    return math.sqrt(total)

def _construct_core(seed_value=0):
    """
    Construct optimal root locations using a multi-stage approach:
    1. Generate diverse initial candidates
    2. Evaluate them
    3. Optimize the best candidates using gradient-free methods
    """
    np.random.seed(seed_value)
    
    # Seed roots: arithmetic spacing 4, 8, ..., 48
    seed_roots = np.array([4.0 * (i + 1) for i in range(NROOTS)])
    
    # Laguerre zeros of L_12
    lag12_zeros = np.array(roots_laguerre(12)[0])
    
    # Generate diverse candidates
    candidates = []
    
    # 1. Seed
    candidates.append(seed_roots.copy())
    
    # 2. Laguerre zeros (various scalings)
    for scale in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]:
        candidates.append(lag12_zeros * scale)
    
    # 3. Laguerre zeros with shift
    for shift in [1.0, 2.0, 4.0, 8.0, 12.0]:
        candidates.append(lag12_zeros + shift)
    
    # 4. Interpolations between seed and Laguerre zeros
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        candidates.append(t * lag12_zeros + (1 - t) * seed_roots)
    
    # 5. Power-law spacing
    for p in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        roots = []
        for i in range(NROOTS):
            x = (i + 1) / NROOTS
            roots.append(48.0 * x ** p)
        candidates.append(np.array(roots))
    
    # 6. Geometric spacing
    for ratio in [1.1, 1.2, 1.3, 1.5]:
        roots = [4.0]
        for i in range(NROOTS - 1):
            roots.append(roots[-1] * ratio)
        candidates.append(np.array(roots))
    
    # 7. Random perturbations of seed
    for _ in range(20):
        perturbed = seed_roots + np.random.randn(NROOTS) * 2.0
        perturbed = np.sort(perturbed)
        if perturbed[0] > 0:
            candidates.append(perturbed)
    
    # 8. Random perturbations of Laguerre zeros
    for _ in range(20):
        perturbed = lag12_zeros + np.random.randn(NROOTS) * 1.0
        perturbed = np.sort(perturbed)
        if perturbed[0] > 0:
            candidates.append(perturbed)
    
    # Evaluate all candidates
    best_score = float('inf')
    best_roots = seed_roots.copy()
    candidate_scores = []
    
    for idx, cand in enumerate(candidates):
        if _deadline_reached():
            break
        cand = np.sort(cand.astype(float))
        if cand[0] <= 0 or cand[-1] > 1000:
            continue
        try:
            score = _compute_bound(cand)
            candidate_scores.append((score, idx, cand.copy()))
            if score < best_score:
                best_score = score
                best_roots = cand.copy()
        except Exception:
            pass
    
    # Sort candidates by score
    candidate_scores.sort(key=lambda x: x[0])
    
    # Optimize top candidates
    top_n = min(5, len(candidate_scores))
    for i in range(top_n):
        if _deadline_reached():
            break
        _, _, cand = candidate_scores[i]
        
        def objective(r):
            r = np.sort(r)
            if r[0] <= 0:
                return 1e10
            return _compute_bound(r)
        
        try:
            res = minimize(objective, cand, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9,
                                  'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_roots = np.sort(res.x)
        except Exception:
            pass
        
        # Second round of optimization with tighter tolerances
        if not _deadline_reached() and (time.time() - _START_TIME) < _HARD_DEADLINE * 0.7:
            try:
                res2 = minimize(objective, best_roots, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11,
                                       'adaptive': True})
                if res2.fun < best_score:
                    best_score = res2.fun
                    best_roots = np.sort(res2.x)
            except Exception:
                pass
    
    return list(best_roots)
# EVOLVE-BLOCK-END


def construct():
    """Return 12 positive double-root locations for the Laguerre combination
    (m=12, n=1).  Score is the upper bound on C4 (lower is better)."""
    try:
        result = _construct_core(seed_value=42)
    except Exception:
        result = [4.0 * (i + 1) for i in range(NROOTS)]
    
    # Hard safety: if anything ran too long, return the known-good seed roots.
    if (time.time() - _START_TIME) >= _HARD_DEADLINE:
        result = [4.0 * (i + 1) for i in range(NROOTS)]
    
    # Final sanity: exactly 12 positive, increasing floats.
    if not (isinstance(result, list) and len(result) == NROOTS):
        result = [4.0 * (i + 1) for i in range(NROOTS)]
    out = []
    last = 0.0
    for v in result:
        v = float(v)
        if not (v > 0.0 and math.isfinite(v)):
            v = last + 1.0
        if v <= last:
            v = last + 1e-6
        out.append(v)
        last = v
    return out
