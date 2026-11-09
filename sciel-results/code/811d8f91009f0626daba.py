"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.special import factorial, gamma

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_HARD_DEADLINE = 560.0   # safety margin before the 630 s wall-clock kill
M = 12
N = 1
NROOTS = 12

def _deadline_reached():
    return (time.time() - _START_TIME) >= _HARD_DEADLINE

def _compute_bound(r):
    """
    Compute the actual uncertainty bound for given double-root locations.
    
    For the Cohn-Goncalves construction with m=12, n=1, we have a polynomial
    P(x) = prod_{i=1}^{12} (x - r_i)^2.
    
    The uncertainty bound C4 involves integrals of the form:
    C4 = [integral_0^inf x^4 e^{-x} |P(x)|^2 dx] / [integral_0^inf e^{-x} |P(x)|^2 dx]
    
    Or more precisely, it's related to the ratio of moments. Let's compute
    the actual bound using numerical quadrature.
    """
    r = np.asarray(r, dtype=float)
    
    # Compute the polynomial P(x) = prod (x - r_i)^2
    # We'll evaluate it numerically on a grid for quadrature
    # Use Gauss-Laguerre quadrature for integrals with e^{-x} weight
    
    # Gauss-Laguerre nodes and weights for degree 24 (since P has degree 24)
    # Actually, we need enough points to resolve the polynomial
    x_nodes, x_weights = np.polynomial.laguerre.laggauss(50)
    
    # Evaluate P(x) at each node
    P_vals = np.ones_like(x_nodes)
    for ri in r:
        P_vals *= (x_nodes - ri) ** 2
    
    # Compute integrals
    # Numerator: integral x^4 e^{-x} |P(x)|^2 dx
    # Denominator: integral e^{-x} |P(x)|^2 dx
    P2 = P_vals ** 2
    
    num = np.sum(x_weights * x_nodes**4 * P2)
    den = np.sum(x_weights * P2)
    
    if den < 1e-30:
        return 1e10  # very large penalty
    
    # The bound C4 is related to this ratio
    # For the seed (4,8,...,48), this should give score 1.0
    # Let's return the ratio as the objective to minimize
    return num / den

def _construct_core(seed_value):
    """
    Optimize root positions to minimize the actual uncertainty bound.
    """
    np.random.seed(seed_value)
    
    # Start with the seed configuration
    best_r = np.array([4.0 * (i + 1) for i in range(NROOTS)], dtype=float)
    best_score = _compute_bound(best_r)
    
    # Try gradient-based optimization
    def objective(r_flat):
        if _deadline_reached():
            return best_score
        r = np.asarray(r_flat, dtype=float)
        # Enforce positivity and ordering
        r = np.maximum.accumulate(np.maximum(r, 0.1))
        # Penalize if not strictly increasing
        gaps = np.diff(r)
        if np.any(gaps < 1e-6):
            return best_score + 1e6
        return _compute_bound(r)
    
    # Try multiple starting points
    starts = [best_r.copy()]
    
    # Add some perturbed starts
    for _ in range(10):
        pert = best_r + np.random.randn(NROOTS) * 0.5
        pert = np.maximum.accumulate(np.maximum(pert, 0.1))
        starts.append(pert.copy())
    
    # Try different initial spreads
    for scale in [0.5, 1.0, 1.5, 2.0]:
        cand = scale * np.array([4.0 * (i + 1) for i in range(NROOTS)])
        if cand[-1] < 200 and cand[0] > 0.5:
            starts.append(cand.copy())
    
    for start in starts:
        if _deadline_reached():
            break
        
        try:
            result = minimize(
                objective,
                start,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_score:
                best_r = result.x.copy()
                best_score = result.fun
        except Exception:
            pass
    
    return best_r.tolist()
# EVOLVE-BLOCK-END


def construct():
    """Return 12 positive double-root locations for the Laguerre combination
    (m=12, n=1).  Score is the upper bound on C4 (lower is better)."""
    # The seed (arithmetic spacing 4,8,...,48) scores 1.0.
    # Previous optimization attempts made the score worse (0.9817).
    # Let's try a few simple alternative patterns and see if any beat the seed.
    
    try:
        result = _construct_core(seed_value=0)
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
