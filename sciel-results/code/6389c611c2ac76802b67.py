"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_genlaguerre
import time

def _lag_val(k, x):
    """Laguerre polynomial L_k(x) for integer k >= 0."""
    return eval_genlaguerre(k, x)

def _lag_der(k, x):
    """Derivative of Laguerre polynomial L_k'(x)."""
    # L_k'(x) = -L_{k-1}^{(1)}(x) for k >= 1
    if k == 0:
        return 0.0
    return -eval_genlaguerre(k - 1, 1, x)

def _F(r, roots):
    """
    Evaluate the Laguerre combination F(r) = sum_i prod_{j != i} (r - r_j)^2 * L_{m-1}(r_i) / (r_i - r)^something?
    
    Actually, for the Cohn-Goncalves construction with double roots,
    the relevant function is built from the Laguerre polynomials.
    
    For m=12, n=1, the refined inequality involves a polynomial P(x) with 12 double roots.
    The sign uncertainty bound C4 depends on the product of roots and the Laguerre coefficients.
    
    Let me think more carefully. The task says "12 double roots with m=12 and dimension n=1".
    The score is the upper bound on C4, and we want to minimize it.
    
    From the knowledge: "Increasing the scale of the roots from 1000 to 10000 increases the product 
    of the roots by a factor of 10^12, which decreases the bound (inversely proportional to 
    sqrt(product)) by 10^6"
    
    But the measured result showed 33.31 -> 33.31, meaning the scale change didn't help.
    This suggests the bound is NOT simply inversely proportional to sqrt(product).
    
    Let me reconsider. The C4 bound for the Fourier sign uncertainty in 1D with m=12...
    
    Actually, looking at this more carefully: the "score" is the raw value of the upper bound on C4.
    The seed program returns [4.0*(i+1) for i in range(12)] = [4, 8, 12, ..., 48].
    
    The evaluator computes some bound C4 from these roots and returns it as the score.
    Lower score is better. The seed gets score 1.0, which means the seed's C4 value IS the reference.
    
    Wait, re-reading: "score is the upper bound on C4" and "the evaluator measures the raw value (lower is better).
    The search score is seed/value, so 1.0 equals the seed program."
    
    So if seed returns roots giving C4 = V_seed, and my program returns roots giving C4 = V_mine,
    then score = V_seed / V_mine. If V_mine < V_seed, score > 1.
    
    So I need to find roots that give a SMALLER C4 bound.
    
    From the knowledge about scale: scaling roots by factor s scales the product by s^12.
    If C4 ~ 1/sqrt(product) = 1/(s^6 * product_original), then scaling up would decrease C4.
    But the measurement showed no change, so this heuristic is wrong.
    
    Let me think about what C4 actually is. In the Cohn-Goncalves framework for uncertainty,
    for a function with m zeros (counting multiplicity), the sign uncertainty is bounded.
    
    For n=1 (1D), m=12 double roots means 24 total zeros. The Laguerre polynomial construction
    gives a specific bound.
    
    Given the complexity, let me try a different approach: use optimization to minimize
    the C4 bound directly. But I don't know the exact formula for C4 from the roots.
    
    Alternative: try various root configurations and see which gives lower C4.
    Since I can't compute C4 directly without knowing the exact formula, let me try:
    
    1. Evenly spaced roots (current seed)
    2. Roots clustered near origin
    3. Roots following Laguerre polynomial zeros
    4. Roots with specific spacing patterns
    
    Actually, wait. Let me re-read the task: "The notebook supplies 12 double roots with m=12 
    and dimension n=1, verifies the root constraints, and computes the last sign change."
    
    "verifies the root constraints" - there are constraints on the roots!
    "computes the last sign change" - this is related to the sign uncertainty.
    
    The root constraints likely come from the Laguerre polynomial construction.
    For the Cohn-Goncalves inequality, the roots must satisfy certain conditions related
    to the Laguerre polynomials.
    
    Given m=12 and n=1, the relevant Laguerre polynomial is L_12(x).
    The zeros of L_12(x) are positive and distinct. Maybe the double roots should be
    placed at or near the zeros of L_12(x)?
    
    Let me try: use the zeros of L_12(x) as the 12 double-root locations.
    """
    pass

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    """
    t_start = time.time()
    deadline = t_start + 600.0  # 600 second budget
    
    # Strategy: Try the zeros of the Laguerre polynomial L_12(x)
    # These are natural candidates for the double-root locations in the CG construction.
    
    # Find zeros of L_12(x)
    # L_n(x) = sum_{k=0}^{n} (-1)^k * C(n+k, k) * x^k / k!
    # Use numpy to find roots
    
    # Coefficients of L_12(x)
    n = 12
    coeffs = np.zeros(n + 1)
    for k in range(n + 1):
        # coefficient of x^k in L_n(x)
        from math import comb
        coeffs[k] = (-1)**k * comb(n + k, k) / float(k)
    
    # Find roots
    roots_lag = np.roots(coeffs)
    # Filter for real, positive roots
    real_roots = [r.real for r in roots_lag if abs(r.imag) < 1e-10 and r.real > 0]
    real_roots.sort()
    
    if len(real_roots) == 12:
        candidate1 = real_roots
    else:
        # Fallback: use approximate known zeros or evenly spaced
        candidate1 = [4.0 * (i + 1) for i in range(12)]
    
    # Also try scaled versions and other configurations
    # Try roots at small values (clustered near origin)
    candidate2 = [0.5 * (i + 1) for i in range(12)]
    
    # Try roots following 1/i pattern
    candidate3 = [12.0 / (i + 1) for i in range(12)]
    
    # Try geometric progression
    candidate4 = [1.0 * (1.5 ** i) for i in range(12)]
    
    # Try the seed
    candidate5 = [4.0 * (i + 1) for i in range(12)]
    
    # Since I can't compute C4 directly, I'll return the Laguerre zeros as the best guess
    # based on the mathematical structure of the problem.
    # The CG construction is specifically tied to Laguerre polynomials, so their zeros
    # are the most natural choice.
    
    return candidate1

# EVOLVE-BLOCK-END
