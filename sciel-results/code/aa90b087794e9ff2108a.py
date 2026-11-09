"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.special import eval_laguerre, factorial
from scipy.optimize import minimize, differential_evolution

# EVOLVE-BLOCK-START
def _laguerre_combination(x, roots, m=12, n=1):
    """
    Compute the Laguerre polynomial combination L_{m,n}(x; roots).
    For the Cohn-Goncalves construction with m double roots at r_i,
    the polynomial is proportional to prod_i (x - r_i)^2 * L_{m}(x)
    but the standard form is:
    P(x) = sum_{k=0}^{m} c_k L_k(x) where the coefficients are chosen
    so that P has double roots at r_i.
    
    Actually, the refined construction uses:
    P(x) = L_m^{(n-1)}(x) * prod_i (x - r_i)^2  ... no.
    
    Let me reconsider. The Cohn-Goncalves theorem states that for a polynomial
    P of degree m with m distinct real roots, if P is a linear combination of
    Laguerre polynomials L_0, ..., L_m, then certain sign uncertainty bounds hold.
    
    The 'double root' version: we want P(x) = prod_{i=1}^{m} (x - r_i)^2? No, that's degree 2m.
    
    Re-reading the task: "12 double roots with m=12 and dimension n=1". 
    This likely means a polynomial of degree m=12 that has 12 real roots (each with multiplicity 2? 
    No, that would be degree 24). 
    
    More likely: the construction involves a polynomial P of degree m where the roots 
    r_1, ..., r_m are the 'double roots' in the sense that they are roots of a related 
    polynomial, or the bound is derived from a polynomial with double roots.
    
    Actually, in the context of uncertainty principles, the Cohn-Goncalves inequality 
    often involves a polynomial P(x) = sum a_k L_k(x) with P having m real roots. 
    The 'double root' construction might refer to the fact that the optimal polynomial 
    for the bound has roots that come in pairs or the bound is sharpened by considering 
    a polynomial with double roots at specific locations.
    
    Let's assume the standard form: P(x) = sum_{k=0}^{m} c_k L_k(x). 
    The condition that P has double roots at r_1, ..., r_m means:
    P(r_i) = 0 and P'(r_i) = 0 for i=1,...,m.
    This gives 2m equations for m+1 coefficients (c_0, ..., c_m), so the system is 
    overdetermined. However, we can scale, so we have m degrees of freedom for the 
    roots, and the coefficients are determined up to scale.
    
    Wait, if P is a linear combination of L_0, ..., L_m, it has m+1 coefficients. 
    Requiring 2m conditions (m roots, each double) is too many. 
    So perhaps 'double roots' means something else.
    
    Alternative interpretation: The polynomial is of degree 2m, with m double roots. 
    P(x) = prod_{i=1}^{m} (x - r_i)^2. This polynomial has degree 2m. 
    But the task says m=12. If the polynomial degree is 2m=24, then the Laguerre 
    combination would be of degree 24? But the task says 'Laguerre combination (m=12, n=1)'.
    
    Let's look at the seed program: it returns 12 values [4, 8, ..., 48]. 
    The score is the 'upper bound on C4'. The seed score is 1.0.
    
    Given the ambiguity, I will implement a generic optimization:
    1. Define a function that computes the uncertainty bound C4 given the roots.
    2. Optimize the roots to minimize C4.
    
    Since I don't have the exact formula for C4, I will make an educated guess based on 
    common forms in uncertainty principles. A common form is:
    C4 = (some norm of P) * (some norm of P') / (some normalization)
    
    However, without the exact formula, I cannot compute the score. 
    The task says 'score is the upper bound on C4'. The evaluator computes this.
    My job is to return the roots. The evaluator will compute the score from the roots.
    
    So I just need to find roots that minimize the score. Since I don't know the score 
    function, I cannot optimize it directly. 
    
    BUT, the task says 'the notebook supplies... verifies the root constraints, and computes 
    the last sign change.' This implies there is a known construction.
    
    Let's assume the roots should be the roots of a specific orthogonal polynomial or 
    a known optimal configuration. For Laguerre polynomials, the roots of L_m(x) are 
    known. Perhaps the optimal 'double roots' are related to the roots of L_m(x) or 
    L_{m-1}(x).
    
    The roots of L_m(x) are positive. For m=12, they are approximately:
    0.54, 1.80, 3.84, 6.66, 10.25, 14.61, 19.73, 25.61, 32.25, 39.65, 47.81, 56.73 (approx).
    
    The seed program uses [4, 8, ..., 48], which are evenly spaced.
    
    Let's try using the roots of the Laguerre polynomial L_12(x) as the 'double roots'.
    """
    # This function is a placeholder for the actual computation if needed.
    pass

def _compute_score(roots, m=12, n=1):
    """
    Placeholder for score computation. 
    In reality, the evaluator computes this. 
    For optimization, we need a proxy. 
    If we cannot compute the score, we cannot optimize.
    
    However, the task implies that the roots should satisfy certain constraints. 
    Perhaps the best strategy is to use the roots of the Laguerre polynomial L_m(x).
    """
    # Since we don't have the score function, we'll assume the optimal roots 
    # are the roots of L_m(x) or a similar orthogonal polynomial.
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    m = 12
    n = 1
    
    # Strategy 1: Use roots of Laguerre polynomial L_m(x)
    # The roots of L_m(x) are the zeros of the Laguerre polynomial.
    try:
        # Use numpy's polynomial roots or scipy's roots
        # L_m(x) = sum_{k=0}^m (-1)^k binom(m, k) x^k / k!
        # Coefficients for L_m(x):
        coeffs = []
        for k in range(m, -1, -1):
            c = (-1)**k * math.comb(m, k) / math.factorial(k)
            coeffs.append(c)
        # Find roots
        roots_lag = np.roots(coeffs)
        # Filter real positive roots
        real_roots = [r.real for r in roots_lag if abs(r.imag) < 1e-10 and r.real > 0]
        real_roots.sort()
        if len(real_roots) == m:
            # Check if these are good
            # The seed uses [4, 8, ..., 48]. Laguerre roots are clustered near 0.
            # Maybe the 'double roots' are scaled?
            # Let's try scaling the Laguerre roots to match the scale of the seed.
            # Seed mean: 26, Laguerre mean: ~29 (for m=12). Similar scale.
            # Let's return the Laguerre roots as a candidate.
            candidate1 = real_roots
        else:
            candidate1 = [4.0 * (i + 1) for i in range(m)]
    except Exception:
        candidate1 = [4.0 * (i + 1) for i in range(m)]
    
    # Strategy 2: Use roots of the derivative of L_m(x)? Or L_{m-1}(x)?
    # Let's try a few candidates and return the one that seems most 'optimal'.
    # Without a score function, we have to guess.
    
    # The task mentions 'refined Cohn-Goncalves Laguerre-polynomial construction'.
    # A common refinement is to use the roots of the Laguerre polynomial itself.
    
    # Let's also try the roots of the associated Laguerre polynomial L_m^{(n-1)}(x).
    # For n=1, this is L_m^{(0)}(x) = L_m(x).
    
    # Given the lack of a score function, I will return the roots of L_m(x).
    # If that doesn't work, the evaluator will give a score, and future iterations
    # can use that to guide optimization.
    
    # For now, return the Laguerre roots.
    return candidate1
# EVOLVE-BLOCK-END
