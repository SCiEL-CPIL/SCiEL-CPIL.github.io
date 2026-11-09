"""Refined Cohn-Goncalves Laguerre-polynomial construction for Fourier sign uncertainty."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre
import time

def _compute_bound(roots):
    """Compute the upper bound on C4 for given root locations.
    
    For the Cohn-Goncalves construction with m=12 double roots,
    the bound is related to 1/sqrt(product of roots).
    """
    prod = np.prod(roots)
    if prod <= 0:
        return float('inf')
    return 1.0 / np.sqrt(prod)

def construct():
    """Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    
    Returns list of 12 positive double-root locations.
    Score is the upper bound on C4 (lower is better).
    """
    deadline = time.time() + 590.0  # Safety margin before 600s limit
    
    # Start with Laguerre-Gauss nodes (zeros of L_12)
    m = 12
    n = 1  # dimension
    
    # Get initial roots: zeros of the Laguerre polynomial L_12
    # These are the Gauss-Laguerre quadrature nodes
    roots, _ = roots_genlaguerre(m, alpha=0)
    roots = np.sort(roots)
    
    # Verify roots are positive
    if np.any(roots <= 0):
        # Fallback: use a simple geometric sequence
        roots = np.logspace(0, 2, m)
        return list(roots)
    
    # Try to improve by optimizing the product of roots
    # while maintaining the constraint that roots come from a valid
    # Laguerre polynomial combination
    
    # The key insight: we want to maximize the product of the 12 roots
    # The Laguerre polynomial L_m(x) has m positive zeros
    # But we can consider generalized Laguerre polynomials L_m^(alpha)(x)
    
    # Try different alpha values for generalized Laguerre polynomials
    best_roots = roots.copy()
    best_prod = np.prod(roots)
    
    # Search over alpha parameter of generalized Laguerre polynomials
    # L_m^(alpha)(x) = sum_{k=0}^{m} (-1)^k * C(m+alpha, m-k) * x^k / k!
    # For alpha > -1, all zeros are positive
    
    # First, try a range of alpha values
    for alpha in np.linspace(0, 50, 100):
        if time.time() > deadline:
            break
        try:
            r, _ = roots_genlaguerre(m, alpha=alpha)
            r = np.sort(r)
            if np.all(r > 0):
                prod = np.prod(r)
                if prod > best_prod:
                    best_prod = prod
                    best_roots = r.copy()
        except:
            continue
    
    # Now do a local optimization using the fact that we can scale
    # But wait - the roots of L_m^(alpha) are fixed for each alpha.
    # We can't freely scale them.
    
    # Alternative approach: use the constraint structure more carefully.
    # The Cohn-Goncalves construction requires that the roots are zeros
    # of a specific polynomial related to Laguerre polynomials.
    
    # Let's try: the polynomial is sum_{k=0}^{m} c_k * L_k(x) where
    # the coefficients are chosen so that the polynomial has m double roots.
    # For m=12, we need a degree-24 polynomial with 12 double roots.
    
    # Actually, let's reconsider. The standard construction uses:
    # P(x) = sum_{k=0}^{m} (-1)^k * C(m, k) * L_k(x)
    # This is related to the m-th order Laguerre polynomial structure.
    
    # Let me compute the product for several candidate constructions:
    
    candidates = []
    
    # Candidate 1: Zeros of L_12(x)
    candidates.append(('L12', roots.copy()))
    
    # Candidate 2: Zeros of L_12^(alpha)(x) for optimal alpha
    for alpha in [0, 1, 5, 10, 20, 50, 100]:
        try:
            r, _ = roots_genlaguerre(m, alpha=alpha)
            r = np.sort(r)
            if np.all(r > 0):
                candidates.append((f'L12^{alpha}', r.copy()))
        except:
            pass
    
    # Candidate 3: Try to use the structure where roots are placed
    # to maximize the product subject to the polynomial constraint.
    # The constraint is that the polynomial Q(x) = prod_{i=1}^{12} (x - r_i)^2
    # must satisfy certain moment conditions related to the Laguerre basis.
    
    # For now, let's also try a different structural approach:
    # Use the fact that for the Fourier sign uncertainty, the optimal
    # configuration might correspond to specific known bounds.
    
    # Let's evaluate which candidate gives the best (lowest) bound
    best_candidate = candidates[0][1]
    best_bound = _compute_bound(candidates[0][1])
    
    for name, r in candidates:
        b = _compute_bound(r)
        if b < best_bound:
            best_bound = b
            best_candidate = r
    
    # Try one more structural idea: scale the roots
    # If the constraint allows scaling, larger roots give better bounds.
    # But the Laguerre polynomial zeros are fixed in scale.
    # However, if we use L_m^(alpha) with large alpha, the roots spread out.
    
    # Let's try very large alpha
    for alpha in [50, 100, 500, 1000, 5000, 10000]:
        if time.time() > deadline:
            break
        try:
            r, _ = roots_genlaguerre(m, alpha=alpha)
            r = np.sort(r)
            if np.all(r > 0):
                prod = np.prod(r)
                if prod > np.prod(best_candidate):
                    best_candidate = r.copy()
        except:
            continue
    
    # Final check: ensure all roots are positive
    if np.any(best_candidate <= 0):
        best_candidate = np.logspace(0, 3, m)
    
    return list(best_candidate)
# EVOLVE-BLOCK-END
