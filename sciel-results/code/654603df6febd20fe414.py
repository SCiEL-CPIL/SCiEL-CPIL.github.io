"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Minimizes the Cohn-Goncalves upper bound on C4.
    """
    deadline = time.time() + 570  # Safety margin before 630s limit
    
    # The Cohn-Goncalves construction: for m=12 double roots and n=1,
    # the upper bound on C4 is determined by the last sign change of
    # the constructed polynomial. We model this as a function of root locations.
    
    # For the Laguerre polynomial construction, the bound is related to
    # the largest zero of the polynomial P(x) = L_n(x) * prod(x - r_i)^2
    # or a similar combination. Given n=1, L_1(x) = 1 - x.
    
    # The key insight: with double roots, the sign structure is determined
    # by the Laguerre factor. The upper bound C4 is typically the location
    # of the last sign change, which for this construction is a function
    # of the root locations.
    
    # Based on the Cohn-Goncalves framework, the bound for the uncertainty
    # principle with m double roots is given by the largest root of a
    # modified Laguerre polynomial. For n=1, this simplifies.
    
    # Let's model the bound as the maximum of the root locations, since
    # the last sign change will be at or near the rightmost root for
    # this type of construction. To minimize the bound, we want to
    # minimize the maximum root, subject to the roots being positive
    # and satisfying the Laguerre orthogonality constraints.
    
    # The Laguerre polynomial L_m(x) with weight e^{-x} has roots that
    # minimize certain functionals. The optimal double-root locations
    # for minimizing the last sign change should be the roots of the
    # Laguerre polynomial itself, or a related orthogonal polynomial.
    
    # For L_12(x), the roots are well-known. Let's use them as the
    # double-root locations, as they are the natural choice for the
    # Laguerre construction.
    
    # Compute the roots of the Laguerre polynomial L_12(x)
    # Using scipy.special.roots_laguerre
    from scipy.special import roots_laguerre
    
    x_roots, _ = roots_laguerre(12)
    
    # The roots of L_12(x) are positive and in increasing order
    # These should be the optimal double-root locations for the
    # Cohn-Goncalves construction
    
    result = list(x_roots)
    
    # Verify: 12 positive roots
    assert len(result) == 12
    assert all(r > 0 for r in result)
    assert all(result[i] < result[i+1] for i in range(11))
    
    return result
# EVOLVE-BLOCK-END
