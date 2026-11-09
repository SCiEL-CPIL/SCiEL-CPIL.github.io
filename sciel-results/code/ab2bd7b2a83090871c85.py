"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    # Compute zeros of the generalized Laguerre polynomial L_12^(1)
    # These are the natural double-root locations for the m=12, n=1 construction
    roots = roots_genlaguerre(12, 1)[0]
    
    # The roots from roots_genlaguerre are positive and sorted
    # Scale to optimize the upper bound on C4
    # The bound is typically related to the spread of roots
    # Try scaling factor to minimize the largest root while maintaining validity
    
    # The seed used [4, 8, 12, ..., 48] with max = 48
    # Laguerre L_12^(1) zeros range from ~0.5 to ~35
    # Let's use them directly first, then consider scaling
    
    # For the Cohn-Goncalves construction, the optimal scaling
    # minimizes the extremal quantity. A common choice is to
    # scale so that the geometric mean or a specific moment is fixed.
    
    # Let's try: scale so that the sum of roots matches a target,
    # or simply use the natural zeros which are already optimized
    
    # The natural zeros of L_m^(n) are known to be optimal for
    # certain extremal problems. Use them directly.
    result = list(roots)
    
    # Ensure all are positive (they should be for Laguerre)
    result = [float(r) for r in result if r > 0]
    
    # Pad if needed (shouldn't be)
    while len(result) < 12:
        result.append(result[-1] * 1.01 if result else 1.0)
    
    return result[:12]
# EVOLVE-BLOCK-END
