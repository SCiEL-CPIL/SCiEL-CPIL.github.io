"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

_T0 = time.time()
_DEADLINE = _T0 + 285.0  # 45s safety margin before 330s kill

def priority(vector, n):
    """
    Priority for a vector in {0,1,2}^n.
    
    Strategy:
    1. Balance: Prefer vectors with counts of 0, 1, 2 close to n/3.
    2. Quadratic edge-distance: Penalize vectors with many 0s or 2s (corners).
    3. Deterministic position-dependent hash tie-break.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        return -1e9

    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
            
    # Target is n/3
    target = n / 3.0
    
    # Quadratic deviation from balance (L2 norm squared)
    quad_dev = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    
    # We want higher priority for lower deviation
    score = -quad_dev
    
    # Position-dependent hash for better symmetry breaking
    # Use a weighted sum with position-dependent coefficients
    h = 0
    for i, v in enumerate(vector):
        h = (h * 37 + v * (i + 1)) & 0xFFFFFFFF
    score += (h / 4294967295.0) * 1e-6
    
    return score

# EVOLVE-BLOCK-END
