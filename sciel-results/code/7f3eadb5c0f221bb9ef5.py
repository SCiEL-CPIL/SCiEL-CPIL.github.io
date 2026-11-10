# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points maximizing k
    subject to: min pairwise squared distance >= max squared norm.
    
    Key insight: All 220 norm-2 vectors (vectors with exactly two ±1 entries)
    are mutually compatible. For any two distinct such vectors, the dot product
    is at most 1, so dist^2 = 4 - 2*dot >= 2 = max_norm^2.
    
    We can also try to add norm-4 vectors (single coordinate ±2) if they are
    compatible with the norm-2 set, but the 220 norm-2 vectors alone give a
    strong baseline.
    """
    start_time = time.time()
    DEADLINE = 600.0
    
    d = 11
    
    # Generate all 220 norm-2 vectors
    candidates = []
    for i in range(d):
        for j in range(i+1, d):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # That's C(11,2) * 4 = 55 * 4 = 220 vectors
    n2 = len(candidates)
    assert n2 == 220, f"Expected 220 norm-2 vectors, got {n2}"
    
    points_arr = np.array(candidates, dtype=np.int64)
    
    # Verify all pairs are compatible
    # For norm-2 vectors: dist^2 = 4 - 2*dot, need dist^2 >= 2, so dot <= 1
    # Max dot between distinct norm-2 vectors is 1 (share one coordinate with same sign)
    # So all 220 are mutually compatible
    
    # Try to add norm-4 vectors (single coordinate ±2)
    # For a norm-4 vector u and norm-2 vector v:
    # dist^2 = 4 + 2 - 2*dot(u,v) = 6 - 2*dot(u,v)
    # Need dist^2 >= 4 (since max_norm^2 would be 4 if we include norm-4 vectors)
    # So 6 - 2*dot >= 4 => dot <= 1
    # 
    # A norm-4 vector has one coordinate ±2. A norm-2 vector has two coordinates ±1.
    # dot(u,v) can be: -2, -1, 0, 1, 2
    # If dot = 2: dist^2 = 6-4 = 2 < 4, INCOMPATIBLE
    # If dot = 1: dist^2 = 6-2 = 4 >= 4, OK
    # If dot = 0: dist^2 = 6 >= 4, OK
    # etc.
    #
    # dot = 2 when u[i] = 2 and v[i] = 1, or u[i] = -2 and v[i] = -1
    # So a norm-4 vector u = 2*e_i is incompatible with norm-2 vectors that have
    # v[i] = sign(u[i]) = 1 if u[i]=2, or v[i]=-1 if u[i]=-2.
    #
    # There are C(10,1)*2 = 20 norm-2 vectors with v[i] = 1 (for fixed i)
    # and 20 with v[i] = -1.
    # So u = 2*e_i conflicts with 20 norm-2 vectors (those with v[i]=1).
    # u = -2*e_i conflicts with 20 norm-2 vectors (those with v[i]=-1).
    #
    # If we add both 2*e_i and -2*e_i, they conflict with 40 norm-2 vectors.
    # But we could choose to NOT include those conflicting norm-2 vectors.
    #
    # Strategy: For each coordinate i, we can add up to 2 norm-4 vectors (±2*e_i).
    # Adding 2*e_i removes 20 norm-2 vectors. Net gain: +1 - 20 = -19. Bad.
    # So it's not worth adding norm-4 vectors if it costs 20 norm-2 vectors each.
    #
    # What if we add norm-4 vectors and remove the conflicting norm-2 vectors?
    # Start with 220. Add one norm-4 vector: remove 20, add 1. Net: 220 - 20 + 1 = 201. Worse.
    #
    # So the optimal is just the 220 norm-2 vectors.
    
    result = points_arr.copy()
    result = np.around(result).astype(np.int64)
    
    # Final validation
    k = result.shape[0]
    if k < 2:
        result = np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
