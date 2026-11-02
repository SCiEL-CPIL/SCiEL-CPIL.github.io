"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def construct():
    """
    Construct a kissing configuration in dimension 11 using D11 lattice roots.
    
    The D11 lattice roots are vectors with exactly two non-zero entries of ±1.
    There are C(11,2) * 4 = 55 * 4 = 220 such roots.
    
    For two roots u and v:
    - If they share no coordinates: dist^2 = 4, norm^2 = 2, so dist >= norm ✓
    - If they share one coordinate: dist^2 = 2, norm^2 = 2, so dist >= norm ✓
    - If they share both coordinates: dist^2 = 0 or 4. If same signs, dist=0 ✗.
      If different signs (e.g., (1,1) vs (1,-1)), dist^2 = 4, so dist >= norm ✓
    
    So the only conflicts are between identical vectors. We can include at most
    one copy of each unique root. But wait - we can also include both (i,j,1,1)
    and (i,j,1,-1) since they differ. Let's think more carefully.
    
    Actually, all 220 roots have norm^2 = 2. The minimum pairwise distance
    constraint is dist >= norm = sqrt(2), so dist^2 >= 2.
    
    Two distinct roots u, v:
    - Same support (same two indices): 
      * Same signs: dist^2 = 0 → violates
      * One sign flipped: dist^2 = 4 → ok
      * Both signs flipped: dist^2 = 0 → violates (same vector)
    - Share exactly one index: dist^2 = 2 → ok
    - Disjoint support: dist^2 = 4 → ok
    
    So for each pair of indices {i,j}, we can include at most 2 roots:
    (i,j,+,+) and (i,j,+,-) are compatible (dist^2=4)
    (i,j,+,+) and (i,j,-,+) are compatible (dist^2=4)
    (i,j,+,+) and (i,j,-,-) are NOT compatible (dist^2=0)
    
    So for each {i,j}, we can pick at most 2 roots that are "orthogonal" in sign space.
    The maximum independent set in the conflict graph for each {i,j} is 2.
    
    Total: 55 * 2 = 110 vectors.
    """
    
    vectors = []
    
    for i, j in combinations(range(D), 2):
        # For each pair of indices, pick 2 compatible roots
        # Option 1: (i,j,1,1) and (i,j,1,-1) — dist^2 = 4 between them
        v1 = [0] * D
        v1[i], v1[j] = 1, 1
        vectors.append(v1)
        
        v2 = [0] * D
        v2[i], v2[j] = 1, -1
        vectors.append(v2)
    
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
