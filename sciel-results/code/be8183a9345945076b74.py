"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 105  # safety margin
    
    # Represent weight-2 vectors as (i, j, si, sj)
    # Weight-3 vectors as (i, j, k, si, sj, sk)
    
    # For efficiency, use bitmasks or structured representation
    # Key insight: for weight-2 vectors with entries in {-1,1}, 
    # dot product between two such vectors depends on shared coordinates and signs.
    
    # Let's use a greedy approach with efficient dot product checking.
    # Store selected vectors as arrays for dot product computation.
    
    selected = []
    
    # Generate candidate vectors: mix of weight 2 and weight 3
    # To maximize count, prioritize weight-2 (more of them, smaller norm)
    
    # All weight-2 vectors: C(11,2) * 4 = 220
    # All weight-3 vectors: C(11,3) * 8 = 16*8 = 128... wait C(11,3)=165, so 165*8=1320
    
    # Constraint: for any two selected vectors a, b:
    #   ||a||^2 + ||b||^2 - 2*<a,b> >= 3  (since max norm^2 = 3)
    
    # Greedy: try to add vectors, check against all selected.
    # Use numpy for batch dot products when possible.
    
    # Build candidate list: weight-2 first, then weight-3
    candidates = []
    
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
    
    # Greedy selection
    for v in candidates:
        if time.time() > deadline:
            break
        v_norm2 = int(np.dot(v, v))
        
        # Check against all selected
        compatible = True
        for u in selected:
            u_norm2 = int(np.dot(u, u))
            dist2 = v_norm2 + u_norm2 - 2 * int(np.dot(v, u))
            if dist2 < 3:
                compatible = False
                break
        
        if compatible:
            selected.append(v)
    
    if not selected:
        # Fallback
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return np.array([v], dtype=np.int64)
    
    return np.array(selected, dtype=np.int64)
# EVOLVE-BLOCK-END
