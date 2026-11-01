"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # We aim for 512 vectors using the weight-4 construction.
    # We need 32 supports of size 4, pairwise intersecting in at most 2 elements.
    
    # Generate all combinations of 4 indices
    all_supports = list(itertools.combinations(range(D), 4))
    
    selected_supports = []
    
    # Greedy selection
    for s in all_supports:
        # Check if s intersects all selected supports in <= 2 elements
        valid = True
        for t in selected_supports:
            intersection = set(s).intersection(t)
            if len(intersection) > 2:
                valid = False
                break
        if valid:
            selected_supports.append(s)
            if len(selected_supports) >= 32:
                break
        
    # If we didn't find 32 (unlikely), we just use what we have.
    # But 32 is easily achievable.
    
    vectors = []
    for s in selected_supports:
        # Generate all 16 sign variations for this support
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            vectors.append(v)
            
    if not vectors:
        # Fallback: return some valid small set, e.g., basis vectors scaled
        # But the greedy above should always produce at least 1 support.
        # If 0 supports, return empty? No, must be non-zero vectors.
        # Let's add a simple fallback: 220 weight-2 vectors from D_11
        # Actually, just return the first few weight-4 vectors if greedy failed (it won't)
        # Or just return the 220 D_11 minimal vectors.
        # Let's stick to the weight-4 logic, it's robust.
        pass
    
    result = np.array(vectors, dtype=np.int64)
    
    # Optional: Verify validity and count
    # k = result.shape[0]
    # if k == 0: return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
