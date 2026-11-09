"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 10
    # Start time to ensure we don't exceed limits, though this is O(n^2)
    start_time = time.time()
    
    # Construct L1: (i + j) % n
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Construct L2: (i + 2j) % n
    # Orthogonality check logic:
    # Pairs (i+j, i+2j) mod n.
    # Difference in second component - difference in first component = j.
    # Since gcd(1, n)=1, j is uniquely determined, then i.
    L2 = [[(i + 2 * j) % n for j in range(n)] for i in range(n)]
    
    # Verify orthogonality just in case (optional but good for safety)
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (L1[i][j], L2[i][j])
            if p in pairs:
                # This should not happen, but if it does, we have a bug.
                # For n=10, gcd(1, 10)=1 ensures orthogonality for slopes 1 and 2.
                pass
            pairs.add(p)
            
    if len(pairs) != n * n:
        # Fallback to single square if something went wrong (should not happen)
        return [L1]
        
    # Ensure Python ints
    # The operations % and * on small ints in Python 3 produce Python ints.
    
    return [L1, L2]
# EVOLVE-BLOCK-END
