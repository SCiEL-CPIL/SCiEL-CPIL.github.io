"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import itertools

# EVOLVE-BLOCK-START

def priority(vector, n):
    """
    Greedy cap set construction priority.
    
    The evaluator generates all vectors in {0,1,2}^n, sorts by priority descending,
    and greedily adds vectors that keep the set a cap set.
    
    We use a static heuristic:
    1. Primary: Maximize the number of coordinates that are not 0 (i.e., vectors with more 1s and 2s).
       Vectors with many 0s tend to form lines with each other more easily.
    2. Secondary: Among vectors with the same count of non-zero coords, prefer those
       with a more balanced distribution of 1s and 2s.
    3. Tertiary: A deterministic hash to break remaining ties.
    
    This approach favors vectors that are "spread out" in the space, which helps
    the greedy algorithm maintain more degrees of freedom.
    """
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
    
    non_zero = c1 + c2
    
    # Balance term: prefer vectors where c1 and c2 are close to each other
    # The difference |c1 - c2| should be small
    balance_penalty = abs(c1 - c2)
    
    # Combine: non_zero is the primary key (higher is better)
    # balance_penalty is secondary (lower is better, so subtract)
    # Use a hash for tertiary tie-breaking
    
    val = 0
    for x in vector:
        val = val * 3 + x
    
    # Simple deterministic hash
    h = val
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    
    # Priority: non_zero dominates, then -balance_penalty, then small hash contribution
    # Scale so that non_zero differences dominate everything else
    # non_zero ranges from 0 to n
    # balance_penalty ranges from 0 to n
    # hash contribution is tiny, just for tie-breaking
    
    return non_zero * 1.0 - balance_penalty * 0.1 + (h / 4294967296.0) * 0.001

# EVOLVE-BLOCK-END
