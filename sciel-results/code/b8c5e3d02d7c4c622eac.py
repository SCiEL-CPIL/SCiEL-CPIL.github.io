"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Assign priority to a vector in {0,1,2}^n for greedy cap set construction.
    
    Strategy: Prioritize vectors with fewer 2s, then fewer 1s.
    This favors vectors in the "lower" part of the space, which empirically
    tend to form larger cap sets when selected greedily.
    
    The priority is a float; higher priority means selected earlier.
    """
    # Count occurrences of each value
    count_0 = 0
    count_1 = 0
    count_2 = 0
    for v in vector:
        if v == 0:
            count_0 += 1
        elif v == 1:
            count_1 += 1
        elif v == 2:
            count_2 += 1
    
    # Higher priority for fewer 2s (heavily penalized), then fewer 1s
    # Use a large weight for 2s to dominate
    score = - (10 * count_2 + count_1)
    
    return float(score)
# EVOLVE-BLOCK-END
