import time
import math

_T0 = time.time()
_DEADLINE = _T0 + 280.0

def priority(vector, n):
    """
    Priority for a vector in {0,1,2}^n for greedy cap-set construction.
    
    Strategy: Corner-First Routing (Inverse Balanced).
    The current program (Balanced) prioritizes vectors with many 1s (center).
    This new program prioritizes vectors with FEW 1s (corners).
    
    Rationale:
    The subspace {0,2}^n is a cap set of size 2^n. 
    By prioritizing vectors with 0 ones (i.e., vectors in {0,2}^n) first, 
    we ensure that this large, structurally safe subspace is fully populated 
    before attempting to add vectors with 1s. 
    Vectors with 1s are more "dangerous" as they can form APs with many 
    other vectors. By filling the safe corners first, we might allow more 
    "safe" vectors with 1s to be added later, as the greedy algorithm 
    will skip those that conflict with the already-placed corners. 
    This structural change moves the exploration from the high-density 
    center to the high-safety corners.
    """
    if time.time() > _DEADLINE:
        return 0.0

    # Count 1s
    count_1 = 0
    for v in vector:
        if v == 1:
            count_1 += 1
    
    # We want vectors with FEWER 1s to have HIGHER priority.
    # So, priority is inversely related to count_1.
    # base = n - count_1
    # If count_1 = 0, base = n (Max)
    # If count_1 = n, base = 0 (Min)
    
    base = n - count_1

    # Compute index for hash tie-breaking
    idx = 0
    for v in reversed(vector):
        idx = idx * 3 + v

    def mix64(h):
        h ^= h >> 33
        h = (h * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 33
        h = (h * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 33
        return h

    seed_val = idx ^ (n * 0x9E3779B97F4A7C15)
    mixed = mix64(seed_val)
    
    # Add a small perturbation to break ties deterministically
    perturb = (mixed / (2**64)) * 0.5
    
    return float(base + perturb)
