import numpy as np
import hashlib

def priority(vector, n):
    # Convert to numpy array for fast operations
    v = np.array(vector, dtype=np.int8)
    
    # Heuristic: Prioritize vectors with balanced counts of 0, 1, and 2.
    # This corresponds to vectors that are "uniform" in their symbol distribution,
    # which may reduce the likelihood of forming arithmetic progressions 
    # (x+y+z=0 mod 3) with other similarly balanced vectors.
    
    # Count occurrences of 0, 1, 2
    c0 = np.count_nonzero(v == 0)
    c1 = np.count_nonzero(v == 1)
    c2 = np.count_nonzero(v == 2)
    
    # Target is n/3 for each
    target = n / 3.0
    
    # Calculate deviation from the balanced distribution
    dist = abs(c0 - target) + abs(c1 - target) + abs(c2 - target)
    
    # Base priority is negative distance (higher is better for balanced vectors)
    base_priority = -dist
    
    # Tie-breaking: Use a stable hash to ensure deterministic but non-lexicographic order
    # among vectors with the same balance distance.
    v_bytes = bytes(v)
    h = int(hashlib.md5(v_bytes).hexdigest(), 16)
    tie_break = (h % 10000) / 10000.0
    
    return base_priority + tie_break * 0.01
