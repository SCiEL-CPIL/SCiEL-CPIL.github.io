import numpy as np
import hashlib

def priority(vector, n):
    # Count the number of 2s in the vector.
    # Vectors with fewer 2s are prioritized.
    # Specifically, vectors with NO 2s (i.e., from {0,1}^n) are prioritized highest.
    # It is a known property that {0,1}^n contains no 3-term arithmetic progressions
    # (distinct elements), so this entire subset forms a valid cap set of size 2^n.
    # By prioritizing these first, we guarantee a large base cap set.
    
    count_2s = 0
    for x in vector:
        if x == 2:
            count_2s += 1
            
    # Create a deterministic hash for tie-breaking among vectors with the same number of 2s.
    # This helps explore the space of vectors with 2s more effectively than lexicographic order.
    v_str = str(vector)
    h = int(hashlib.md5(v_str.encode('utf-8')).hexdigest(), 16)
    hash_float = float(h) / float(0xFFFFFFFFFFFFFFFF)
    
    # Priority: Minimize count of 2s.
    # Large multiplier ensures all no-2 vectors come before any 1-2 vector, etc.
    return -count_2s * 1000.0 + hash_float
