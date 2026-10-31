import numpy as np
import hashlib

def priority(vector, n):
    # Calculate Hamming weight (number of non-zero elements)
    # We prioritize vectors with HIGHER Hamming weight.
    # Hypothesis: High-weight vectors are less likely to form lines with each other
    # in the early stages of the greedy process compared to low-weight vectors,
    # which tend to cluster around the origin and form lines like e_i, e_j, e_i+e_j.
    weight = sum(1 for x in vector if x != 0)
    
    # Use a hash for tie-breaking to ensure a deterministic but non-lexicographic order
    # within each weight class.
    v_str = str(vector)
    h = int(hashlib.md5(v_str.encode('utf-8')).hexdigest(), 16)
    
    # Combine weight and hash.
    # We want higher weight to have higher priority.
    # We scale the hash so it doesn't dominate the weight difference.
    # Max weight is n. Max hash is ~2^128.
    # We can use: weight * LARGE_NUMBER + normalized_hash
    # Or simply: weight + normalized_hash (if weight is integer and hash is in [0,1))
    # Since weight is an integer, adding a hash in [0,1) will preserve the weight ordering
    # while randomizing the order within the same weight.
    
    normalized_hash = float(h) / float(0xFFFFFFFFFFFFFFFF)
    
    # Return higher priority for higher weight
    return weight + normalized_hash

# EVOLVE-BLOCK-END
