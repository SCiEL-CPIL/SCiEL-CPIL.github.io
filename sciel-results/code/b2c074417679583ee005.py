import numpy as np
import hashlib

def priority(vector, n):
    # Count non-zero elements (Hamming Weight)
    weight = sum(1 for x in vector if x != 0)
    
    # Use a hash to break ties randomly/deterministically
    v_str = str(vector)
    h = int(hashlib.md5(v_str.encode('utf-8')).hexdigest(), 16)
    hash_float = float(h) / float(0xFFFFFFFFFFFFFFFF)
    
    # Prioritize higher Hamming Weight
    # Scale weight so it dominates the hash tie-breaker
    return float(weight) * 1000.0 + hash_float
