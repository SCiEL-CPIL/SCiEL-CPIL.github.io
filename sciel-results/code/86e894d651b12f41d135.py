import hashlib
import time

# Cache for priorities to avoid recomputation if the same vector is queried multiple times
_priority_cache = {}

def priority(vector, n):
    """
    Returns a priority for the vector. Higher priority means the vector is 
    added earlier in the greedy process.
    
    Strategy: 
    1. Use a 64-bit slice of SHA256 for a high-quality random order.
    2. Add a small bias towards vectors with more zeros (lower weight).
    """
    # Deadline guard: We can't easily track global time here without state,
    # but we can make the function very fast.
    # If n is very large, the hash is still O(n).
    
    # Check cache
    if vector in _priority_cache:
        return _priority_cache[vector]
    
    # Create a string representation of the vector for hashing
    # Using bytes is faster than string
    # vector is a tuple of ints 0, 1, 2
    # We can pack them into bytes
    # Since n is small, we can just use the tuple directly in the hash
    # But hashlib requires bytes
    # We can convert the tuple to a bytes object
    # Each element is 0, 1, or 2. We can use 1 byte per element.
    v_bytes = bytes(vector)
    
    # Compute SHA256
    h = hashlib.sha256(v_bytes).digest()
    
    # Take the first 8 bytes for a 64-bit integer
    # Use big-endian
    hash_val = int.from_bytes(h[:8], byteorder='big')
    
    # Normalize to [0, 1)
    # 2**64 is approx 1.8e19
    random_part = hash_val / (2**64)
    
    # Compute weight: number of non-zero elements
    # This is a cheap operation
    weight = sum(1 for x in vector if x != 0)
    
    # Bias: We want to prioritize vectors with LOWER weight (more zeros).
    # So we subtract a small value proportional to weight.
    # Or add a value proportional to (n - weight).
    # Let's use a small epsilon.
    # The random part is in [0, 1).
    # We want the bias to be small enough not to disrupt the random order significantly,
    # but large enough to break ties in favor of sparse vectors.
    # Let's say the bias is at most 0.01.
    bias = 0.01 * (n - weight) / n
    
    # Final priority
    # We want higher priority for "better" vectors.
    # If random_part is high, it's good.
    # If weight is low (more zeros), bias is high, so it's good.
    p = random_part + bias
    
    # Cache the result
    _priority_cache[vector] = p
    
    return p
