import time
import zlib

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin

# Precompute nothing, as n is not fixed. 
# However, we can use a fixed seed for any random components if we used them.
# Here we use deterministic hashes.

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy: Chaotic, non-local ordering via double polynomial hashing.
    This decouples the greedy selection from lattice symmetries and local density traps.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Fast polynomial rolling hashes
    # Using two different bases to reduce correlation
    h1 = 2166136261  # FNV offset basis
    h2 = 0x811C9DC5  # Another FNV variant or just a large prime
    
    # We want the priority to be a float.
    # We will compute a hash that looks random.
    
    # Base 31 and Base 131 are common for strings, let's use them for integers.
    # To avoid overflow issues in Python (which handles big ints), we can mod by a large prime.
    # But Python ints are arbitrary precision, so we can just let them grow and then mod at the end.
    # However, for speed, we should mod periodically or use a fixed width.
    
    # Let's use a simpler approach: 
    # priority = sum(vector[i] * weight[i]) where weight[i] is a pseudo-random large integer.
    # But we can't precompute weights for arbitrary n easily without state.
    # We can generate weights on the fly based on index i.
    
    # Let's use a hash of the vector itself.
    # h = 0
    # for v in vector: h = (h * 31 + v) & 0xFFFFFFFF
    # This is O(n).
    
    # To make it more "chaotic" and less dependent on prefix, we can use a polynomial hash 
    # where the position matters more.
    
    h1 = 0
    h2 = 0
    for i, v in enumerate(vector):
        # Combine value and index to make it position-dependent
        h1 = (h1 * 31 + v + i) & 0x7FFFFFFF
        h2 = (h2 * 131 + v * (i + 1)) & 0x7FFFFFFF
        
    # Combine the two hashes into a single priority score
    # We want a float in [0, 1) or similar range, but higher is better.
    # Let's just return a large integer converted to float.
    # The evaluator sorts by this value.
    
    # To ensure a good distribution, we can mix the two hashes.
    final_hash = (h1 * 1000003) ^ h2
    return float(final_hash)
