import hashlib
import zlib
import itertools

# Global cache for pre-computed priority maps
_prio_cache = {}

# Maximum n for which we pre-compute the full sorted order
# 3^14 = 4,782,969 vectors. Sorting this takes ~2-5s and ~500MB RAM.
# 3^15 = 14,348,907 vectors. Might be too slow/memory heavy.
# Let's stick to 14 to be safe with the 300s limit and memory.
_MAX_PRECOMPUTE_N = 14

def _get_prio_map(n):
    """
    Pre-computes a priority map for all vectors in {0,1,2}^n.
    The priority is the index in a list sorted by a high-quality random hash.
    Higher index means higher priority (added first in descending order).
    """
    if n in _prio_cache:
        return _prio_cache[n]
    
    # Generate all vectors
    # For n=14, this is ~4.8M tuples.
    vectors = list(itertools.product([0, 1, 2], repeat=n))
    
    # Define sorting key: high-quality hash
    def sort_key(v):
        # blake2b is fast and provides good distribution
        # digest_size=8 gives 64 bits
        h = int(hashlib.blake2b(v, digest_size=8).hexdigest(), 16)
        return h
        
    # Sort vectors by hash
    # This creates a random permutation
    sorted_vectors = sorted(vectors, key=sort_key)
    
    # Create mapping: vector -> priority (index)
    # The last vector in sorted_vectors has the highest hash, thus highest priority
    prio_map = {v: i for i, v in enumerate(sorted_vectors)}
    
    _prio_cache[n] = prio_map
    return prio_map

def priority(vector, n):
    """
    Returns a priority float for the given vector.
    Higher priority means the vector is considered earlier in the greedy process.
    """
    if n <= _MAX_PRECOMPUTE_N:
        prio_map = _get_prio_map(n)
        return float(prio_map[vector])
    else:
        # Fallback for large n: direct hash
        # Use zlib.crc32 for speed
        # Convert vector to bytes
        # vector is a tuple of ints 0, 1, 2
        v_bytes = bytes(vector)
        
        # Get a 64-bit hash to reduce collisions
        h1 = zlib.crc32(v_bytes)
        h2 = zlib.crc32(v_bytes[::-1])
        
        # Combine into a 64-bit integer
        h = (h1 << 32) | h2
        
        return float(h)
