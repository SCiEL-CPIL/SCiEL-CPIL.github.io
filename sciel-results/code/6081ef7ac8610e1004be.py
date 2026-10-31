import time
import hashlib
import struct

# EVOLVE-BLOCK-START

# Pre-compute a fixed random projection vector using a deterministic seed.
# This creates a chaotic, non-local ordering that decouples greedy selection
# from structured lattice symmetries.
_SEED = 42
_MAX_N = 12  # Support up to n=12 for safety

def _generate_projection_vector(n, seed):
    """Generate a deterministic pseudo-random projection vector in {-1, 1}^n."""
    # Use SHA256 to generate a long stream of random bits
    data = struct.pack('>I', seed)
    h = hashlib.sha256(data).digest()
    # Expand to enough bytes for n elements
    while len(h) < n:
        h = hashlib.sha256(h).digest()
    # Map each byte to -1 or 1
    return [1 if (b >> (i % 8)) & 1 == 0 else -1 for i, b in enumerate(h[:n])]

# Pre-compute projection vectors for common n values
_PROJECTION_VECTORS = {}
for _n in range(1, _MAX_N + 1):
    _PROJECTION_VECTORS[_n] = _generate_projection_vector(_n, _SEED + _n)

def priority(vector, n):
    """
    Priority function using random linear ordering.
    
    Maximizes the magnitude of projection onto a fixed random vector with
    entries in {-1, 1}. This imposes a chaotic, non-local ordering that
    decouples greedy selection from lattice symmetries and local density traps.
    """
    if n not in _PROJECTION_VECTORS:
        # Fallback: generate on the fly for larger n
        _PROJECTION_VECTORS[n] = _generate_projection_vector(n, _SEED + n)
    
    proj = _PROJECTION_VECTORS[n]
    
    # Compute projection: sum of vector[i] * proj[i]
    # vector entries are in {0, 1, 2}, proj entries are in {-1, 1}
    # We want to maximize |projection|, so use negative for descending order
    projection = 0
    for i in range(n):
        projection += vector[i] * proj[i]
    
    # Return negative projection so higher priority = larger |projection|
    # Add a small deterministic tiebreaker for exact ties
    if projection >= 0:
        tiebreak = 0
    else:
        tiebreak = 1
    
    # Scale to ensure integer-like behavior and add tiebreak
    return -(abs(projection) * 1000000 + tiebreak)

# EVOLVE-BLOCK-END
