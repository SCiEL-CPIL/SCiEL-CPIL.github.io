import time
import math

_START_TIME = time.time()
TIME_LIMIT = 320.0  # Safety margin before 330s kill

# Static configuration for hash functions
# We use a deterministic set of multipliers to create a high-quality hash
# that serves as a tie-breaker and diversity enhancer.
_HASH_MULTIPLIERS = [
    17, 31, 53, 97, 193, 389, 769, 1543, 3079, 6151, 12301, 24601, 49201, 98401, 196801, 393601
]

def _fast_hash(vector, n):
    """
    Compute a high-resolution hash for the vector.
    We mix the vector values with the multipliers to create a pseudo-random
    distribution that depends on the vector's content and position.
    """
    h = 0
    for i in range(n):
        # Mix in the value and the index
        h = (h * _HASH_MULTIPLIERS[i % len(_HASH_MULTIPLIERS)] + vector[i] * (i + 1)) & 0xFFFFFFFF
    # Final avalanche mixing
    h ^= (h >> 16)
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h ^= (h >> 16)
    return h

def _entropy(vector, n):
    """Compute entropy of coordinate distribution."""
    if n == 0:
        return 0.0
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
    
    entropy = 0.0
    for c in (c0, c1, c2):
        if c > 0:
            p = c / n
            entropy -= p * math.log(p, 3)
    return entropy

def priority(vector, n):
    """
    Priority for a candidate vector in a greedy cap set construction.
    
    Structural Change:
    1. Pure function (no dynamic state) to avoid iteration-order bias.
    2. High-weight Entropy component to select structurally "central" vectors.
    3. High-resolution Hash component for tie-breaking and diversity.
    4. Weight-Interleaving: The hash is perturbed by the Hamming weight to ensure
       that vectors of similar structural complexity (weight) are not always grouped
       together in the priority sort, promoting a mixed-weight cap set.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > TIME_LIMIT:
        return 0.5
    
    if n == 0:
        return 1.0
    
    # Component 1: Entropy (0 to 1.0)
    # High entropy means balanced 0,1,2 distribution.
    ent = _entropy(vector, n)
    
    # Component 2: Hamming Weight (number of non-zeros)
    # Used to perturb the hash for diversity
    weight = sum(1 for v in vector if v != 0)
    
    # Component 3: Hash (0 to 1.0)
    # We use the hash as a tie-breaker. To prevent the hash from being
    # correlated with the vector's "quality" in a bad way, we mix it
    # with the weight. This ensures that vectors with the same entropy
    # but different weights are interleaved in the sort.
    h_val = _fast_hash(vector, n)
    # Perturb hash with weight to break potential correlations
    h_val = (h_val ^ (weight * 0x9E3779B9)) & 0xFFFFFFFF
    hash_norm = h_val / 4294967295.0
    
    # Combine components
    # We prioritize Entropy strongly, as balanced vectors are generally
    # better seeds for cap sets. The hash provides the necessary
    # stochasticity to explore different parts of the space.
    # Weighting: 80% Entropy, 20% Hash
    score = 0.8 * ent + 0.2 * hash_norm
    
    return score
