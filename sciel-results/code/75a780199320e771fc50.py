import time
import math
import random
from collections import defaultdict

_START_TIME = time.time()
TIME_LIMIT = 310.0  # Safety margin before 330s kill

# Dynamic LSH conflict density tracker
# Key insight: in F_3^n, three points a, b, c form an AP iff a + c = 2b (mod 3)
# i.e., a + c = -b (mod 3). So a new vector v conflicts with existing {a, b}
# if v = -a - b (mod 3). We can't check all pairs, but we can use bucket-based
# approximation: hash vectors into buckets, and count how many existing vectors
# share the same bucket signature. High conflict density = risky.

# LSH configuration: multiple hash functions mapping vectors to small buckets
_NUM_HASHES = 8
_BUCKET_SIZE = 16  # Each hash maps to one of _BUCKET_SIZE buckets
_MAX_DIM_TRACKED = 40  # Track up to this dimension for LSH signatures

# Precompute random hash parameters for LSH
_rng = random.Random(42)
# For each hash function, we have random weights for each coordinate
# h_i(v) = sum_j w_{i,j} * v_j mod _BUCKET_SIZE
_HASH_WEIGHTS = []
for _ in range(_NUM_HASHES):
    weights = [_rng.randint(0, _BUCKET_SIZE - 1) for _ in range(_MAX_DIM_TRACKED)]
    _HASH_WEIGHTS.append(weights)

# Conflict table: maps LSH signature (tuple of bucket indices) -> count of vectors with that signature
_conflict_table = defaultdict(int)
_total_vectors_tracked = 0

def _compute_lsh_signature(vector, n):
    """Compute LSH signature for a vector: tuple of bucket indices."""
    eff_n = min(n, _MAX_DIM_TRACKED)
    if eff_n == 0:
        return tuple([0] * _NUM_HASHES)
    
    sig = []
    for h_idx in range(_NUM_HASHES):
        bucket = 0
        weights = _HASH_WEIGHTS[h_idx]
        for i in range(eff_n):
            bucket += weights[i] * vector[i]
        sig.append(bucket % _BUCKET_SIZE)
    return tuple(sig)

def _update_conflict_table(vector, n):
    """Update the conflict table with a new vector (called after adding to cap set)."""
    global _total_vectors_tracked
    if n > _MAX_DIM_TRACKED:
        return
    sig = _compute_lsh_signature(vector, n)
    _conflict_table[sig] += 1
    _total_vectors_tracked += 1

def _conflict_density(vector, n):
    """
    Estimate conflict density for a candidate vector.
    Returns the number of existing vectors that share the same LSH signature.
    Higher = more likely to form AP with existing vectors.
    """
    if n > _MAX_DIM_TRACKED or _total_vectors_tracked == 0:
        return 0.0
    
    sig = _compute_lsh_signature(vector, n)
    return _conflict_table.get(sig, 0)


def priority(vector, n):
    """
    Priority for a candidate vector in a greedy cap set construction.
    
    Uses:
    1. Entropy of coordinate distribution (prefer balanced vectors)
    2. Dynamic LSH conflict density (prefer vectors far from existing cap set members)
    3. A lightweight hash-based diversity term
    
    The LSH conflict density is the key structural change: it adapts dynamically
    as vectors are added to the cap set, providing a proxy for how many existing
    vectors are "close" to the candidate in the F3 metric.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > TIME_LIMIT:
        # Emergency: return a neutral value to avoid timeout
        return 0.5
    
    if n == 0:
        return 1.0
    
    # Component 1: Entropy of coordinate distribution
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    entropy = 0.0
    for c in (c0, c1, c2):
        if c > 0:
            p = c / n
            entropy -= p * math.log(p, 3)
    # entropy is in [0, 1], higher is better (more balanced)
    
    # Component 2: Dynamic LSH conflict density
    # Get the conflict density for this candidate
    conflict = _conflict_density(vector, n)
    # Normalize: conflict is in [0, _total_vectors_tracked]
    # We want LOW conflict, so we use 1 - normalized_conflict
    if _total_vectors_tracked > 0:
        norm_conflict = conflict / _total_vectors_tracked
        anti_conflict = 1.0 - norm_conflict
    else:
        anti_conflict = 1.0
    
    # Component 3: Lightweight hash for diversity
    # Use a simple hash to break ties and add diversity
    h = 0
    for i, v in enumerate(vector):
        h = (h * 31 + v * (i + 1)) & 0xFFFFFFFF
    hash_norm = h / 4294967295.0
    
    # Combine components
    # We want high entropy, high anti-conflict (low conflict density), and hash as tie-breaker
    # The hash should have low weight
    score = 0.5 * entropy + 0.4 * anti_conflict + 0.1 * hash_norm
    
    # Update conflict table with this vector (it's being added to the cap set)
    _update_conflict_table(vector, n)
    
    return score
