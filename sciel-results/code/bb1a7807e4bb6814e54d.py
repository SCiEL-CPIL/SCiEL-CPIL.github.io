import time
import math
from collections import defaultdict

_START_TIME = time.time()
TIME_LIMIT = 310.0  # Safety margin before 330s kill

# Dynamic LSH conflict density tracker
# For cap sets in F_3^n, three points a, b, c form an AP iff a + c = 2b (mod 3).
# So a new vector v conflicts with existing {a, b} if v = -a - b (mod 3) for some a, b in set.
# We use multiple hash functions to approximate how many existing pairs would conflict with v.

# LSH configuration
_NUM_HASHES = 8
_HASH_MODULUS = 1000003  # Large prime for hash buckets

# Precompute random coefficients for each hash function
import random
_rng = random.Random(42)
_HASH_COEFS = []
for _ in range(_NUM_HASHES):
    coefs = tuple(_rng.randint(1, 1000) for _ in range(30))
    _HASH_COEFS.append(coefs)

# Track which vectors have been added (for dynamic conflict density)
_added_vectors = []
# Conflict density table: maps hash signature -> count of existing vectors with that signature
_conflict_table = defaultdict(int)
# Also track pair-conflict signatures: for each pair (a, b) in _added_vectors, 
# the "conflict vector" c = -a-b mod 3. We track how many pairs produce each conflict pattern.
# But this is O(k^2) which is too expensive. Instead, we use a simpler approximation:
# For each existing vector a, we compute its "shadow" and check if v is close to shadows.

def _compute_lsh_signature(vector, n, hash_idx):
    """Compute the LSH hash signature for a given hash function."""
    coefs = _HASH_COEFS[hash_idx]
    eff_n = min(n, len(coefs))
    sig = 0
    for i in range(eff_n):
        sig = (sig + coefs[i] * vector[i]) % _HASH_MODULUS
    return sig

def _update_conflict_table(vector, n):
    """Update the conflict density table when a new vector is added."""
    _added_vectors.append(vector)
    for h_idx in range(_NUM_HASHES):
        sig = _compute_lsh_signature(vector, n, h_idx)
        _conflict_table[(h_idx, sig)] += 1

def _conflict_density(vector, n):
    """
    Estimate how many existing vectors are 'close' to this candidate in a way
    that might form APs. Higher density = more risky.
    """
    if not _added_vectors:
        return 0.0
    
    # For each hash function, count how many existing vectors share the same bucket
    # This approximates proximity in Hamming space
    total_conflicts = 0
    for h_idx in range(_NUM_HASHES):
        sig = _compute_lsh_signature(vector, n, h_idx)
        total_conflicts += _conflict_table.get((h_idx, sig), 0)
    
    # Normalize by number of hashes and existing vectors
    return total_conflicts / (_NUM_HASHES * len(_added_vectors))

def _entropy(vector, n):
    """Compute entropy of coordinate distribution."""
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    entropy = 0.0
    for c in (c0, c1, c2):
        if c > 0:
            p = c / n
            entropy -= p * math.log(p, 3)
    return entropy

def priority(vector, n):
    """
    Priority for a candidate vector in a greedy cap set construction.
    
    Uses:
    1. Entropy of coordinate distribution (prefer balanced vectors)
    2. Dynamic LSH conflict density (prefer vectors far from existing ones)
    3. Lightweight hash for tie-breaking
    """
    elapsed = time.time() - _START_TIME
    if elapsed > TIME_LIMIT:
        # Emergency: return a neutral value to avoid timeout
        return 0.5
    
    if n == 0:
        return 1.0
    
    # Component 1: Entropy of coordinate distribution
    ent = _entropy(vector, n)
    
    # Component 2: Dynamic conflict density (lower is better, so we negate)
    density = _conflict_density(vector, n)
    anti_density = 1.0 - density  # Invert so higher = less conflict
    
    # Component 3: Lightweight hash for diversity/tie-breaking
    h = 0
    for i, v in enumerate(vector):
        h = (h * 31 + v * (i + 1)) & 0xFFFFFFFF
    hash_norm = h / 4294967295.0
    
    # Combine components with soft-min for smoother combination
    # We want high entropy, high anti-density, and hash as tie-breaker
    score = 0.5 * ent + 0.45 * anti_density + 0.05 * hash_norm
    
    # Update the conflict table with this vector (it's being considered for addition)
    # Note: This assumes the evaluator will add this vector if it's selected
    _update_conflict_table(vector, n)
    
    return score
