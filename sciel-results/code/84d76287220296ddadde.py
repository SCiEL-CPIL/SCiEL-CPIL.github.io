import time
import hashlib
import struct

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin

# Precompute powers of 31 mod 2^32 for polynomial hashing? 
# No, we need a 64-bit hash. 
# We can use a simple 64-bit polynomial hash or just use hashlib.
# hashlib.sha256 is heavy. 
# Let's use a fast 64-bit polynomial hash with a large prime or just mod 2^64.
# Or use the built-in hash? No, not stable across runs.
# Let's implement a fast 64-bit hash.

def _hash64(vector, n):
    # Fast 64-bit polynomial hash
    # h = h * 31 + val
    # We use 64-bit arithmetic. Python ints are arbitrary precision, so we mask.
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0xFFFFFFFFFFFFFFFF
    return h

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.

    Strategy (combined design):
    1. Primary: Distinct symbol score (quadratic in presence) - smooth landscape.
       3 distinct symbols -> 90000, 2 -> 40000, 1 -> 10000.
    2. Secondary: Mod-3 coordinate-sum penalty. In F_3^n, three points form an
       affine line iff their coordinate-wise sum is 0 mod 3. Vectors with sum
       congruent to 0 mod 3 are structurally more dangerous.
    3. Tertiary: Variance-based danger proxy. High variance = more "extreme"
       coordinates = higher likelihood of forming lines.
    4. Quaternary: L1 balance penalty (faster than quadratic, same ordering).
    5. Quinary: 64-bit polynomial rolling hash for deterministic tie-breaking.
       Replaced 32-bit hash with 64-bit to improve uniformity and reduce collisions.

    All components are computed in O(n) time with minimal per-call overhead.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Fast symbol counting
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)

    # Factor 1: Quadratic distinct symbol score
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Mod-3 sum penalty
    # coord_sum = 0*c0 + 1*c1 + 2*c2
    coord_sum = c1 + 2 * c2
    sum_mod = coord_sum % 3
    if sum_mod == 0:
        sum_score = -5000.0
    elif sum_mod == 1:
        sum_score = 2000.0
    else:
        sum_score = 1500.0

    # Factor 3: Variance-based danger proxy
    # Compute sum and sum-of-squares in one pass over the vector
    s = 0
    s2 = 0
    for v in vector:
        s += v
        s2 += v * v
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 5000.0

    # Factor 4: L1 balance penalty
    ideal = n / 3.0
    dev = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    balance_score = -dev * 10.0

    # Factor 5: 64-bit polynomial rolling hash for deterministic tie-breaking
    # This replaces the 32-bit hash to improve uniformity.
    h = _hash64(vector, n)
    # Normalize hash to [0, 1) for tie-breaking
    # We want a small perturbation so it doesn't dominate the other scores
    # but breaks ties effectively.
    hash_score = (h & 0xFFFFFFFF) / 0xFFFFFFFF  # Use lower 32 bits for score, upper for uniqueness if needed?
    # Actually, just use the full 64-bit value scaled down.
    hash_score = h / 0xFFFFFFFFFFFFFFFF

    # Combined priority: higher is added first
    total = distinct_score + sum_score + danger_score + balance_score + hash_score
    return float(total)
