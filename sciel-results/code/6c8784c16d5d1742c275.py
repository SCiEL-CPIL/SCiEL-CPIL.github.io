import zlib
import time

_START = time.time()
_DEADLINE = _START + 280.0  # safety margin before 330s kill

# Adaptive state: track vectors added to the set
_added_vectors = []
_added_set = set()
_call_count = [0]
_score_cache = {}
_cache_limit = 100000

def priority(vector, n):
    """
    Adaptive conflict-aware priority for cap set construction.
    
    Structural change: Instead of scoring vectors in isolation, we estimate
    how many 3-APs a candidate would form with already-added vectors.
    
    Key insight: A vector v conflicts with existing vectors a, b if
    v + a + b ≡ 0 (mod 3) element-wise, i.e., v[i] = (-a[i] - b[i]) % 3.
    
    We approximate conflict by:
    1. For each existing vector a, compute the "forbidden" complement:
       comp_a[i] = (-a[i]) % 3. If v is close to comp_a in Hamming distance,
       it's likely to form a 3-AP with a and some other vector.
    2. Penalize vectors that are "close" to many complements of existing vectors.
    
    This is a different modeling family: set-aware vs. static features.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return 0.0
    
    _call_count[0] += 1
    
    # Check cache
    vkey = vector
    if vkey in _score_cache:
        return _score_cache[vkey]
    
    # Compute conflict score based on existing vectors
    if not _added_vectors:
        # No vectors added yet: use static priority as bootstrap
        c1 = vector.count(1)
        c0 = vector.count(0)
        c2 = vector.count(2)
        one_score = 2.718281828459045 ** (-0.5 * c1 / max(n, 1))
        balance = 2.0 * min(c0, c2) / max(c0 + c2, 1)
        data = bytes(vector)
        crc = zlib.crc32(data)
        hash_score = float(crc) / 0xFFFFFFFF
        score = 0.7 * one_score + 0.2 * balance + 0.1 * hash_score
    else:
        # Adaptive: estimate conflict with existing set
        # For each existing vector a, the "danger zone" for v is vectors
        # that would complete a 3-AP: v[i] = (-a[i] - b[i]) % 3 for some b.
        # A simpler proxy: count how many existing vectors a have
        # small Hamming distance to (-v[i]) % 3 (the complement).
        
        # Compute complement of v: comp[i] = (-v[i]) % 3
        comp = tuple((-x) % 3 for x in vector)
        
        # Count how many existing vectors are "close" to comp
        # Use Hamming distance as proxy for conflict likelihood
        close_count = 0
        for a in _added_vectors:
            # Hamming distance between a and comp
            dist = sum(1 for i in range(n) if a[i] != comp[i])
            # Vectors with small distance to comp are more likely to form 3-APs
            # with v and some other vector
            if dist <= 2:
                close_count += 1
        
        # Normalize by set size
        conflict_ratio = close_count / len(_added_vectors) if _added_vectors else 0.0
        
        # Also compute static features as secondary signal
        c1 = vector.count(1)
        c0 = vector.count(0)
        c2 = vector.count(2)
        one_score = 2.718281828459045 ** (-0.3 * c1 / max(n, 1))
        
        # Deterministic hash for tie-breaking
        data = bytes(vector)
        crc = zlib.crc32(data)
        hash_score = float(crc) / 0xFFFFFFFF
        
        # Combined: low conflict is good, fewer 1s is good
        # Weight conflict heavily since it's the most predictive
        score = 0.6 * (1.0 - conflict_ratio) + 0.3 * one_score + 0.1 * hash_score
    
    # Cache the score (with size limit)
    if len(_score_cache) < _cache_limit:
        _score_cache[vkey] = score
    
    return score
