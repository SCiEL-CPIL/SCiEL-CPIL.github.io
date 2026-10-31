import time
import hashlib
import zlib
import math
from collections import Counter

_START = time.time()
_DEADLINE = _START + 310.0  # Safety margin before 330s kill

_cache = {}

def _generate_vectors(n):
    """Generate all vectors in {0,1,2}^n with a hash-based ordering for diversity."""
    vectors = []
    for coords in range(3 ** n):
        # Convert integer to base-3 representation
        v = [0] * n
        temp = coords
        for i in range(n):
            v[i] = temp % 3
            temp //= 3
        vectors.append(tuple(v))
    
    # Sort by a hash to break symmetries and ensure diverse ordering
    # Using SHA256 for better uniformity than 32-bit hash
    def hash_key(v):
        # Create a string representation for hashing
        s = ''.join(str(c) for c in v)
        return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)
    
    vectors.sort(key=hash_key)
    return vectors

def _base_score(vector, n):
    """
    Score vectors based on multiple features:
    1. Balance of 0s, 1s, 2s (closer to n/3 each is better)
    2. Distinct value count (more distinct is better)
    3. Sum of squares (arbitrary but consistent)
    4. A small random-like component based on hash for tie-breaking
    """
    c = Counter(vector)
    c0 = c.get(0, 0)
    c1 = c.get(1, 0)
    c2 = c.get(2, 0)
    
    distinct_count = sum(1 for c_val in [c0, c1, c2] if c_val > 0)
    sum_sq = c1 + 4 * c2
    
    ideal = n / 3.0
    dev = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    
    # Add a small hash-based component for tie-breaking and diversity
    s = ''.join(str(c) for c in vector)
    hash_val = int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) / (16 ** 8)
    
    # Higher distinct count is better
    # Higher sum_sq is better
    # Lower deviation is better
    # Small hash component for diversity
    return distinct_count * 10000.0 + sum_sq * 100.0 - dev * 10.0 + hash_val * 0.01

def _compute_penalties(vectors, base_scores, n):
    """
    Compute penalty for each vector based on how many lines it forms with other vectors.
    A line in {0,1,2}^n is a set {v, u, w} where v + u + w = 0 (mod 3) and all distinct.
    Penalty for v = sum over all u != v of min(B[u], B[w]) where w = -v-u mod 3
    """
    num_vecs = len(vectors)
    penalties = [0.0] * num_vecs
    vec_to_idx = {v: i for i, v in enumerate(vectors)}
    
    for i, v in enumerate(vectors):
        total_pen = 0.0
        for j, u in enumerate(vectors):
            if i == j:
                continue
            # Compute w = -v - u mod 3
            w = tuple((3 - (v[k] + u[k]) % 3) % 3 for k in range(n))
            if w == u:
                continue  # Not a distinct triplet
            k = vec_to_idx.get(w)
            if k is None:
                continue
            bu = base_scores[j]
            bw = base_scores[k]
            total_pen += 0.5 * min(bu, bw)
        penalties[i] = total_pen
    
    return penalties

def _precompute(n):
    """Precompute scores for all vectors of dimension n."""
    if time.time() - _START > _DEADLINE - 5.0:
        return None
    
    vectors = _generate_vectors(n)
    base_scores = [_base_score(v, n) for v in vectors]
    
    if n <= 6:
        # For small n, compute full penalties
        penalties = _compute_penalties(vectors, base_scores, n)
        num_pairs = (len(vectors) - 1) / 2.0
        if num_pairs > 0:
            avg_penalties = [p / num_pairs for p in penalties]
        else:
            avg_penalties = [0.0] * len(vectors)
        final_scores = [b - p for b, p in zip(base_scores, avg_penalties)]
    else:
        # For larger n, skip penalties due to time constraints
        final_scores = base_scores
    
    score_map = {v: s for v, s in zip(vectors, final_scores)}
    return score_map

def priority(vector, n):
    """
    Return a float priority for a vector in {0,1,2}^n.
    Vectors are added greedily in descending priority when they keep the set a cap set.
    """
    # Deadline check
    if time.time() - _START > _DEADLINE:
        return 0.0
    
    if n not in _cache:
        if time.time() - _START > _DEADLINE - 30.0:
            _cache[n] = None
        else:
            _cache[n] = _precompute(n)
    
    if _cache[n] is None:
        return _base_score(vector, n)
    
    return _cache[n][vector]
