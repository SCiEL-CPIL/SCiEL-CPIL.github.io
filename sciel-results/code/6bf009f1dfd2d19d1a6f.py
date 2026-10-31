import time
from collections import Counter
import itertools

_START = time.time()
_DEADLINE = _START + 280.0

_cache = {}

def _generate_vectors(n):
    return [tuple(v) for v in itertools.product([0, 1, 2], repeat=n)]

def _base_score(vector, n):
    c = Counter(vector)
    c0 = c.get(0, 0)
    c1 = c.get(1, 0)
    c2 = c.get(2, 0)
    
    distinct_count = sum(1 for c_val in [c0, c1, c2] if c_val > 0)
    sum_sq = c1 + 4 * c2
    
    ideal = n / 3.0
    dev = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    
    # Higher distinct count is better
    # Higher sum_sq is better (arbitrary choice, consistent with previous)
    # Lower deviation is better
    return distinct_count * 10000.0 + sum_sq * 100.0 - dev * 10.0

def _compute_penalties(vectors, base_scores, n):
    num_vecs = len(vectors)
    penalties = [0.0] * num_vecs
    # Map vector to index
    vec_to_idx = {v: i for i, v in enumerate(vectors)}
    
    # Pre-compute "neg" operation?
    # w = -v - u (mod 3)
    # For each v, iterate u, find w.
    
    # Optimization:
    # For a fixed v, the pairs {u, w} are disjoint.
    # We can just iterate u, and if u < w (some ordering), process.
    # But we need to sum min(B[u], B[w]) for all pairs.
    
    for i, v in enumerate(vectors):
        total_pen = 0.0
        # To avoid double counting and self-pairs, we can just iterate all u != v
        # and add 0.5 * min(B[u], B[w])?
        # No, if we iterate all u, we count each pair twice (once for u, once for w).
        # So sum over all u != v of 0.5 * min(B[u], B[w]) is correct, provided w != v.
        # If w == v, then 2u = -v => u = 2v. This happens for exactly one u.
        # In that case, the "pair" is {u, u}. Is it a line?
        # Line is {v, u, u}. This requires 3u + v = 0? No, v+u+u = 0 => v = -2u = u.
        # So if v=u, it's not a distinct triplet.
        # The condition for a line is distinct points.
        # So we must exclude cases where u == w.
        # u == w => 2u = -v => u = 2v.
        # So for u = 2v, w = 2v. This is not a valid line (points not distinct).
        # So we skip u if w == u.
        
        for j, u in enumerate(vectors):
            if i == j: continue # u == v
            # Compute w = -v - u mod 3
            w = tuple((3 - (v[k] + u[k]) % 3) % 3 for k in range(n))
            if w == u: continue # Not a distinct triplet
            # Find index of w
            # This lookup is O(1) but inside a loop of 3^n * 3^n.
            # For n=7, 4.7M lookups.
            # To speed up, we can pre-compute the index of w?
            # Or just use the dict.
            
            # Optimization: We are doing this for EVERY v.
            # This is O(3^(2n)).
            # For n=7, 4.7M iterations. Python loop ~ 0.5-1s.
            # For n=8, 43M iterations. Python loop ~ 5-10s.
            # Acceptable.
            
            k = vec_to_idx.get(w)
            if k is None:
                continue # Should not happen
            # We count each pair {u, w} twice (once when u is the current, once when w is the current? No, we are fixing v).
            # For a fixed v, as we iterate u, we hit both u and w.
            # So we sum 0.5 * min(B[u], B[w]).
            bu = base_scores[j]
            bw = base_scores[k]
            total_pen += 0.5 * min(bu, bw)
            
        penalties[i] = total_pen
        
    return penalties

def _precompute(n):
    if time.time() - _START > _DEADLINE - 10.0:
        return None # Timeout risk
        
    vectors = _generate_vectors(n)
    base_scores = [_base_score(v, n) for v in vectors]
    
    if n <= 7:
        penalties = _compute_penalties(vectors, base_scores, n)
        # Normalize penalty
        num_pairs = (len(vectors) - 1) / 2.0
        if num_pairs > 0:
            avg_penalties = [p / num_pairs for p in penalties]
        else:
            avg_penalties = [0.0] * len(vectors)
        
        # Combine
        # Scale factor: Base score is ~10000-30000.
        # Avg penalty is ~10000-30000 (since it's an average of base scores).
        # So they are on similar scales.
        # Score = Base - Avg_Penalty
        final_scores = [b - p for b, p in zip(base_scores, avg_penalties)]
    else:
        # For large n, skip penalty
        final_scores = base_scores
        
    score_map = {v: s for v, s in zip(vectors, final_scores)}
    return score_map

def priority(vector, n):
    """
    Return a float priority for a vector in {0,1,2}^n.
    """
    # Deadline check
    if time.time() - _START > _DEADLINE:
        return 0.0
    
    if n not in _cache:
        # Pre-compute if not already done
        # Check time before starting heavy pre-computation
        if time.time() - _START > _DEADLINE - 60.0:
            # Not enough time for pre-computation, use fallback
            _cache[n] = None # Mark as fallback
        else:
            _cache[n] = _precompute(n)
            
    if _cache[n] is None:
        # Fallback to simple heuristic
        return _base_score(vector, n)
        
    return _cache[n][vector]
