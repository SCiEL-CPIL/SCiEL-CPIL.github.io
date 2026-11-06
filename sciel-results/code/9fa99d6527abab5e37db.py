"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from functools import lru_cache

# EVOLVE-BLOCK-START

# Cache for precomputed cap sets per n
_cap_set_cache = {}
_deadline = None

def _init_deadline():
    global _deadline
    if _deadline is None:
        _deadline = time.time() + 290  # 290s safety margin

def _is_cap_set(S, n):
    """Check if S (set of tuples) is a cap set in F_3^n."""
    S_list = list(S)
    S_set = set(S_list)
    for i in range(len(S_list)):
        for j in range(i+1, len(S_list)):
            a = S_list[i]
            b = S_list[j]
            # c = -(a+b) mod 3
            c = tuple((-a[k] - b[k]) % 3 for k in range(n))
            if c in S_set and c != a and c != b:
                return False
    return True

def _greedy_cap_set(n, priority_func, max_time=10.0):
    """Greedy cap set construction with given priority function."""
    start = time.time()
    # Generate all vectors
    all_vecs = [tuple(v) for v in np.ndindex(*([3]*n))]
    # Sort by priority descending
    all_vecs.sort(key=lambda v: -priority_func(v, n))
    
    S = set()
    S_list = []
    # Precompute pair sums for conflict checking
    pair_sums = set()
    
    for v in all_vecs:
        if time.time() - start > max_time:
            break
        # Check if adding v creates a conflict
        # v conflicts if -v is in pair_sums of S
        neg_v = tuple((-x) % 3 for x in v)
        if neg_v not in pair_sums:
            # Add v
            # Update pair sums
            for u in S_list:
                s = tuple((u[k] + v[k]) % 3 for k in range(n))
                pair_sums.add(s)
            S.add(v)
            S_list.append(v)
    
    return S, S_list

def _local_search_improve(S, n, max_time=5.0):
    """Try to improve cap set S via local search (add rejected vectors)."""
    start = time.time()
    S_list = list(S)
    S_set = set(S)
    # Get all vectors not in S
    all_vecs = [tuple(v) for v in np.ndindex(*([3]*n))]
    rejected = [v for v in all_vecs if v not in S_set]
    
    # Sort rejected by some heuristic
    rejected.sort(key=lambda v: sum((x-1)**2 for x in v))
    
    improved = True
    while improved and time.time() - start < max_time:
        improved = False
        for v in rejected:
            if time.time() - start > max_time:
                break
            # Try to add v by removing one existing element
            # For each u in S, check if S - {u} + {v} is a cap set
            # This is expensive, so do it smartly
            # v conflicts with S if there exist a,b in S with a+b+v=0
            # After removing u, v conflicts if there exist a,b in S-{u} with a+b+v=0
            # So v is addable if the only conflicts involve u
            # Let's just try: remove each u, check if v can be added
            for u in S_list:
                if time.time() - start > max_time:
                    break
                # Check if S - {u} + {v} is a cap set
                # This is O(|S|^2) which is too slow for large S
                # Skip for now, just try adding v directly if no conflict with S-{u}
                # Simplified: just try to add v to S (without removing), if conflict, skip
                pass
        # This local search is too slow; skip for now
    return S

def priority(vector, n):
    _init_deadline()
    v = tuple(vector)
    
    # For small n, use precomputed cap set
    if n <= 8:
        if n not in _cap_set_cache:
            # Precompute a good cap set
            # Use greedy with a simple priority first
            def simple_priority(vec, n):
                return sum((x-1)**2 for x in vec)
            S, _ = _greedy_cap_set(n, simple_priority, max_time=5.0)
            _cap_set_cache[n] = S
        S = _cap_set_cache[n]
        if v in S:
            return 1.0
        else:
            # Priority based on distance to S
            min_dist = min(sum(1 for i in range(n) if v[i] != u[i]) for u in S)
            return 0.5 - min_dist / (2*n)
    else:
        # For larger n, use heuristic
        # Prioritize vectors with more "central" digits
        score = sum((x-1)**2 for x in v)
        return -score  # Lower score (more central) gets higher priority

# EVOLVE-BLOCK-END
