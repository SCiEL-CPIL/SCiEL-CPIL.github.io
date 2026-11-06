"""
Strategy: Offline search for the best static priority ordering.
Instead of using a single fixed hash-based priority, we simulate the greedy 
cap set construction for several candidate ordering strategies during the 
first call to `priority` (or on a global init) and cache the best one.
Subsequent calls use the best ordering found.

Candidate strategies:
1. Lexicographic order (baseline)
2. Reverse lexicographic
3. Sum of coordinates (ascending)
4. Sum of coordinates (descending)
5. Number of 0s (descending) - favors sparse vectors
6. Number of 2s (descending)
7. A few random linear projections (fixed seeds for determinism)
8. The original hash-based priority
"""

import time
import math
import itertools
import random

# EVOLVE-BLOCK-START

# Global state for the offline search
_best_priority_func = None
_search_done = False
_deadline = time.time() + 290  # 290s to be safe within 300s limit

def _generate_all_vectors(n):
    """Generate all vectors in {0, 1, 2}^n as tuples."""
    return list(itertools.product([0, 1, 2], repeat=n))

def _greedy_cap_set_size(vectors, n, priority_func):
    """
    Simulate the greedy algorithm with a given priority function.
    Returns the size of the resulting cap set.
    """
    # Sort vectors by priority (descending)
    sorted_vectors = sorted(vectors, key=lambda v: priority_func(v, n), reverse=True)
    
    cap_set = set()
    # For efficiency in checking, we can just check against the current set
    # A vector v can be added if for no a in cap_set does (-a - v) mod 3 exist in cap_set
    # and is distinct from a and v.
    
    for v in sorted_vectors:
        # Check if adding v violates cap set property
        can_add = True
        for a in cap_set:
            # c = -a - v mod 3
            c = tuple((-a[i] - v[i]) % 3 for i in range(n))
            if c in cap_set:
                # Check if c is distinct from a and v
                if c != a and c != v:
                    can_add = False
                    break
                # If c == a, then 2a + v = 0 => v = -2a = a (mod 3)? No.
                # If c == a, then a+a+v = 0 => 2a+v=0 => v = -2a = a (since -2=1 mod 3).
                # So if v == a, but v is not in cap_set yet (we are adding it), so v != a.
                # If c == v, then a + v + v = 0 => a = -2v = v (mod 3). So a == v.
                # But a is in cap_set, v is not yet. So a != v.
                # Thus, if c in cap_set, it must be distinct from a and v?
                # Wait, if c == a, then v = a. But v is being added, so v not in cap_set. a is in cap_set. So v != a.
                # If c == v, then a = v. Same logic.
                # So any collision c in cap_set implies a line {a, v, c} with distinct elements.
                pass
        
        if can_add:
            cap_set.add(v)
            
    return len(cap_set)

def _create_hash_priority(seed=12345):
    """Create a priority function based on a hash with a specific seed."""
    def priority(vector, n):
        val = 0
        for i, v in enumerate(vector):
            val = val * 3 + v
        # Simple hash
        h = (val + seed) & 0xFFFFFFFF
        h = (h ^ (h >> 16)) & 0xFFFFFFFF
        h = (h * 0x45d9f3b) & 0xFFFFFFFF
        h = (h ^ (h >> 16)) & 0xFFFFFFFF
        return float(h)
    return priority

def _run_offline_search(n):
    """
    Run a quick search over different priority strategies to find the best one.
    This is called once, on the first invocation of priority().
    """
    global _best_priority_func, _search_done
    if _search_done:
        return
    
    _search_done = True
    
    if n > 8:
        # For larger n, simulation is expensive. Fall back to a good heuristic.
        # Let's use a simple linear projection which is fast to compute and sort.
        _best_priority_func = lambda v, n: sum(v[i] * (i + 1) for i in range(n))
        return

    # Generate all vectors
    all_vectors = _generate_all_vectors(n)
    
    # Candidate priority functions
    candidates = []
    
    # 1. Lexicographic (natural order)
    candidates.append(("lex", lambda v, n: float(sum(v[i] * (3 ** (n - 1 - i)) for i in range(n)))))
    
    # 2. Reverse Lexicographic
    candidates.append(("rev_lex", lambda v, n: -float(sum(v[i] * (3 ** (n - 1 - i)) for i in range(n)))))
    
    # 3. Sum of coordinates (ascending -> lower sum first? No, descending priority means higher value first)
    # Let's try favoring low sum (sparse) or high sum.
    candidates.append(("sum_asc", lambda v, n: -sum(v)))  # Higher priority for lower sum
    candidates.append(("sum_desc", lambda v, n: sum(v)))   # Higher priority for higher sum
    
    # 4. Count of 0s
    candidates.append(("zeros_desc", lambda v, n: -v.count(0))) # Higher priority for more zeros
    candidates.append(("zeros_asc", lambda v, n: v.count(0)))
    
    # 5. Count of 1s
    candidates.append(("ones_desc", lambda v, n: -v.count(1)))
    
    # 6. Count of 2s
    candidates.append(("twos_desc", lambda v, n: -v.count(2)))
    
    # 7. Original Hash
    candidates.append(("hash", _create_hash_priority(12345)))
    
    # 8. A few random linear projections
    random.seed(42)
    for i in range(5):
        weights = [random.uniform(-1, 1) for _ in range(n)]
        candidates.append((f"rand_{i}", lambda v, n, w=weights: sum(v[i] * w[i] for i in range(n))))
        
    # 9. Alternating weights (checkerboard-like)
    weights_alt = [1 if i % 2 == 0 else -1 for i in range(n)]
    candidates.append(("alt", lambda v, n, w=weights_alt: sum(v[i] * w[i] for i in range(n))))
    
    best_score = -1
    best_func = None
    
    for name, func in candidates:
        if time.time() > _deadline - 10:
            break
            
        try:
            score = _greedy_cap_set_size(all_vectors, n, func)
            if score > best_score:
                best_score = score
                best_func = func
        except Exception:
            pass
            
    if best_func is None:
        # Fallback to hash
        best_func = _create_hash_priority(12345)
        
    _best_priority_func = best_func

def priority(vector, n):
    """
    Returns the priority for a given vector.
    On the first call, it triggers an offline search to determine the best static ordering.
    """
    global _best_priority_func, _search_done
    
    if not _search_done:
        _run_offline_search(n)
        
    if _best_priority_func is None:
        # Absolute fallback
        return float(hash(vector))
        
    return _best_priority_func(vector, n)

# EVOLVE-BLOCK-END
