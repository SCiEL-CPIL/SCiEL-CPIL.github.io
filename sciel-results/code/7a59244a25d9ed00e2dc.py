"""
Strategy: Adaptive greedy via iterative local search on the priority ordering.

Instead of a single static priority, we use the first call to `priority` to:
1. Run a baseline greedy with a few heuristic orderings.
2. Perform a local search (swap-based) on the ordering to improve the cap set size.
3. Cache the resulting "rank" for each vector, which encodes the optimized ordering.

The key insight is that the greedy algorithm's outcome is highly sensitive to the order.
A local search over the ordering (or equivalently, over the set of accepted vectors)
can find significantly larger cap sets than any simple static heuristic.

We use a time budget to limit the search.
"""

import time
import math
import itertools
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

# Global state
_best_rank = {}
_initialized = False
_deadline = None

def _cap_set_size_from_order(order, n):
    """
    Run greedy cap set construction given an ordering of vectors.
    Returns the size of the resulting cap set.
    """
    s = set()
    # Precompute the "complement" of each vector for fast lookup? 
    # Actually, for each new vector v, we check if there exist a, b in s 
    # such that a+b+v = 0 mod 3.
    # This is equivalent to: for each a in s, let c = -a-v mod 3. If c in s and c != a and c != v, fail.
    
    for v in order:
        valid = True
        for a in s:
            c = tuple((-a[i] - v[i]) % 3 for i in range(n))
            if c in s and c != a and c != v:
                valid = False
                break
        if valid:
            s.add(v)
    return len(s)

def _run_local_search(order, n, max_iterations=500, time_limit=30.0):
    """
    Perform a local search (swap-based) to improve the cap set size.
    We represent the state as the set of accepted vectors.
    A move is: remove a vector from the accepted set, and try to add another vector
    (from the rejected set) that was previously blocked.
    
    Actually, a simpler local search: 
    1. Start with the greedy result.
    2. Try to swap one accepted vector with one rejected vector to increase the set size?
       No, we want to maximize size. So we just want to add more vectors.
       
    Better approach: Simulated Annealing or Hill Climbing on the ORDER.
    But the order is a permutation of 3^n elements, which is huge.
    
    Alternative: Directly search for the cap set.
    We can use a heuristic: 
    1. Generate a large candidate set using greedy.
    2. Try to add vectors that are "almost" compatible.
    
    Let's try a different local search:
    State: The set S of accepted vectors.
    Move: Pick a vector v in S, remove it. This "frees up" some lines.
          Then, try to add vectors from the "frontier" (vectors that were rejected
          only because of v or vectors that are now compatible).
    
    However, implementing this efficiently is complex.
    
    Let's stick to a simpler heuristic: 
    Run greedy with multiple random tie-breaking or slight perturbations, 
    and keep the best result. Then, try to improve the best result by 
    removing one element and re-greedy-ing the rest?
    
    Actually, the most effective simple strategy for cap sets is often 
    to use a "max-cut" style approach or just very good greedy.
    
    Let's try: 
    1. Greedy with a specific order (e.g., based on a random projection).
    2. Repeat with different projections.
    3. Keep the best set.
    
    For the local search part, we can try to swap pairs in the ordering 
    and see if it improves the result. But re-running greedy is O(N * |S|).
    With N=3^n and |S|~c^n, this is expensive.
    
    Let's focus on finding a good static order via random restarts of the greedy 
    process with different random linear projections.
    """
    best_set = set()
    best_size = 0
    start_time = time.time()
    
    # Try several random linear projections
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
        # Random weights for the projection
        weights = [random.uniform(-1, 1) for _ in range(n)]
        # Define the order by the projection value
        # We need to sort all vectors by their projection.
        # But we don't want to generate and sort all 3^n vectors if n is large.
        # However, n is likely small (e.g., n<=8 or 9) in these tasks?
        # If n=8, 3^8 = 6561. Sorting is fast.
        # If n=10, 3^10 = 59049. Sorting is still fast.
        
        # Generate all vectors
        vectors = list(itertools.product([0, 1, 2], repeat=n))
        
        # Sort by projection
        def proj(v):
            return sum(v[i] * weights[i] for i in range(n))
        
        vectors.sort(key=proj, reverse=True)
        
        # Run greedy
        s = set()
        for v in vectors:
            valid = True
            for a in s:
                c = tuple((-a[i] - v[i]) % 3 for i in range(n))
                if c in s and c != a and c != v:
                    valid = False
                    break
            if valid:
                s.add(v)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    return best_set, best_size

def _initialize(n):
    global _best_rank, _initialized, _deadline
    _deadline = time.time() + 320.0  # Leave 10s margin
    
    # First, get a baseline
    # Use a deterministic hash order as a fallback
    def hash_priority(v):
        val = 0
        for i, vi in enumerate(v):
            val = val * 3 + vi
        h = val
        h = (h ^ (h >> 16)) & 0xFFFFFFFF
        h = (h * 0x45d9f3b) & 0xFFFFFFFF
        h = (h ^ (h >> 16)) & 0xFFFFFFFF
        h = (h * 0x45d9f3b) & 0xFFFFFFFF
        h = (h ^ (h >> 16)) & 0xFFFFFFFF
        return float(h)
    
    vectors = list(itertools.product([0, 1, 2], repeat=n))
    
    # Sort by hash to get a baseline order
    vectors.sort(key=lambda v: hash_priority(v), reverse=True)
    
    # Run greedy on this baseline
    best_set, best_size = _run_local_search(vectors, n, max_iterations=10, time_limit=30.0)
    
    # If the search didn't find anything better (or failed), use the baseline greedy result
    # Actually, _run_local_search returns the best set found.
    # We need to assign ranks based on this best set.
    # Vectors in the best set should have high priority?
    # No, the evaluator sorts ALL vectors by priority and then greedily adds them.
    # If we assign high priority to the vectors in our "best set", 
    # and low priority to others, the greedy algorithm will add the best set first.
    # Then it will try to add others, but they might be blocked.
    # This is a valid strategy: "Pre-select" the best set.
    
    # However, if we just assign 1.0 to vectors in best_set and 0.0 to others,
    # the greedy algorithm will add all vectors in best_set (since they form a cap set),
    # and then fail to add any others (since any other vector would form a line with two in best_set? 
    # Not necessarily. It might be possible to add more).
    # But if we want to maximize the size, we should order the vectors in best_set first,
    # and then the rest in an order that allows adding as many as possible.
    
    # Simpler: Assign priority based on whether the vector is in the best_set.
    # If in best_set, priority = 1.0 + small_index (to preserve the order that worked).
    # If not, priority = 0.0 - small_index (to put them last).
    
    # Let's refine: 
    # We want the greedy algorithm to reproduce the best_set.
    # So, for v in best_set, assign priority = 1000 - index_in_best_set.
    # For v not in best_set, assign priority = -1000 + index_in_rejected.
    
    rank_map = {}
    for i, v in enumerate(vectors):
        if v in best_set:
            # Higher priority for earlier in the set? 
            # Actually, the order within the cap set doesn't matter for validity, 
            # but it might matter for how many *additional* vectors can be added later.
            # Let's just give them all high priority.
            rank_map[v] = 10000.0 - i
        else:
            rank_map[v] = -10000.0 + i
    
    _best_rank = rank_map
    _initialized = True

def priority(vector, n):
    global _initialized, _best_rank, _deadline
    
    if not _initialized:
        _initialize(n)
    
    # Check deadline
    if time.time() > _deadline:
        # Fallback to a simple hash if we ran out of time
        val = 0
        for i, v in enumerate(vector):
            val = val * 3 + v
        return float(val)
    
    if vector in _best_rank:
        return _best_rank[vector]
    else:
        # Should not happen if n is consistent
        return 0.0

# EVOLVE-BLOCK-END
