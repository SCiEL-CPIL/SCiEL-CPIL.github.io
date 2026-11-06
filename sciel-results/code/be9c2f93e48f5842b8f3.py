"""Find largest graph with no regular induced subgraph on 6+ vertices.

Strategy:
1. Use a fast NumPy-vectorized validator (bitmask degrees + per-subset degree
   comparison) so that many candidate graphs can be checked quickly.
2. Start from known-good small graphs and grow via local search (edge flips)
   to reach n=20.
3. Spend remaining time searching for n=21 with many random restarts and
   local improvement, plus a few deterministic algebraic candidates.
"""
import time
import random
import numpy as np
import itertools

def _make_adj(n, edges):
    adj = np.zeros((n, n), dtype=np.uint8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    return adj

def _edges_from_adj(adj):
    n = adj.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append([i, j])
    return edges

def _is_valid_fast(adj, n, deadline_check=None):
    """Return True iff no induced subgraph on >=6 vertices is regular.
    
    Uses vectorized degree computation per subset via numpy.
    For each subset size k from 6 to n, check all combinations.
    A subset S is regular iff all vertices in S have the same degree within S.
    """
    if n < 6:
        return True
    # Precompute adjacency as int8 for fast matmul-like ops
    A = adj.astype(np.int8)
    vertices = np.arange(n)
    
    for k in range(6, n + 1):
        if deadline_check is not None and deadline_check():
            return None  # timeout, unknown
        for subset in itertools.combinations(range(n), k):
            idx = np.array(subset, dtype=np.intp)
            # Submatrix
            sub = A[np.ix_(idx, idx)]
            # Degrees within subset
            degs = sub.sum(axis=1)
            if np.all(degs == degs[0]):
                return False
    return True

def _is_valid_fast_with_timeout(adj, n, start_time, deadline):
    """Like _is_valid_fast but checks deadline periodically."""
    if n < 6:
        return True
    A = adj.astype(np.int8)
    
    for k in range(6, n + 1):
        if time.time() > deadline:
            return None
        count = 0
        for subset in itertools.combinations(range(n), k):
            count += 1
            if count % 5000 == 0 and time.time() > deadline:
                return None
            idx = np.array(subset, dtype=np.intp)
            sub = A[np.ix_(idx, idx)]
            degs = sub.sum(axis=1)
            if np.all(degs == degs[0]):
                return False
    return True

def _random_adj(n, rng):
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                adj[i, j] = 1
                adj[j, i] = 1
    return adj

def _local_search(adj, n, rng, max_steps=5000, deadline=None):
    """Try to fix a graph by flipping edges. Returns (adj, is_valid)."""
    A = adj.astype(np.int8)
    # Find a violating subset to guide flips
    def find_violation():
        for k in range(6, min(n + 1, 12)):  # only check up to size 12 for speed
            for _ in range(200):
                subset = tuple(rng.sample(range(n), k))
                idx = np.array(subset, dtype=np.intp)
                sub = A[np.ix_(idx, idx)]
                degs = sub.sum(axis=1)
                if np.all(degs == degs[0]):
                    return subset
            # Also check all size-6 subsets if n is small
            if n <= 15:
                for subset in itertools.combinations(range(n), 6):
                    idx = np.array(subset, dtype=np.intp)
                    sub = A[np.ix_(idx, idx)]
                    degs = sub.sum(axis=1)
                    if np.all(degs == degs[0]):
                        return subset
        return None
    
    current = adj.copy()
    for step in range(max_steps):
        if deadline and time.time() > deadline:
            break
        viol = find_violation()
        if viol is None:
            # No small violation found; do full check
            result = _is_valid_fast(current, n)
            if result is True:
                return current, True
            # If result is None (timeout), keep going
            if result is None:
                break
            # result is False but we didn't find it in small search; continue
            # Just flip a random edge
            i, j = rng.sample(range(n), 2)
            current[i, j] = 1 - current[i, j]
            current[j, i] = current[i, j]
            continue
        
        # Flip a random edge within the violating subset to break regularity
        if len(viol) >= 2:
            u, v = rng.sample(list(viol), 2)
            current[u, v] = 1 - current[u, v]
            current[v, u] = current[u, v]
        else:
            i, j = rng.sample(range(n), 2)
            current[i, j] = 1 - current[i, j]
            current[j, i] = current[i, j]
    
    # Final check
    result = _is_valid_fast(current, n)
    if result is True:
        return current, True
    return current, False

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    def timed_out():
        return elapsed() > DEADLINE - 5
    
    best_n = 5
    best_edges = []
    
    # Start with a trivially valid small graph (empty graph on 5 vertices)
    # Actually any graph on 5 vertices is valid since we need 6+ for violation
    best_adj = np.zeros((5, 5), dtype=np.uint8)
    best_edges = []
    
    # Try to build up to n=20 using local search from random starts
    for target_n in range(6, 21):
        if timed_out():
            break
        
        found = False
        max_attempts = 200 if target_n <= 12 else 100
        
        for attempt in range(max_attempts):
            if timed_out():
                break
            
            rng = random.Random(attempt * 7919 + target_n)
            adj = _random_adj(target_n, rng)
            
            # Quick check: is it already valid?
            result = _is_valid_fast_with_timeout(adj, target_n, start_time, DEADLINE - 5)
            if result is True:
                found = True
                best_n = target_n
                best_adj = adj
                best_edges = _edges_from_adj(adj)
                break
            elif result is None:
                # Timeout during check, skip
                continue
            
            # Try local search
            if not timed_out():
                adj2, valid = _local_search(adj, target_n, rng, max_steps=3000, deadline=DEADLINE - 5)
                if valid:
                    found = True
                    best_n = target_n
                    best_adj = adj2
                    best_edges = _edges_from_adj(adj2)
                    break
        
        if found:
            continue
        else:
            # Couldn't find valid graph for this n, try next (smaller might work)
            # Actually if we can't do target_n, we can't do larger, so break
            # But maybe we just didn't find it; try a bit more
            if target_n >= 14:
                break
    
    # If we reached n=20, try for n=21
    if best_n >= 20 and not timed_out():
        # Try many random restarts for n=21
        for attempt in range(2000):
            if timed_out():
                break
            
            rng = random.Random(attempt * 104729 + 999)
            adj = _random_adj(21, rng)
            
            # Quick partial check (only size 6 subsets for speed)
            A = adj.astype(np.int8)
            violation_found = False
            for subset in itertools.combinations(range(21), 6):
                idx = np.array(subset, dtype=np.intp)
                sub = A[np.ix_(idx, idx)]
                degs = sub.sum(axis=1)
                if np.all(degs == degs[0]):
                    violation_found = True
                    break
            
            if not violation_found:
                # Passed size-6 check, do full check
                result = _is_valid_fast_with_timeout(adj, 21, start_time, DEADLINE - 5)
                if result is True:
                    best_n = 21
                    best_adj = adj
                    best_edges = _edges_from_adj(adj)
                    break
                elif result is None:
                    continue
            
            # Try local search if time permits
            if not timed_out() and (attempt % 5 == 0):
                adj2, valid = _local_search(adj, 21, rng, max_steps=2000, deadline=DEADLINE - 5)
                if valid:
                    best_n = 21
                    best_adj = adj2
                    best_edges = _edges_from_adj(adj2)
                    break
    
    # If we didn't get 21, try a few more strategies for n=20 if we don't have it
    if best_n < 20 and not timed_out():
        for target_n in range(best_n + 1, 21):
            if timed_out():
                break
            for attempt in range(500):
                if timed_out():
                    break
                rng = random.Random(attempt * 31 + target_n * 1000)
                adj = _random_adj(target_n, rng)
                adj2, valid = _local_search(adj, target_n, rng, max_steps=5000, deadline=DEADLINE - 5)
                if valid:
                    best_n = target_n
                    best_adj = adj2
                    best_edges = _edges_from_adj(adj2)
                    break
            if best_n >= target_n:
                continue
            else:
                break
    
    if best_n < 6:
        # Fallback: return a valid 5-vertex graph
        return {'n': 5, 'edges': []}
    
    return {'n': best_n, 'edges': best_edges}
