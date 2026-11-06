import time
import random
import math
from functools import lru_cache

# Global cache: n -> (best_scores_dict, best_size)
_cache = {}
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _generate_all_vectors(n):
    """Generate all 3^n vectors."""
    if n == 0:
        return [()]
    result = []
    def _gen(prefix, depth):
        if depth == n:
            result.append(tuple(prefix))
            return
        for v in range(3):
            prefix.append(v)
            _gen(prefix, depth+1)
            prefix.pop()
    _gen([], 0)
    return result

def _greedy_size_with_scores(scores, n):
    """
    Given a dict of vector -> score, return the cap set size from greedy.
    """
    # Get all vectors
    # We need to sort by score descending.
    # To avoid regenerating vectors every time, we can pass them in or generate once.
    # For efficiency in the search loop, we should pre-sort indices or use a list of (score, vector).
    
    # This function is called frequently. We need it to be fast.
    # Let's assume 'scores' is a list of (vector, score) or we pass the vector list.
    pass

def _run_greedy(order, n):
    """
    Given a list of vectors in priority order, return the cap set size.
    """
    added_set = set()
    for v in order:
        conflict = False
        # Check if v conflicts with any pair in added_set
        # a + b + v = 0 mod 3 => b = -a - v
        for a in added_set:
            b = tuple((-a[i] - v[i]) % 3 for i in range(n))
            if b in added_set:
                # b is in set. We need to ensure distinctness a, b, v
                # If b == a, then 2a + v = 0. If b == v, then a + 2v = 0.
                # The cap set definition usually requires distinct a,b,c.
                # "a set with no 3-term arithmetic progression" implies distinct elements.
                if b != a and b != v:
                    conflict = True
                    break
        if not conflict:
            added_set.add(v)
    return len(added_set)

def _find_best_priorities(n, time_limit=280):
    """
    Find optimal priority scores using Simulated Annealing.
    """
    if n in _cache:
        return _cache[n]
    
    all_vecs = _generate_all_vectors(n)
    total = len(all_vecs)
    
    # Initialize scores
    # Heuristic: vectors with more 1s might be "better" or "worse"?
    # Let's start with random scores.
    rng = random.Random(42)
    scores = {v: rng.uniform(0, 1) for v in all_vecs}
    
    # Initial evaluation
    order = sorted(all_vecs, key=lambda v: scores[v], reverse=True)
    current_size = _run_greedy(order, n)
    best_size = current_size
    best_scores = scores.copy()
    
    # Simulated Annealing
    # T_start, T_end, cooling rate
    # We don't know how many iterations we can do.
    # Let's estimate cost of one iteration:
    # Sort: O(N log N)
    # Greedy: O(N * K)
    # For n=8, N=6561, K~700. ~4.5M ops.
    # Python can do ~10-50M simple ops/sec.
    # So ~0.1-0.5s per iteration?
    # 280s / 0.5s = 560 iterations.
    # This is too few for SA to be very effective from scratch.
    
    # We need a faster evaluation or a smarter search.
    
    # Alternative: Local Search on the order directly, but with better moves.
    # Or, use the fact that we only need the *relative* order.
    
    # Let's try a hybrid:
    # 1. Get a good initial solution using a heuristic (e.g., greedy with random restarts on small subsets? No, too complex).
    # 2. Use Hill Climbing on the scores.
    
    # Let's stick to optimizing the scores but use a faster heuristic for initial guess.
    # A known good heuristic for cap sets: prioritize vectors with high "weight" (sum of coords) or specific patterns.
    # Actually, the "weight" sort in the previous program was a decent start.
    
    # Let's re-initialize with weight-based scores + noise.
    def weight(v):
        return sum(v)
        
    # Sort by weight desc, then add small noise to break ties
    for v in all_vecs:
        scores[v] = weight(v) + rng.uniform(0, 0.1)
        
    order = sorted(all_vecs, key=lambda v: scores[v], reverse=True)
    current_size = _run_greedy(order, n)
    best_size = current_size
    best_scores = scores.copy()
    
    # Local Search / Hill Climbing on scores
    # Move: Pick a random vector, change its score slightly.
    # This is slow because it requires re-sorting and re-greedy.
    
    # Better Move: Swap the priority of two vectors.
    # This is equivalent to swapping their positions in the sorted list.
    # But we are optimizing scores, so "swapping scores" is a discrete move.
    
    # Let's try: Pick two random vectors u, v.
    # Swap their scores.
    # This changes the order.
    # Evaluate new size.
    
    start_time = time.time()
    max_time = time_limit
    
    while time.time() - start_time < max_time:
        # Pick two random vectors
        u = rng.choice(all_vecs)
        v = rng.choice(all_vecs)
        if u == v:
            continue
            
        # Swap scores
        scores[u], scores[v] = scores[v], scores[u]
        
        # Re-sort and evaluate
        # Optimization: If we only swap two elements, we can update the sorted list locally?
        # No, their positions could change significantly.
        # Just re-sort.
        order = sorted(all_vecs, key=lambda v: scores[v], reverse=True)
        new_size = _run_greedy(order, n)
        
        if new_size > current_size:
            current_size = new_size
            if new_size > best_size:
                best_size = new_size
                best_scores = scores.copy()
        else:
            # Revert
            scores[u], scores[v] = scores[v], scores[u]
            
        # Check time
        if time.time() - start_time > max_time:
            break
            
    _cache[n] = best_scores
    return best_scores

def priority(vector, n):
    """Return priority for a vector. Higher = added earlier in greedy."""
    if n not in _cache:
        _find_best_priorities(n, time_limit=280)
    
    scores = _cache[n]
    return float(scores[vector])
