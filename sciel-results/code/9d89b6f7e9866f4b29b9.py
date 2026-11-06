"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

# Global cache: n -> list of vectors in optimal greedy order
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

def _greedy_size(order, n):
    """Given an ordering of all 3^n vectors, return the cap set size from greedy."""
    added_set = set()
    count = 0
    for v in order:
        # Check if adding v keeps cap set
        # v conflicts with a,b in added_set if a+b+v = 0 mod 3 for some a!=b!=v
        conflict = False
        for a in added_set:
            b = tuple((-a[i] - v[i]) % 3 for i in range(n))
            if b in added_set and b != a and b != v:
                conflict = True
                break
        if not conflict:
            added_set.add(v)
            count += 1
    return count

def _find_best_order(n, time_limit=250):
    """Find a good ordering for greedy cap set construction."""
    if n in _cache:
        return _cache[n]
    
    all_vecs = _generate_all_vectors(n)
    total = len(all_vecs)
    
    best_order = None
    best_size = 0
    
    # 1. Weight-based ordering (sum of coordinates)
    def weight(v):
        return sum(v)
    
    # Initial candidate: sorted by weight descending
    initial_order = sorted(all_vecs, key=weight, reverse=True)
    best_size = _greedy_size(initial_order, n)
    best_order = initial_order[:]
    
    if _elapsed() > time_limit - 50:
        _cache[n] = best_order
        return best_order

    # Local Search / Hill Climbing
    rng = random.Random(42)
    current_order = best_order[:]
    current_size = best_size
    
    steps = 0
    max_steps = 10000 # Adjust based on time
    
    while _elapsed() < time_limit - 50:
        # Pick two random indices
        i = rng.randint(0, total - 1)
        j = rng.randint(0, total - 1)
        if i == j:
            continue
            
        # Swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        # Evaluate
        new_size = _greedy_size(current_order, n)
        
        if new_size > current_size:
            current_size = new_size
            if new_size > best_size:
                best_size = new_size
                best_order = current_order[:]
        else:
            # Revert
            current_order[i], current_order[j] = current_order[j], current_order[i]
            
        steps += 1
        # Optional: Random restarts if stuck? 
        # For now, simple hill climbing.
        
    _cache[n] = best_order
    return best_order

def priority(vector, n):
    """Return priority for a vector. Higher = added earlier in greedy."""
    if n not in _cache:
        _find_best_order(n, time_limit=250)
    
    order = _cache[n]
    
    if not hasattr(priority, '_index_cache'):
        priority._index_cache = {}
    
    if n not in priority._index_cache:
        idx = {v: i for i, v in enumerate(order)}
        priority._index_cache[n] = idx
    
    idx_map = priority._index_cache[n]
    # Higher priority = earlier in order
    return float(len(order) - idx_map[vector])
# EVOLVE-BLOCK-END
