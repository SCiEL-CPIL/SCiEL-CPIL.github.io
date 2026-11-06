import time
import random
import itertools
import numpy as np
import sys

# Global state
_start_time = None
_DEADLINE = 325.0
_best_set = {}
_vector_cache = {}

def _init_state(vector, n):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _is_deadline_exceeded():
    return _elapsed() > _DEADLINE

def _get_vectors(n):
    if n not in _vector_cache:
        _vector_cache[n] = list(itertools.product([0, 1, 2], repeat=n))
    return _vector_cache[n]

def _greedy_cap_set(order, n):
    """
    Greedily build cap set from ordered list of vectors.
    Returns the set of vectors in the cap set.
    """
    cap_set = set()
    forbidden = set()
    for v in order:
        if v not in forbidden:
            cap_set.add(v)
            for u in cap_set:
                if u == v: continue
                w = tuple((-v[i] - u[i]) % 3 for i in range(n))
                forbidden.add(w)
    return cap_set

def _random_linear_order(n, vectors):
    """
    Generate a random linear ordering based on dot product with a random vector.
    """
    w = tuple(random.randint(0, 2) for _ in range(n))
    items = []
    for v in vectors:
        score = sum(v[i] * w[i] for i in range(n)) % 3
        # Add noise to break symmetry and ensure unique-ish ordering
        noise = random.uniform(0, 1.0)
        items.append((score, noise, v))
    items.sort(key=lambda x: (x[0], x[1]))
    return [v for _, _, v in items]

def _balance_order(n, vectors):
    """
    Generate an ordering based on balance of 0s and 2s.
    Vectors with more balanced 0/2 counts are prioritized? 
    Actually, cap sets in F_3^n often relate to avoiding arithmetic progressions.
    In {0,1,2}, a line is (x, y, z) with x+z=2y mod 3.
    If we pick vectors with few 1s, we might avoid lines?
    Let's try: prioritize vectors with extreme balance (mostly 0s or mostly 2s) or specific patterns.
    """
    def score(v):
        c0 = sum(1 for x in v if x == 0)
        c1 = sum(1 for x in v if x == 1)
        c2 = sum(1 for x in v if x == 2)
        # Heuristic: prefer vectors with low c1 (few 1s) because 1 is the midpoint in 0,1,2?
        # If a line is (0, 1, 2), then 1 is the middle. Avoiding 1s might help.
        # Or prefer vectors where c0 == c2?
        # Let's try minimizing c1 first.
        return c1 + random.uniform(0, 0.1)
    
    items = [(score(v), v) for v in vectors]
    items.sort(key=lambda x: x[0])
    return [v for _, v in items]

def _local_search_improve(cap_set, n, vectors, time_budget):
    """
    Try to improve the cap set by swapping out elements.
    This is a simplified local search.
    """
    if not cap_set:
        return cap_set
        
    current_set = set(cap_set)
    current_size = len(current_set)
    forbidden = set()
    for v in current_set:
        for u in current_set:
            if u == v: continue
            w = tuple((-v[i] - u[i]) % 3 for i in range(n))
            forbidden.add(w)
            
    start = time.time()
    improved = True
    while improved and time.time() - start < time_budget:
        improved = False
        # Try to add any vector not in set and not forbidden
        # If none, try swapping
        candidates = [v for v in vectors if v not in current_set and v not in forbidden]
        
        if candidates:
            # Pick a random candidate
            v_add = random.choice(candidates)
            # Check if adding v_add is valid (it is by construction, not in forbidden)
            # But we need to check if it creates new conflicts with existing set?
            # The 'forbidden' set contains all w such that (v, u, w) is a line for v,u in set.
            # So if v_add is not in forbidden, it doesn't form a line with any pair in current_set.
            # So we can just add it.
            current_set.add(v_add)
            # Update forbidden
            for u in list(current_set):
                if u == v_add: continue
                w = tuple((-v_add[i] - u[i]) % 3 for i in range(n))
                forbidden.add(w)
            current_size += 1
            improved = True
        else:
            # No direct additions. Try swaps.
            # This is expensive. For now, just break or do a simple random restart inside.
            # Let's try to remove a random vector and see if we can add two?
            # Too complex for now. Break.
            break
            
    return current_set

def _heuristic_search(n, time_budget):
    """
    Run multiple greedy iterations with different orderings and local search.
    """
    vectors = _get_vectors(n)
    best_set = set()
    best_size = 0
    
    start = time.time()
    
    while time.time() - start < time_budget:
        if _is_deadline_exceeded():
            break
            
        # Mix of strategies
        r = random.random()
        if r < 0.5:
            order = _random_linear_order(n, vectors)
        else:
            order = _balance_order(n, vectors)
            
        cap_set = _greedy_cap_set(order, n)
        
        # Local search improvement
        # Allocate a small fraction of time for local search per iteration
        remaining = time_budget - (time.time() - start)
        if remaining > 1.0:
            ls_time = min(0.5, remaining * 0.1)
            cap_set = _local_search_improve(cap_set, n, vectors, ls_time)
            
        if len(cap_set) > best_size:
            best_size = len(cap_set)
            best_set = cap_set
            
    return best_set

def priority(vector, n):
    global _start_time
    _init_state(vector, n)
    
    if _is_deadline_exceeded():
        # Fallback heuristic
        c0 = sum(1 for x in vector if x == 0)
        c2 = sum(1 for x in vector if x == 2)
        c1 = sum(1 for x in vector if x == 1)
        # Prefer low c1
        return -c1
        
    if n not in _best_set:
        cap_set = None
        
        # For small n, we could try ILP, but let's stick to heuristics for speed and simplicity
        # unless n is very small.
        if n <= 5:
             # For n<=5, the search space is small enough that random restarts are very effective
             pass
             
        # Use heuristic search
        # Allocate 250 seconds for search
        cap_set = _heuristic_search(n, 250.0)
        
        _best_set[n] = cap_set
    
    # Assign priority
    # Vectors in best_set get 1.0
    # Others get 0.0
    if vector in _best_set[n]:
        return 1.0
    else:
        return 0.0
