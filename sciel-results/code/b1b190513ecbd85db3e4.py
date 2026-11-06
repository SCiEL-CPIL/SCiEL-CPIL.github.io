import time
import random
from functools import lru_cache

# Global cache: n -> list of vectors in optimal greedy order
_cache = {}
_start_time = None
_DEADLINE = 310  # Safety margin from 330s kill

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _time_left():
    return _DEADLINE - _elapsed()

def _greedy_size(order, n):
    """Given an ordering of all 3^n vectors, return the cap set size from greedy."""
    added_set = set()
    count = 0
    for v in order:
        # Check if adding v keeps cap set
        # v conflicts with a,b in added_set if a+b+v = 0 mod 3 for some a!=b!=v
        conflict = False
        for a in added_set:
            # Calculate b such that a + b + v = 0 mod 3
            # b = -a - v mod 3
            b = tuple((-a[i] - v[i]) % 3 for i in range(n))
            if b in added_set and b != a and b != v:
                conflict = True
                break
        if not conflict:
            added_set.add(v)
            count += 1
    return count

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

def _initial_order_weight_desc(all_vecs):
    return sorted(all_vecs, key=lambda v: sum(v), reverse=True)

def _initial_order_weight_asc(all_vecs):
    return sorted(all_vecs, key=lambda v: sum(v))

def _initial_order_lex(all_vecs):
    return sorted(all_vecs)

def _initial_order_greedy_construct(all_vecs, n, seed=123):
    """
    Construct an order by first building a cap set greedily from a random shuffle,
    then appending the remaining vectors.
    This tends to put 'good' vectors at the front.
    """
    rng = random.Random(seed)
    shuffled = all_vecs[:]
    rng.shuffle(shuffled)
    
    added_set = set()
    added_order = []
    for v in shuffled:
        conflict = False
        for a in added_set:
            b = tuple((-a[i] - v[i]) % 3 for i in range(n))
            if b in added_set and b != a and b != v:
                conflict = True
                break
        if not conflict:
            added_set.add(v)
            added_order.append(v)
            
    # Append the rest (those not in added_set)
    remaining = [v for v in shuffled if v not in added_set]
    # Shuffle remaining to break ties randomly
    rng.shuffle(remaining)
    return added_order + remaining

def _find_best_order(n, time_limit=250):
    """Find a good ordering for greedy cap set construction."""
    if n in _cache:
        return _cache[n]
    
    all_vecs = _generate_all_vectors(n)
    total = len(all_vecs)
    
    best_order = None
    best_size = 0
    
    # 1. Generate diverse initial candidates
    candidates = [
        _initial_order_weight_desc(all_vecs),
        _initial_order_weight_asc(all_vecs),
        _initial_order_lex(all_vecs),
        _initial_order_greedy_construct(all_vecs, n, seed=42),
        _initial_order_greedy_construct(all_vecs, n, seed=43),
        _initial_order_greedy_construct(all_vecs, n, seed=44)
    ]
    
    # Add a few random shuffles
    rng_init = random.Random(100)
    for _ in range(5):
        shuffled = all_vecs[:]
        rng_init.shuffle(shuffled)
        candidates.append(shuffled)
        
    # Evaluate candidates to find the best starting point
    for order in candidates:
        if _time_left() < 30:
            break
        sz = _greedy_size(order, n)
        if sz > best_size:
            best_size = sz
            best_order = order[:]
            
    if best_order is None:
        best_order = all_vecs[:]
        best_size = _greedy_size(best_order, n)

    # 2. Hill Climbing with Random Swaps
    # We use a random swap strategy. To avoid getting stuck in local minima,
    # we can implement a simple restart mechanism or accept slightly worse moves 
    # with low probability (Simulated Annealing style), but pure hill climbing 
    # with restarts is often more robust for these discrete problems.
    
    current_order = best_order[:]
    current_size = best_size
    
    rng = random.Random(12345)
    no_improve_steps = 0
    max_no_improve = 5000  # Restart if stuck for this many steps
    
    while _time_left() > 10:
        # Pick two random indices to swap
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
            no_improve_steps = 0
            if new_size > best_size:
                best_size = new_size
                best_order = current_order[:]
        else:
            # Revert if not improved
            current_order[i], current_order[j] = current_order[j], current_order[i]
            no_improve_steps += 1
            
            # If stuck, restart from best_order with a new random perturbation
            if no_improve_steps >= max_no_improve:
                # Reset to best and apply a larger perturbation (e.g., swap 3-5 items)
                current_order = best_order[:]
                current_size = best_size
                no_improve_steps = 0
                # Apply a few random swaps to kickstart
                for _ in range(5):
                    i = rng.randint(0, total - 1)
                    j = rng.randint(0, total - 1)
                    if i != j:
                        current_order[i], current_order[j] = current_order[j], current_order[i]
                # Re-evaluate current state after perturbation
                current_size = _greedy_size(current_order, n)

    _cache[n] = best_order
    return best_order

def priority(vector, n):
    """Return priority for a vector. Higher = added earlier in greedy."""
    if n not in _cache:
        _find_best_order(n, time_limit=250)
    
    order = _cache[n]
    # Find the index of this vector in the order
    # Use a dict for O(1) lookup
    if not hasattr(priority, '_index_cache'):
        priority._index_cache = {}
    
    if n not in priority._index_cache:
        idx = {v: i for i, v in enumerate(order)}
        priority._index_cache[n] = idx
    
    idx_map = priority._index_cache[n]
    # Higher priority = earlier in order
    return float(len(order) - idx_map[vector])
