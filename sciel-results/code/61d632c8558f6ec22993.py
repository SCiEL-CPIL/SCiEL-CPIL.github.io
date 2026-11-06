import numpy as np
import time
import random

N = 16

_INPUTS_CACHE = None

def _get_inputs():
    global _INPUTS_CACHE
    if _INPUTS_CACHE is None:
        inputs = np.zeros((1 << N, N), dtype=np.uint8)
        for i in range(N):
            col = (np.arange(1 << N) >> i) & 1
            inputs[:, i] = col
        _INPUTS_CACHE = inputs
    return _INPUTS_CACHE

def _verify_network(comparators, inputs):
    wires = inputs.copy()
    for i, j in comparators:
        min_val = np.minimum(wires[:, i], wires[:, j])
        max_val = np.maximum(wires[:, i], wires[:, j])
        wires[:, i] = min_val
        wires[:, j] = max_val
    
    diffs = np.diff(wires, axis=1)
    return not np.any(diffs < 0)

def _batcher(n):
    if n == 1:
        return []
    
    def _merge(k):
        if k == 1:
            return [(0, 1)]
        
        k_half = k // 2
        
        even_comps = []
        for i, j in _merge(k_half):
            mi = 2 * i if i < k_half else 2 * (i - k_half) + k
            mj = 2 * j if j < k_half else 2 * (j - k_half) + k
            even_comps.append((mi, mj))
            
        odd_comps = []
        for i, j in _merge(k_half):
            mi = 2 * i + 1 if i < k_half else 2 * (i - k_half) + k + 1
            mj = 2 * j + 1 if j < k_half else 2 * (j - k_half) + k + 1
            odd_comps.append((mi, mj))
            
        final_comps = [(2 * i + 1, 2 * i + 2) for i in range(k - 1)]
        
        return even_comps + odd_comps + final_comps

    k = n // 2
    left = _batcher(k)
    right = [(i + k, j + k) for i, j in _batcher(k)]
    merge = _merge(k)
    
    return left + right + merge

def _local_search(initial_comparators, time_limit=85):
    start_time = time.time()
    current = list(initial_comparators)
    best = list(current)
    inputs = _get_inputs()
    
    if not _verify_network(current, inputs):
        return None

    def try_remove_all():
        nonlocal current, best
        changed = True
        while changed and time.time() - start_time < time_limit:
            changed = False
            for idx in range(len(current) - 1, -1, -1):
                if time.time() - start_time > time_limit:
                    break
                candidate = current[:idx] + current[idx+1:]
                if _verify_network(candidate, inputs):
                    current = candidate
                    best = list(current)
                    changed = True
                    break

    # Initial aggressive removal
    try_remove_all()
    
    if len(best) == 60:
        return best

    # Local search
    moves = []
    
    def move_remove(comps):
        if len(comps) == 0: return None
        idx = random.randint(0, len(comps) - 1)
        return comps[:idx] + comps[idx+1:]

    def move_swap(comps):
        if len(comps) < 2: return None
        i, j = random.sample(range(len(comps)), 2)
        new_comps = comps.copy()
        new_comps[i], new_comps[j] = new_comps[j], new_comps[i]
        return new_comps

    def move_shift(comps):
        if len(comps) < 2: return None
        i = random.randint(0, len(comps) - 1)
        j = random.randint(0, len(comps) - 1)
        if i == j: return None
        new_comps = comps.copy()
        comp = new_comps.pop(i)
        new_comps.insert(j, comp)
        return new_comps

    def move_add(comps):
        i, j = random.sample(range(N), 2)
        if i > j: i, j = j, i
        new_comps = comps.copy()
        pos = random.randint(0, len(new_comps))
        new_comps.insert(pos, (i, j))
        return new_comps

    moves = [move_remove, move_swap, move_shift]
    # Add 'move_add' only if stuck for a while?
    # For now, keep it simple.
    
    last_improvement = time.time()
    stuck_counter = 0
    
    while time.time() - start_time < time_limit:
        move_func = random.choice(moves)
        candidate = move_func(list(current))
        if candidate is None:
            continue
        
        if _verify_network(candidate, inputs):
            current = candidate
            if len(current) < len(best):
                best = list(current)
                last_improvement = time.time()
                stuck_counter = 0
                if len(best) == 60:
                    break
            else:
                # Valid but not better? 
                # If we are doing random moves, we might wander.
                # We should bias towards removals.
                pass
        else:
            stuck_counter += 1
            
        # If stuck for too long, try a random restart or add a comparator
        if stuck_counter > 1000:
            # Try adding a comparator to escape local minimum
            candidate = move_add(list(current))
            if _verify_network(candidate, inputs):
                current = candidate
                stuck_counter = 0
            else:
                # Random restart from best?
                # current = list(best)
                stuck_counter = 0

    # Final polish
    try_remove_all()
    
    return best

def construct():
    network = _batcher(N)
    optimized = _local_search(network, time_limit=85)
    if optimized is None:
        return network
    return optimized
