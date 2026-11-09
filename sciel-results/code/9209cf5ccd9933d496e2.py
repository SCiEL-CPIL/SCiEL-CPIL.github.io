"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 256

# EVOLVE-BLOCK-START
def _is_sidon(selected):
    """Check if selected is a valid Sidon set (all unordered pair sums distinct)."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            b = selected[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, N):
    """Check if selected is inclusion-maximal: no missing element can be added."""
    selected_set = set(selected)
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            b = selected[j]
            sums.add(a + b)
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        # Check if adding x preserves Sidon property
        ok = True
        for a in selected:
            s = x + a
            if s in sums and s != 2 * x:
                ok = False
                break
            if s == 2 * x:
                # diagonal sum 2x must not equal any existing sum
                if s in sums:
                    ok = False
                    break
        if ok:
            # also check 2*x against existing sums
            if 2 * x in sums:
                ok = False
        if ok:
            return False
    return True

def _greedy_saturation(order, N):
    """Greedy saturation: process elements in given order, add if Sidon preserved."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _try_reduce(selected, N, rng):
    """Try to remove elements while maintaining maximality (inclusion-maximal)."""
    if not _is_maximal(selected, N):
        return selected
    current = list(selected)
    improved = True
    while improved:
        improved = False
        # Try removing each element in random order
        indices = list(range(len(current)))
        rng.shuffle(indices)
        for idx in indices:
            candidate = current[:idx] + current[idx+1:]
            if not candidate:
                continue
            if _is_sidon(candidate) and _is_maximal(candidate, N):
                current = candidate
                improved = True
                break
    return current

def _local_search(selected, N, rng, max_iters=200):
    """Local search: try swapping elements to reduce size while keeping maximality."""
    current = list(selected)
    for _ in range(max_iters):
        if len(current) < 2:
            break
        # Try removing a random element
        idx = rng.randrange(len(current))
        candidate = current[:idx] + current[idx+1:]
        if candidate and _is_sidon(candidate) and _is_maximal(candidate, N):
            current = candidate
        else:
            # Try swapping: remove one, add another
            idx_remove = rng.randrange(len(current))
            missing = [x for x in range(1, N+1) if x not in set(current)]
            if not missing:
                break
            idx_add = rng.randrange(len(missing))
            new_elem = missing[idx_add]
            candidate = [x for x in current if x != current[idx_remove]] + [new_elem]
            if _is_sidon(candidate) and _is_maximal(candidate, N):
                current = candidate
    return current

def construct():
    deadline = time.time() + 55.0  # 55s safety margin within 60s limit
    best = None
    best_size = N + 1
    
    # Initial: ascending greedy
    asc_order = list(range(1, N + 1))
    result = _greedy_saturation(asc_order, N)
    result = _try_reduce(result, N, random.Random(42))
    if len(result) < best_size:
        best = result
        best_size = len(result)
    
    # Randomized greedy with multiple orderings
    rng = random.Random(12345)
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Generate a random ordering
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        result = _greedy_saturation(order, N)
        
        # Try local reduction
        result = _try_reduce(result, N, rng)
        
        if len(result) < best_size:
            best = result
            best_size = len(result)
        
        # Occasionally try local search from current best
        if iterations % 5 == 0 and best is not None:
            polished = _local_search(list(best), N, rng, max_iters=50)
            if len(polished) < best_size:
                best = polished
                best_size = len(polished)
    
    # Final polish on best
    if best is not None and time.time() < deadline - 5.0:
        polished = _try_reduce(list(best), N, rng)
        if len(polished) < best_size:
            best = polished
            best_size = len(polished)
    
    if best is None:
        # Fallback: ascending greedy
        return _greedy_saturation(asc_order, N)
    
    return best
# EVOLVE-BLOCK-END
