"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 256

# EVOLVE-BLOCK-START
def _check_sidon(s):
    """Check if set s is a Sidon set (all pairwise sums distinct)."""
    sums = set()
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            val = s[i] + s[j]
            if val in sums:
                return False
            sums.add(val)
    return True

def _is_maximal(s, N):
    """Check if Sidon set s is inclusion-maximal in [1,N]."""
    s_set = set(s)
    sums = set()
    for a in s:
        for b in s:
            if a <= b:
                sums.add(a + b)
    
    for x in range(1, N + 1):
        if x in s_set:
            continue
        # Check if adding x would create a duplicate sum
        conflict = False
        for a in s:
            val = x + a
            if val in sums:
                conflict = True
                break
        if not conflict:
            # Also check 2*x
            if 2 * x in sums:
                conflict = True
        if not conflict:
            return False
    return True

def _build_greedy_sidon(order, N):
    """Build a Sidon set by processing elements in given order."""
    selected = []
    sums = set()
    for x in order:
        new_sums = [x + a for a in selected] + [2 * x]
        if len(set(new_sums)) == len(new_sums) and not sums.intersection(new_sums):
            selected.append(x)
            sums.update(new_sums)
    return selected

def _try_remove_and_check(s, N):
    """Try to remove elements from s while maintaining maximality. Returns minimized set."""
    s = list(s)
    changed = True
    while changed:
        changed = False
        for i in range(len(s) - 1, -1, -1):
            candidate = s[:i] + s[i+1:]
            if not _check_sidon(candidate):
                continue
            if _is_maximal(candidate, N):
                s = candidate
                changed = True
                break
    return s

def construct():
    best_size = N + 1
    best_set = None
    start_time = time.time()
    time_limit = 28.0
    
    random.seed(42)
    
    # Multiple restarts with different random orderings
    for attempt in range(500):
        if time.time() - start_time > time_limit:
            break
        
        # Create a random permutation for greedy construction
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        # Build initial Sidon set
        s = _build_greedy_sidon(order, N)
        
        if len(s) >= best_size:
            continue
        
        # Try to minimize by removing elements
        s = _try_remove_and_check(s, N)
        
        if len(s) < best_size:
            best_size = len(s)
            best_set = s[:]
        
        # Try local search: swap elements to potentially reduce size
        if best_set is not None and time.time() - start_time < time_limit - 1.0:
            for _ in range(20):
                if time.time() - start_time > time_limit - 0.5:
                    break
                if len(best_set) == 0:
                    break
                # Try removing a random element
                idx = random.randint(0, len(best_set) - 1)
                candidate = best_set[:idx] + best_set[idx+1:]
                if _check_sidon(candidate) and _is_maximal(candidate, N):
                    best_set = candidate
                    best_size = len(best_set)
                else:
                    # Try swapping: remove one, add another
                    removed = best_set[idx]
                    for x in range(1, N + 1):
                        if x in best_set or x == removed:
                            continue
                        candidate = best_set[:idx] + [x] + best_set[idx+1:]
                        if _check_sidon(candidate) and _is_maximal(candidate, N):
                            if len(candidate) < len(best_set):
                                best_set = candidate
                                best_size = len(best_set)
                            break
    
    if best_set is None:
        # Fallback: use deterministic greedy
        return _build_greedy_sidon(list(range(1, N + 1)), N)
    
    return best_set
# EVOLVE-BLOCK-END
