"""Rewritten solver: randomized greedy + local search for small maximal Sidon sets in [1,256]."""
import time
import random

N = 256
DEADLINE = 58.0  # seconds wall-clock budget (safety margin before 60s kill)

# EVOLVE-BLOCK-START

def _is_sidon(selected):
    """Check if selected is a Sidon set (all unordered pair sums distinct)."""
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, N):
    """Check if selected is inclusion-maximal: adding any missing element breaks Sidon property."""
    sel_set = set(selected)
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            sums.add(a + b)
    
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if adding x would create a duplicate sum
        for a in selected:
            s = a + x
            if s in sums:
                return True  # Adding x would break Sidon, so it's blocked
        if 2 * x in sums:
            return True
    return True

def _greedy_randomized(seed, max_time=None):
    """Randomized greedy: pick elements in random order, add if Sidon property holds."""
    random.seed(seed)
    elements = list(range(1, N + 1))
    random.shuffle(elements)
    
    selected = []
    sums = set()
    start = time.time()
    
    for x in elements:
        if max_time is not None and time.time() - start > max_time:
            break
        # Check if x can be added
        added_sums = [x + a for a in selected] + [2 * x]
        if len(set(added_sums)) == len(added_sums) and not sums.intersection(added_sums):
            selected.append(x)
            sums.update(added_sums)
    
    return selected

def _greedy_ascending():
    """Standard ascending greedy."""
    selected = []
    sums = set()
    for x in range(1, N + 1):
        added_sums = [x + a for a in selected] + [2 * x]
        if len(set(added_sums)) == len(added_sums) and not sums.intersection(added_sums):
            selected.append(x)
            sums.update(added_sums)
    return selected

def _try_remove(selected):
    """Try to remove elements from selected while maintaining maximality. Returns smaller set if possible."""
    selected = sorted(selected)
    improved = True
    while improved:
        improved = False
        for i in range(len(selected)):
            candidate = selected[:i] + selected[i+1:]
            if not candidate:
                continue
            if _is_sidon(candidate) and _is_maximal(candidate, N):
                selected = candidate
                improved = True
                break
    return selected

def _local_search(selected, max_iterations=100):
    """Local search: try swapping elements to reduce size while maintaining maximality."""
    selected = sorted(selected)
    best = selected[:]
    start = time.time()
    
    for _ in range(max_iterations):
        if time.time() - start > 2.0:
            break
        # Try removing one element
        for i in range(len(selected)):
            candidate = selected[:i] + selected[i+1:]
            if _is_sidon(candidate) and _is_maximal(candidate, N):
                selected = candidate
                break
        else:
            break
        # Try adding then removing (2-opt style)
        added = False
        for x in range(1, N + 1):
            if x in selected:
                continue
            candidate = sorted(selected + [x])
            if _is_sidon(candidate):
                # Try removing a different element
                for i in range(len(candidate)):
                    if candidate[i] == x:
                        continue
                    sub = candidate[:i] + candidate[i+1:]
                    if len(sub) < len(selected) and _is_sidon(sub) and _is_maximal(sub, N):
                        selected = sub
                        added = True
                        break
                if added:
                    break
        if not added:
            break
        if len(selected) < len(best):
            best = selected[:]
    
    return best

def construct():
    """Find a small inclusion-maximal Sidon set in [1,256]."""
    start = time.time()
    
    # Start with ascending greedy
    best = _greedy_ascending()
    
    # Try multiple randomized greedy runs
    best_size = len(best)
    seed = 0
    while time.time() - start < DEADLINE - 1.0:
        seed += 1
        candidate = _greedy_randomized(seed)
        if len(candidate) < best_size:
            best = candidate
            best_size = len(candidate)
            # Try to improve via local search
            if time.time() - start < DEADLINE - 3.0:
                improved = _local_search(best)
                if len(improved) < best_size:
                    best = improved
                    best_size = len(improved)
    
    # Final attempt: try to reduce best further
    if time.time() - start < DEADLINE - 2.0:
        improved = _local_search(best)
        if len(improved) < len(best):
            best = improved
    
    return best

# EVOLVE-BLOCK-END
