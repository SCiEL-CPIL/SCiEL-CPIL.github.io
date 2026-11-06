"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
import time
import random

def _compute_sums(selected):
    """Compute all pairwise sums a+b for a<=b in selected."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        sums.add(2 * a)
        for j in range(i + 1, n):
            sums.add(a + selected[j])
    return sums

def _is_sidon(selected):
    """Check if selected is a Sidon set."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        if 2 * a in sums:
            return False
        sums.add(2 * a)
        for j in range(i + 1, n):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected):
    """Check if selected is inclusion-maximal Sidon in [1,N]."""
    selected_set = set(selected)
    sums = _compute_sums(selected)
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        # Adding x creates a duplicate iff 2x or x+s is already in sums
        if 2 * x in sums:
            continue
        conflict = False
        for s in selected:
            if x + s in sums:
                conflict = True
                break
        if not conflict:
            return False
    return True

def _greedy_construct(order):
    """Greedy Sidon set construction following the given order."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _local_search_reduce(selected, deadline):
    """Try to remove elements while maintaining maximality."""
    current = list(selected)
    current_set = set(current)
    
    # Try removing elements one at a time
    for x in list(current):
        if time.time() > deadline:
            break
        new_set = current_set - {x}
        new_list = [y for y in current if y != x]
        
        if not _is_sidon(new_list):
            continue
        if _is_maximal(new_list):
            current = new_list
            current_set = new_set
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    best = None
    best_size = N + 1
    
    # Strategy 1: Ascending greedy (baseline)
    order = list(range(1, N + 1))
    selected = _greedy_construct(order)
    if len(selected) < best_size:
        best = selected
        best_size = len(selected)
    
    # Local search to reduce size
    if time.time() < deadline:
        reduced = _local_search_reduce(best, deadline)
        if len(reduced) < best_size:
            best = reduced
            best_size = len(reduced)
    
    # Strategy 2: Try different random orders
    if time.time() < deadline - 5.0:
        for trial in range(50):
            if time.time() > deadline:
                break
            random.shuffle(order)
            selected = _greedy_construct(order)
            if len(selected) < best_size:
                best = selected
                best_size = len(selected)
                # Try local search on this better result
                if time.time() < deadline - 2.0:
                    reduced = _local_search_reduce(best, deadline)
                    if len(reduced) < best_size:
                        best = reduced
                        best_size = len(reduced)
    
    return best
# EVOLVE-BLOCK-END
