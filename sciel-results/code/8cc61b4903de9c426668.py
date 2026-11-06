import random
import time

N = 256

def _can_add_fast(sel, sums, x):
    """Check if x can be added to sel given current sums."""
    for a in sel:
        if (x + a) in sums:
            return False
    if (2 * x) in sums:
        return False
    return True

def _add_element(sel, sums, x):
    """Add x to sel and update sums in-place."""
    sel.append(x)
    for a in sel[:-1]:
        sums.add(x + a)
    sums.add(2 * x)

def _greedy_random(order, sel, sums, deadline):
    for x in order:
        if time.time() > deadline:
            break
        if _can_add_fast(sel, sums, x):
            _add_element(sel, sums, x)

def _saturate(sel, sums, deadline):
    """Add all addable elements to make the set maximal."""
    in_sel = set(sel)
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
        for x in range(1, N + 1):
            if x in in_sel:
                continue
            if _can_add_fast(sel, sums, x):
                _add_element(sel, sums, x)
                in_sel.add(x)
                changed = True
    return sel

def _local_search_reduce(sel, sums, deadline, max_iters=50):
    """Try to remove elements and re-saturate to get smaller maximal set."""
    best = list(sel)
    best_sums = set(sums)
    best_size = len(best)
    in_best = set(best)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        # Try removing each element
        for i in range(len(best)):
            if time.time() > deadline:
                break
            removed = best[i]
            candidate = best[:i] + best[i+1:]
            cand_sums = set()
            for j in range(len(candidate)):
                a = candidate[j]
                for k in range(j, len(candidate)):
                    cand_sums.add(a + candidate[k])
            
            in_cand = set(candidate)
            # Saturate
            changed = True
            while changed:
                changed = False
                if time.time() > deadline:
                    break
                for x in range(1, N + 1):
                    if x in in_cand:
                        continue
                    if _can_add_fast(candidate, cand_sums, x):
                        _add_element(candidate, cand_sums, x)
                        in_cand.add(x)
                        changed = True
            
            if len(candidate) < best_size:
                best = candidate
                best_sums = cand_sums
                best_size = len(best)
                in_best = set(best)
    
    return best, best_sums

def construct():
    start = time.time()
    deadline = start + 55  # safety margin before 60s limit
    
    best_set = None
    best_sums = None
    best_size = float('inf')
    
    # Phase 1: Try many random orderings
    for trial in range(500):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        sel = []
        sums = set()
        _greedy_random(order, sel, sums, deadline)
        # Saturate to make maximal
        sel = _saturate(sel, sums, deadline)
        if len(sel) < best_size:
            best_size = len(sel)
            best_set = list(sel)
            best_sums = set(sums)
            # Try local search
            improved, imp_sums = _local_search_reduce(sel, sums, deadline, max_iters=30)
            if len(improved) < best_size:
                best_size = len(improved)
                best_set = improved
                best_sums = imp_sums
    
    # Phase 2: More targeted local search on best found
    if best_set is not None and time.time() < deadline - 5:
        improved, imp_sums = _local_search_reduce(best_set, best_sums, deadline, max_iters=100)
        if len(improved) < best_size:
            best_set = improved
            best_sums = imp_sums
    
    # Fallback: simple greedy if nothing found
    if best_set is None:
        sel = []
        sums = set()
        for x in range(1, N + 1):
            if _can_add_fast(sel, sums, x):
                _add_element(sel, sums, x)
        best_set = sel
    
    return best_set
