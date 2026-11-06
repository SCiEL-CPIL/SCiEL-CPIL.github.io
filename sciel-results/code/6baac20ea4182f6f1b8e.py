"""Rewritten small maximal Sidon-set constructor for N=256."""
import time
import random

N = 256
_DEADLINE = 55.0  # safety margin before 60s kill


def _conflict_with_selected(x, selected, sums):
    """Return True if adding x to selected (with sums) would break Sidon property."""
    # Check new sums: x+x and x+a for a in selected
    new_sums = [2 * x]
    for a in selected:
        new_sums.append(x + a)
    # Check for internal conflicts among new_sums
    if len(set(new_sums)) != len(new_sums):
        return True
    # Check against existing sums
    for s in new_sums:
        if s in sums:
            return True
    return False


def _is_maximal(selected, sums, N):
    """Check if selected is inclusion-maximal in [1, N]."""
    sel_set = set(selected)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Try adding x
        new_sums = [2 * x]
        for a in selected:
            new_sums.append(x + a)
        if len(set(new_sums)) != len(new_sums):
            continue
        conflict = False
        for s in new_sums:
            if s in sums:
                conflict = True
                break
        if not conflict:
            return False  # can be extended
    return True


def _greedy_construct(order):
    """Greedy Sidon construction following a given element order."""
    selected = []
    sums = set()
    for x in order:
        new_sums = [2 * x]
        for a in selected:
            new_sums.append(x + a)
        if len(set(new_sums)) != len(new_sums):
            continue
        if sums.intersection(new_sums):
            continue
        selected.append(x)
        sums.update(new_sums)
    return selected, sums


def _local_search(selected, sums, N, deadline):
    """Try to remove elements that are still needed for maximality.
    Actually, we want a SMALLER maximal set. So we look for elements whose
    removal still leaves a maximal set.
    """
    best = list(selected)
    best_sums = set(sums)
    
    # First, build a valid set from selected
    sel = list(selected)
    sums_s = set(sums)
    sel_set = set(sel)
    
    # Try removing each element and checking if result is still maximal
    # (and still a valid Sidon set)
    improved = True
    while improved:
        improved = False
        for i in range(len(sel)):
            if time.time() > deadline:
                return best, best_sums
            candidate = sel[:i] + sel[i+1:]
            if not candidate:
                continue
            # Check Sidon validity
            c_sums = set()
            valid = True
            for idx, a in enumerate(candidate):
                for b in candidate[idx:]:
                    s = a + b
                    if s in c_sums:
                        valid = False
                        break
                    c_sums.add(s)
                if not valid:
                    break
            if not valid:
                continue
            # Check maximality
            c_set = set(candidate)
            maximal = True
            for x in range(1, N + 1):
                if x in c_set:
                    continue
                ns = [2 * x]
                for a in candidate:
                    ns.append(x + a)
                if len(set(ns)) != len(ns):
                    continue
                if c_sums.intersection(ns):
                    continue
                maximal = False
                break
            if maximal and len(candidate) < len(best):
                best = candidate
                best_sums = c_sums
                sel = candidate
                sums_s = c_sums
                improved = True
                break
    return best, best_sums


def _random_order(seed):
    """Generate a random permutation of [1, N]."""
    rng = random.Random(seed)
    order = list(range(1, N + 1))
    rng.shuffle(order)
    return order


def _descending_order():
    return list(range(N, 0, -1))


def _ascending_order():
    return list(range(1, N + 1))


def _mixed_order(seed):
    """Mix of strategies: start with some structure."""
    rng = random.Random(seed)
    # Try a few patterns
    patterns = []
    # Random
    o = list(range(1, N + 1))
    rng.shuffle(o)
    patterns.append(o)
    # Descending
    patterns.append(list(range(N, 0, -1)))
    # Ascending
    patterns.append(list(range(1, N + 1)))
    # Prime-like bias (odd numbers first)
    odds = [x for x in range(1, N + 1) if x % 2 == 1]
    evens = [x for x in range(1, N + 1) if x % 2 == 0]
    rng.shuffle(odds)
    rng.shuffle(evens)
    patterns.append(odds + evens)
    # Multiples of 3 first
    m3 = [x for x in range(1, N + 1) if x % 3 == 0]
    not_m3 = [x for x in range(1, N + 1) if x % 3 != 0]
    rng.shuffle(m3)
    rng.shuffle(not_m3)
    patterns.append(m3 + not_m3)
    return patterns


def construct():
    start_time = time.time()
    
    # Try multiple strategies and keep the best (smallest) maximal Sidon set
    best_set = None
    best_sums = None
    
    # Strategy 1: Ascending greedy (baseline)
    sel, sums = _greedy_construct(_ascending_order())
    if sel and _is_maximal(sel, sums, N):
        best_set = sel
        best_sums = sums
    
    # Strategy 2: Descending greedy
    sel, sums = _greedy_construct(_descending_order())
    if sel and _is_maximal(sel, sums, N):
        if best_set is None or len(sel) < len(best_set):
            best_set = sel
            best_sums = sums
    
    # Strategy 3: Random orders with local search
    for seed in range(200):
        if time.time() > _DEADLINE:
            break
        patterns = _mixed_order(seed)
        for order in patterns:
            if time.time() > _DEADLINE:
                break
            sel, sums = _greedy_construct(order)
            if not sel:
                continue
            if not _is_maximal(sel, sums, N):
                continue
            # Local search to shrink
            sel2, sums2 = _local_search(sel, sums, N, _DEADLINE)
            if best_set is None or len(sel2) < len(best_set):
                best_set = sel2
                best_sums = sums2
    
    if best_set is None:
        # Fallback: return ascending greedy (should be maximal)
        sel, sums = _greedy_construct(_ascending_order())
        return sel
    
    return best_set
