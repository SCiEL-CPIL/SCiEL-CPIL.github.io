"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 85.0  # safety margin before 90s kill

def _conflict_count(selected):
    """Count number of (a, triple) pairs where a divides sum(triple).
    Returns list of (a, triple) conflicts."""
    conflicts = []
    n = len(selected)
    if n < 4:
        return conflicts
    sel = list(selected)
    # For each a, check all triples of OTHER elements
    for i in range(n):
        a = sel[i]
        others = [sel[j] for j in range(n) if j != i]
        m = len(others)
        if m < 3:
            continue
        for t in combinations(range(m), 3):
            s = others[t[0]] + others[t[1]] + others[t[2]]
            if s % a == 0:
                conflicts.append((a, t))
    return conflicts

def _is_valid(selected):
    """Check if no element divides sum of any 3 distinct other elements."""
    n = len(selected)
    if n < 4:
        return True
    sel = list(selected)
    for i in range(n):
        a = sel[i]
        others = [sel[j] for j in range(n) if j != i]
        m = len(others)
        if m < 3:
            continue
        for t in combinations(range(m), 3):
            s = others[t[0]] + others[t[1]] + others[t[2]]
            if s % a == 0:
                return False
    return True

def _conflicts_involving(a, selected):
    """Count conflicts where 'a' is the divisor, or 'a' is in a summand triple that sums to a multiple of some other element.
    Actually we need: conflicts where a is the tested element, OR a appears in a triple that is a multiple of another selected element.
    For removal heuristic, count how many conflicts are 'fixed' by removing a.
    A conflict is (divisor d, triple T) where sum(T) % d == 0.
    Removing a fixes:
      - conflicts where a == d (a is the divisor)
      - conflicts where a is in T (a is a summand)
    """
    fixed = 0
    n = len(selected)
    if n < 4:
        return 0
    sel = list(selected)
    for i in range(n):
        d = sel[i]
        others = [sel[j] for j in range(n) if j != i]
        m = len(others)
        if m < 3:
            continue
        # conflicts where d is the divisor
        if d == a:
            for t in combinations(range(m), 3):
                s = others[t[0]] + others[t[1]] + others[t[2]]
                if s % d == 0:
                    fixed += 1
        else:
            # a is a summand in others of d
            # find index of a in others
            try:
                idx = others.index(a)
            except ValueError:
                continue
            for t in combinations(range(m), 3):
                if idx in t:
                    s = others[t[0]] + others[t[1]] + others[t[2]]
                    if s % d == 0:
                        fixed += 1
    return fixed

def _greedy_remove(candidate_set):
    """Greedily remove elements to eliminate all conflicts.
    At each step, remove the element that fixes the most conflicts (or any if tie).
    """
    selected = list(candidate_set)
    while True:
        if time.time() > _DEADLINE:
            break
        # find all conflicts
        conflicts = _conflict_count(selected)
        if not conflicts:
            break
        # count how many conflicts each element is involved in
        from collections import defaultdict
        involvement = defaultdict(int)
        for (d, t) in conflicts:
            involvement[d] += 1
            sel = selected
            others = [v for v in sel if v != d]
            for idx in t:
                involvement[others[idx]] += 1
        # remove element with max involvement
        max_elem = None
        max_count = 0
        for e, c in involvement.items():
            if c > max_count:
                max_count = c
                max_elem = e
        if max_elem is None:
            break
        selected.remove(max_elem)
    return selected

def _local_search(selected, removed_pool):
    """Try to add back elements from removed_pool, or swap, to increase size while keeping valid.
    Returns improved selected list.
    """
    selected = list(selected)
    removed = list(removed_pool)
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        # try adding back each removed element
        for e in removed[:]:
            if time.time() > _DEADLINE:
                break
            trial = selected + [e]
            if _is_valid(trial):
                selected = trial
                removed.remove(e)
                improved = True
    return selected

def _random_restart_greedy_remove(lo, hi, rng):
    """Start with a random subset of [lo,hi], then greedy remove."""
    candidates = list(range(lo, hi+1))
    rng.shuffle(candidates)
    # take a dense subset
    selected = _greedy_remove(candidates)
    return selected

def construct():
    global _START_TIME
    if not hasattr(construct, '_done'):
        _START_TIME = time.time()
    remaining = _DEADLINE - time.time()
    if remaining <= 0:
        return list(range(144, 193))  # fallback
    
    rng = random.Random(42)
    best = None
    best_size = 0
    
    # Strategy 1: dense high range, greedy remove
    # Use range [120, 192] or similar
    for (lo, hi) in [(144, 192), (120, 192), (100, 192), (80, 192), (1, 192)]:
        if time.time() > _DEADLINE:
            break
        candidates = list(range(lo, hi+1))
        # greedy remove from this set
        selected = _greedy_remove(candidates)
        if len(selected) > best_size:
            best_size = len(selected)
            best = list(selected)
    
    if best is None:
        best = []
    
    # Strategy 2: local search to add back elements
    # Determine what's removed
    all_used = set(best)
    removed_pool = [x for x in range(1, 193) if x not in all_used]
    improved = _local_search(best, removed_pool)
    if len(improved) > best_size:
        best = improved
        best_size = len(best)
    
    # Strategy 3: multiple random restarts with greedy remove + local search
    start = time.time()
    while time.time() < _DEADLINE - 3.0:  # leave 3s margin
        # random dense subset
        lo = rng.randint(60, 144)
        hi = rng.randint(lo+20, 192)
        candidates = list(range(lo, hi+1))
        rng.shuffle(candidates)
        selected = _greedy_remove(candidates)
        # local search
        all_used = set(selected)
        removed_pool = [x for x in range(1, 193) if x not in all_used]
        improved = _local_search(selected, removed_pool)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
            # if we hit max, we can try to keep improving or break
            if best_size >= 48:
                # try to see if we can do better, but 48 is the cap
                pass
    
    # Final local search on best
    all_used = set(best)
    removed_pool = [x for x in range(1, 193) if x not in all_used]
    improved = _local_search(best, removed_pool)
    if len(improved) > best_size:
        best = improved
        best_size = len(best)
    
    # Ensure we return a valid list, sorted
    best = sorted(best)
    # Final validation
    if not _is_valid(best):
        # fallback to greedy remove of full high range
        selected = _greedy_remove(list(range(144, 193)))
        best = sorted(selected)
    
    return best
# EVOLVE-BLOCK-END
