import time
import random

_START = time.time()
_DEADLINE = 60.0  # 60s limit, keep margin
_MARGIN = 2.0

def _valid_bitmask(s, target=200):
    """
    Check if any subset of s sums to target using bitmasks.
    Returns True if NO subset sums to target.
    """
    reach = 1  # Bit 0 set
    mask = (1 << (target + 1)) - 1
    for x in s:
        if x > target:
            continue
        reach |= (reach << x)
        reach &= mask
        if (reach >> target) & 1:
            return False
    return True

def _greedy_add(order, target=200):
    """
    Greedy construction: add elements in given order if they don't create sum 'target'.
    """
    selected = []
    reach = 1
    mask = (1 << (target + 1)) - 1
    for x in order:
        if x > target:
            # Elements > target cannot be part of a sum equal to target (positive integers)
            # But they are valid to include as they don't contribute to the sum.
            # Wait, the problem says "subset of [1,100]".
            # If x > 200, it's not in [1,100].
            # All x in [1,100] are <= 100 < 200.
            # So this check is not strictly necessary for [1,100], but good for safety.
            pass
        
        new_reach = reach | (reach << x)
        new_reach &= mask
        if not (new_reach >> target) & 1:
            selected.append(x)
            reach = new_reach
    return selected

def _local_search(s, target=200):
    """
    Try to add back excluded elements to maximize cardinality.
    """
    s = list(s)
    s_set = set(s)
    all_elements = set(range(1, 101))
    excluded = [x for x in all_elements if x not in s_set]
    
    # Shuffle to avoid bias
    random.shuffle(excluded)
    
    changed = True
    while changed:
        changed = False
        # We iterate over a copy because we modify excluded
        for x in excluded[:]:
            if x in s_set:
                continue
            # Test adding x
            trial = s + [x]
            if _valid_bitmask(trial, target):
                s = trial
                s_set.add(x)
                excluded.remove(x)
                changed = True
    return s

def construct():
    if time.time() - _START > _DEADLINE - _MARGIN:
        return [1]
        
    best = []
    
    # 1. Descending order
    s1 = _greedy_add(range(100, 0, -1))
    if len(s1) > len(best):
        best = s1
        
    # 2. Ascending order
    s2 = _greedy_add(range(1, 101))
    if len(s2) > len(best):
        best = s2
        
    # 3. Random restarts
    # Budget: ~50 restarts
    # Each restart is O(100 * big_int_ops) which is very fast.
    # Local search is the expensive part, so we do it once on the best.
    
    num_restarts = 100
    for _ in range(num_restarts):
        if time.time() - _START > _DEADLINE - _MARGIN:
            break
        order = list(range(1, 101))
        random.shuffle(order)
        s_rand = _greedy_add(order)
        if len(s_rand) > len(best):
            best = s_rand
            
    # 4. Local Search on the best candidate
    if time.time() - _START < _DEADLINE - _MARGIN:
        best = _local_search(best)
        
    # 5. Final Verification
    if not _valid_bitmask(best):
        # Fallback
        best = [1]
        
    return best
