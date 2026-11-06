"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

MASK = (1 << 201) - 1  # bits 0 through 200

def _can_add(reachable, x):
    """Check if adding x would create sum 200."""
    return not ((reachable << x) & MASK & (1 << 200))

def _add(reachable, x):
    """Update reachable sums after adding x."""
    return reachable | ((reachable << x) & MASK)

def _greedy(order, deadline):
    """Greedy construction following given order."""
    selected = []
    selected_set = set()
    reachable = 1  # bit 0
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(reachable, x):
            selected.append(x)
            selected_set.add(x)
            reachable = _add(reachable, x)
    return selected, selected_set, reachable

def _local_search(selected, selected_set, reachable, deadline):
    """Try to add excluded elements."""
    excluded = [x for x in range(1, 101) if x not in selected_set]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in excluded[:]:
            if time.time() > deadline:
                break
            if _can_add(reachable, x):
                selected.append(x)
                selected_set.add(x)
                reachable = _add(reachable, x)
                excluded.remove(x)
                improved = True
    return selected, selected_set, reachable

def construct():
    deadline = time.time() + 85  # 85s deadline, 5s safety margin
    
    best = []
    best_len = 0
    
    # Strategy 1: Greedy high to low
    selected, sset, reachable = _greedy(range(100, 0, -1), deadline)
    selected, sset, reachable = _local_search(selected, sset, reachable, deadline)
    if len(selected) > best_len:
        best = selected
        best_len = len(selected)
    
    # Strategy 2: Greedy low to high
    if time.time() < deadline:
        selected2, sset2, reachable2 = _greedy(range(1, 101), deadline)
        selected2, sset2, reachable2 = _local_search(selected2, sset2, reachable2, deadline)
        if len(selected2) > best_len:
            best = selected2
            best_len = len(selected2)
    
    # Strategy 3: Start with {66..100} known good, try to extend
    if time.time() < deadline:
        base = list(range(66, 101))
        bset = set(base)
        breachable = 1
        for x in base:
            breachable = _add(breachable, x)
        base, bset, breachable = _local_search(base, bset, breachable, deadline)
        if len(base) > best_len:
            best = base
            best_len = len(base)
    
    # Strategy 4: Try to build larger by being selective in 66-100
    # Skip one element from 66-100 and see if we can add more from 1-65
    if time.time() < deadline:
        for skip in range(66, 101):
            if time.time() > deadline:
                break
            base = [x for x in range(66, 101) if x != skip]
            bset = set(base)
            breachable = 1
            for x in base:
                breachable = _add(breachable, x)
            base, bset, breachable = _local_search(base, bset, breachable, deadline)
            if len(base) > best_len:
                best = base
                best_len = len(base)
    
    # Strategy 5: More aggressive - skip two from 66-100
    if time.time() < deadline:
        import itertools
        for skip1, skip2 in itertools.combinations(range(66, 101), 2):
            if time.time() > deadline:
                break
            base = [x for x in range(66, 101) if x != skip1 and x != skip2]
            bset = set(base)
            breachable = 1
            for x in base:
                breachable = _add(breachable, x)
            base, bset, breachable = _local_search(base, bset, breachable, deadline)
            if len(base) > best_len:
                best = base
                best_len = len(base)
    
    return best
# EVOLVE-BLOCK-END
