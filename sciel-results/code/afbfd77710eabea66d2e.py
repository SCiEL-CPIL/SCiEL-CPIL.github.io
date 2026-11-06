"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

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

def _local_search_remove(selected, selected_set, reachable, deadline):
    """Try removing one element and adding two others (2-swap local search)."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i, x in enumerate(selected[:]):
            if time.time() > deadline:
                break
            # Remove x
            new_set = selected_set - {x}
            new_list = [y for y in selected if y != x]
            # Recompute reachable without x
            new_reach = 1
            for y in new_list:
                new_reach = _add(new_reach, y)
            # Try adding two excluded elements
            excluded = [z for z in range(1, 101) if z not in new_set]
            added = []
            for z1 in excluded:
                if time.time() > deadline:
                    break
                if _can_add(new_reach, z1):
                    temp_reach = _add(new_reach, z1)
                    for z2 in excluded:
                        if z2 == z1:
                            continue
                        if _can_add(temp_reach, z2):
                            # We can add both z1 and z2
                            final_reach = _add(temp_reach, z2)
                            if len(added) < 2:
                                added = [z1, z2]
                                break
                    if len(added) == 2:
                        break
            if len(added) == 2:
                selected = new_list + added
                selected_set = new_set | set(added)
                reachable = final_reach
                improved = True
    return selected, selected_set, reachable

def construct():
    start = time.time()
    deadline = start + 80  # 80s deadline, 10s safety margin
    
    best = []
    best_len = 0
    
    def try_update(selected):
        nonlocal best, best_len
        if len(selected) > best_len:
            best = list(selected)
            best_len = len(selected)
            return True
        return False
    
    # Strategy 1: Greedy high to low
    if time.time() < deadline:
        selected, sset, reachable = _greedy(range(100, 0, -1), deadline)
        selected, sset, reachable = _local_search(selected, sset, reachable, deadline)
        try_update(selected)
    
    # Strategy 2: Greedy low to high
    if time.time() < deadline:
        selected2, sset2, reachable2 = _greedy(range(1, 101), deadline)
        selected2, sset2, reachable2 = _local_search(selected2, sset2, reachable2, deadline)
        try_update(selected2)
    
    # Strategy 3: Start with {66..100} known good, try to extend
    if time.time() < deadline:
        base = list(range(66, 101))
        bset = set(base)
        breachable = 1
        for x in base:
            breachable = _add(breachable, x)
        base, bset, breachable = _local_search(base, bset, breachable, deadline)
        try_update(base)
    
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
            try_update(base)
    
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
            try_update(base)
    
    # Strategy 6: Randomized greedy constructions with multiple seeds
    rng = random.Random(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        order = list(range(1, 101))
        rng.shuffle(order)
        selected, sset, reachable = _greedy(order, deadline)
        selected, sset, reachable = _local_search(selected, sset, reachable, deadline)
        try_update(selected)
    
    # Strategy 7: More random trials
    for trial in range(100):
        if time.time() > deadline:
            break
        order = list(range(1, 101))
        rng.shuffle(order)
        selected, sset, reachable = _greedy(order, deadline)
        selected, sset, reachable = _local_search(selected, sset, reachable, deadline)
        try_update(selected)
    
    # Strategy 8: Try 2-swap local search on best solution
    if time.time() < deadline and best:
        bset = set(best)
        reach = 1
        for x in best:
            reach = _add(reach, x)
        best, bset, reach = _local_search_remove(best, bset, reach, deadline)
        try_update(best)
    
    # Strategy 9: More aggressive random restarts with different initial biases
    for bias in [0.3, 0.5, 0.7, 0.9]:
        if time.time() > deadline:
            break
        order = []
        for x in range(1, 101):
            if rng.random() < bias:
                order.append(x)
        for x in range(1, 101):
            if x not in order:
                order.append(x)
        selected, sset, reachable = _greedy(order, deadline)
        selected, sset, reachable = _local_search(selected, sset, reachable, deadline)
        try_update(selected)
    
    # Strategy 10: Final polish - try all single removals and re-additions
    if time.time() < deadline and best:
        for x in best[:]:
            if time.time() > deadline:
                break
            new_list = [y for y in best if y != x]
            new_set = set(new_list)
            new_reach = 1
            for y in new_list:
                new_reach = _add(new_reach, y)
            # Try to add back x and potentially others
            excluded = [z for z in range(1, 101) if z not in new_set]
            added_any = False
            for z in excluded:
                if _can_add(new_reach, z):
                    new_list.append(z)
                    new_set.add(z)
                    new_reach = _add(new_reach, z)
                    added_any = True
            if len(new_list) > len(best):
                best = new_list
                best_len = len(new_list)
    
    return best
# EVOLVE-BLOCK-END
