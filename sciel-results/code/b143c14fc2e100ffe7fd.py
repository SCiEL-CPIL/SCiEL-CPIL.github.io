"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _greedy(order):
    selected, reachable = [], {0}
    for x in order:
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    return selected, reachable

def _can_add(selected_set, reachable, x):
    """Check if adding x to the set keeps no subset summing to 200."""
    new = {s + x for s in reachable if s + x <= 200}
    return 200 not in new, new

def _local_search(selected, reachable, deadline):
    """Try to improve the solution via local search (swap operations)."""
    selected_set = set(selected)
    best_size = len(selected)
    
    # Try adding each excluded element
    excluded = [x for x in range(1, 101) if x not in selected_set]
    
    for x in excluded:
        if time.time() > deadline:
            break
        # Try adding x directly
        ok, new_reach = _can_add(selected_set, reachable, x)
        if ok:
            selected.append(x)
            selected_set.add(x)
            reachable = new_reach
            best_size = len(selected)
            # After adding, try to add more excluded elements
            for y in excluded:
                if y == x or y in selected_set:
                    continue
                if time.time() > deadline:
                    break
                ok2, new_reach2 = _can_add(selected_set, reachable, y)
                if ok2:
                    selected.append(y)
                    selected_set.add(y)
                    reachable = new_reach2
                    best_size = len(selected)
    
    # Try swap: remove one included, add one or two excluded
    for i, a in enumerate(list(selected)):
        if time.time() > deadline:
            break
        # Remove a
        new_sel = [s for s in selected if s != a]
        new_sel_set = set(new_sel)
        # Recompute reachable without a
        new_reach = {0}
        for s in new_sel:
            new_reach = {v for v in new_reach} | {v + s for v in new_reach if v + s <= 200}
        
        # Try adding one excluded element
        for x in excluded:
            if x in new_sel_set:
                continue
            ok, nr = _can_add(new_sel_set, new_reach, x)
            if ok:
                candidate = new_sel + [x]
                if len(candidate) > best_size:
                    best_size = len(candidate)
                    selected = candidate
                    selected_set = set(selected)
                    reachable = nr
                    excluded = [x for x in range(1, 101) if x not in selected_set]
        
        # Try adding two excluded elements
        for x in excluded:
            if x in new_sel_set:
                continue
            ok1, nr1 = _can_add(new_sel_set, new_reach, x)
            if not ok1:
                continue
            temp_set = new_sel_set | {x}
            for y in excluded:
                if y == x or y in temp_set:
                    continue
                ok2, nr2 = _can_add(temp_set, nr1, y)
                if ok2:
                    candidate = new_sel + [x, y]
                    if len(candidate) > best_size:
                        best_size = len(candidate)
                        selected = candidate
                        selected_set = set(selected)
                        reachable = nr2
                        excluded = [x for x in range(1, 101) if x not in selected_set]
    
    return selected, reachable

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    # Try multiple greedy orderings
    orders = [
        list(range(100, 0, -1)),  # descending
        list(range(1, 101)),       # ascending
        list(range(50, 0, -1)) + list(range(100, 50, -1)),  # mid then high
        list(range(100, 0, -1, 2)) + list(range(99, 0, -2)),  # evens then odds
        list(range(1, 101, 2)) + list(range(100, 0, -2)),    # odds then evens
    ]
    
    best = None
    best_reach = None
    best_size = 0
    
    for order in orders:
        if time.time() > deadline:
            break
        sel, reach = _greedy(order)
        if len(sel) > best_size:
            best_size = len(sel)
            best = sel
            best_reach = reach
    
    if best is None:
        best = list(range(100, 0, -1))
        best_reach = {0}
    
    # Local search to improve
    best, best_reach = _local_search(best[:], best_reach.copy(), deadline)
    
    return best
# EVOLVE-BLOCK-END
