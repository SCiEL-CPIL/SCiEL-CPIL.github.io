"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _precompute_degrees():
    """Precompute for each element x in [1,100], how many subsets of [1,100]
    containing x sum to 200. This gives the 'degree' in the hypergraph model."""
    # For each x, count the number of subsets of [1,100] \ {x} that sum to 200-x
    # We'll use DP to count subset sums
    degrees = {}
    for x in range(1, 101):
        target = 200 - x
        if target < 0:
            degrees[x] = 0
            continue
        # Count subsets of {1,...,100} \ {x} that sum to target
        # Use DP: dp[s] = number of ways to make sum s from available elements
        available = [i for i in range(1, 101) if i != x]
        dp = {0: 1}
        for elem in available:
            new_dp = dict(dp)
            for s, cnt in dp.items():
                ns = s + elem
                if ns <= target:
                    new_dp[ns] = new_dp.get(ns, 0) + cnt
            dp = new_dp
        degrees[x] = dp.get(target, 0)
    return degrees

def _greedy_by_degree(degrees, deadline=None):
    """Greedy: repeatedly pick the element with lowest degree among those that
    can still be added without creating a subset sum of 200."""
    selected = []
    reachable = {0}
    remaining = list(range(1, 101))
    selected_set = set()
    
    while remaining:
        if deadline and time.time() > deadline:
            break
        # Find the element with lowest degree that can be added
        best_x = None
        best_degree = float('inf')
        for x in remaining:
            if x in selected_set:
                continue
            new = {s + x for s in reachable if s + x <= 200}
            if 200 not in new:
                deg = degrees.get(x, 0)
                if deg < best_degree:
                    best_degree = deg
                    best_x = x
        
        if best_x is None:
            break
        
        selected.append(best_x)
        selected_set.add(best_x)
        reachable = {s + best_x for s in reachable if s + best_x <= 200} | reachable
        remaining.remove(best_x)
    
    return selected, reachable

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
    
    # Sort excluded by degree (prefer low-degree elements)
    # We'll use a simple heuristic: smaller elements tend to have more subset sums
    # but we'll just try all and keep the best
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in excluded:
            if time.time() > deadline:
                break
            ok, new_reach = _can_add(selected_set, reachable, x)
            if ok:
                selected.append(x)
                selected_set.add(x)
                reachable = new_reach
                best_size = len(selected)
                excluded.remove(x)
                improved = True
                # After adding, try to add more excluded elements
                for y in list(excluded):
                    if time.time() > deadline:
                        break
                    ok2, new_reach2 = _can_add(selected_set, reachable, y)
                    if ok2:
                        selected.append(y)
                        selected_set.add(y)
                        reachable = new_reach2
                        best_size = len(selected)
                        excluded.remove(y)
                        improved = True
    
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
    
    # Precompute degrees for the hypergraph model
    degrees = _precompute_degrees()
    
    # Try multiple greedy orderings
    orders = [
        list(range(100, 0, -1)),  # descending
        list(range(1, 101)),       # ascending
        list(range(50, 0, -1)) + list(range(100, 50, -1)),  # mid then high
        list(range(100, 0, -2)) + list(range(99, 0, -2)),    # evens then odds
        list(range(1, 101, 2)) + list(range(100, 0, -2)),    # odds then evens
    ]
    
    best = None
    best_reach = None
    best_size = 0
    
    # First try the degree-based greedy
    if time.time() < deadline:
        sel, reach = _greedy_by_degree(degrees, deadline)
        if len(sel) > best_size:
            best_size = len(sel)
            best = sel
            best_reach = reach
    
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
