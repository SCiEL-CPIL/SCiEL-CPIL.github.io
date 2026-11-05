import time
import itertools
import random

def _can_add(x, selected):
    """Check if x can be added to selected without violating the constraint."""
    n = len(selected)
    if n >= 3:
        # x must not divide sum of any 3 distinct elements in selected
        for i in range(n):
            for j in range(i+1, n):
                s_ij = selected[i] + selected[j]
                for k in range(j+1, n):
                    if (s_ij + selected[k]) % x == 0:
                        return False
    
    # For each a in selected, a must not divide x + u + v for any u, v in selected \ {a}
    for a_idx, a in enumerate(selected):
        # Collect indices of selected elements that are not a
        others = [selected[i] for i in range(n) if i != a_idx]
        m = len(others)
        for i in range(m):
            for j in range(i+1, m):
                if (x + others[i] + others[j]) % a == 0:
                    return False
    return True

def _greedy(order, deadline):
    """Run greedy in the given order."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _greedy_with_priority(selected, deadline, max_iterations=200):
    """Greedy that at each step picks the element that is 'safest' to add."""
    in_set = set(selected)
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        best_x = None
        best_score = -1
        # Score each candidate by how few conflicts it would create
        for x in range(192, 0, -1):
            if x in in_set:
                continue
            if not _can_add(x, selected):
                continue
            # Heuristic: prefer larger numbers (they're less likely to divide sums)
            score = x
            if score > best_score:
                best_score = score
                best_x = x
        if best_x is None:
            break
        selected.append(best_x)
        in_set.add(best_x)
    return selected

def _local_search(selected, deadline):
    """Try to improve the set by swapping or adding elements."""
    selected = list(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding any element not in selected
        in_set = set(selected)
        for x in range(192, 0, -1):
            if time.time() > deadline:
                return selected
            if x not in in_set and _can_add(x, selected):
                selected.append(x)
                in_set.add(x)
                improved = True
                break
        # Try swapping: remove one, add another (only if it enables future growth)
        if not improved:
            for i in range(len(selected)):
                if time.time() > deadline:
                    return selected
                candidate = selected[i]
                selected_without = selected[:i] + selected[i+1:]
                # Try to find a replacement that allows adding more elements
                best_swap = None
                for x in range(192, 0, -1):
                    if x in in_set:
                        continue
                    if _can_add(x, selected_without):
                        new_set = selected_without + [x]
                        # Check if this new set can accommodate more elements
                        new_in_set = set(new_set)
                        extra_count = 0
                        for y in range(192, 0, -1):
                            if time.time() > deadline:
                                break
                            if y not in new_in_set and _can_add(y, new_set):
                                extra_count += 1
                                if extra_count >= 2:
                                    break
                        if best_swap is None or extra_count > best_swap[0]:
                            best_swap = (extra_count, x, new_set)
                if best_swap and best_swap[0] > 0:
                    selected = best_swap[2]
                    in_set = set(selected)
                    improved = True
                    break
    return selected

def _build_conflict_graph():
    """Build a conflict graph where edge (i,j) means i and j cannot coexist.
    
    Two integers a and b conflict if:
    - a divides some sum of 3 distinct OTHER selected integers (but we can't know others yet)
    
    Actually, the constraint is global: no selected integer divides the sum of any 3 
    distinct OTHER selected integers. This is a 4-uniform hypergraph constraint, not a 
    simple pairwise graph.
    
    However, we can use a relaxation: if a and b are both selected, and there exist 
    c, d such that a | (b+c+d), then {a,b,c,d} is invalid. This is a 4-clique constraint.
    
    For a simpler approach, let's use the fact that large numbers are less likely to 
    divide small sums. We'll use a bitset-based branch and bound on a relaxed problem.
    """
    # For now, return empty conflict graph - the constraint is 4-uniform
    # We'll use a different approach: branch and bound with the actual constraint
    pass

def _branch_and_bound(deadline):
    """Branch and bound to find maximum valid set.
    
    Uses the actual constraint check but with smart ordering and pruning.
    """
    start_time = time.time()
    best_set = []
    best_size = 0
    
    # Order numbers by how "safe" they are (larger numbers first, as they're 
    # less likely to divide sums of other numbers)
    order = list(range(192, 0, -1))
    
    # Precompute: for each pair (i,j), what sums can they form with a third?
    # Actually, let's use a recursive branch and bound
    
    def _search(selected, remaining, current_size):
        nonlocal best_set, best_size, start_time
        
        if time.time() > deadline:
            return
        
        if current_size > best_size:
            best_set = list(selected)
            best_size = current_size
        
        if not remaining:
            return
        
        # Upper bound: current_size + len(remaining)
        if current_size + len(remaining) <= best_size:
            return
        
        # Try adding the first remaining element
        x = remaining[0]
        rest = remaining[1:]
        
        # Option 1: Include x
        if _can_add(x, selected):
            _search(selected + [x], rest, current_size + 1)
        
        # Option 2: Exclude x
        _search(selected, rest, current_size)
    
    # This naive approach is too slow. Let's use a better strategy.
    # Use greedy to get a good initial solution, then try to improve with swaps.
    
    # First, get a good initial solution with greedy
    initial = _greedy(range(192, 0, -1), deadline)
    best_set = list(initial)
    best_size = len(best_set)
    
    # Now try local search with more aggressive swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        in_set = set(best_set)
        
        # Try each possible swap
        for i in range(len(best_set)):
            if time.time() > deadline:
                return best_set
            to_remove = best_set[i]
            without = best_set[:i] + best_set[i+1:]
            
            # Try to find a better replacement
            for x in range(192, 0, -1):
                if time.time() > deadline:
                    return best_set
                if x in in_set:
                    continue
                if _can_add(x, without):
                    new_set = without + [x]
                    # Check if this new set can accommodate more elements
                    new_in_set = set(new_set)
                    extra = 0
                    for y in range(192, 0, -1):
                        if time.time() > deadline:
                            break
                        if y not in new_in_set and _can_add(y, new_set):
                            extra += 1
                            if extra >= 3:
                                break
                    
                    if extra > 0:
                        best_set = new_set
                        in_set = set(best_set)
                        improved = True
                        break
            if improved:
                break
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    best = []
    
    # Strategy 1: Greedy descending
    result = _greedy(range(192, 0, -1), deadline)
    if len(result) > len(best):
        best = result
    
    # Strategy 2: Greedy ascending
    if time.time() < deadline - 5:
        result = _greedy(range(1, 193), deadline)
        if len(result) > len(best):
            best = result
    
    # Strategy 3: Random shuffles
    random.seed(42)
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        order = list(range(1, 193))
        random.shuffle(order)
        result = _greedy(order, deadline)
        if len(result) > len(best):
            best = result
    
    # Strategy 4: Branch and bound with local search
    if time.time() < deadline - 10:
        result = _branch_and_bound(deadline)
        if len(result) > len(best):
            best = result
    
    # Local search on best
    if time.time() < deadline - 15:
        best = _local_search(best, deadline)
    
    return best
