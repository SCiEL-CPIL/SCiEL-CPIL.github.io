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

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    best = []
    
    # Strategy 1: Greedy descending
    t1 = time.time()
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
    
    # Strategy 4: Priority-based greedy from different starting points
    if time.time() < deadline - 10:
        for start_size in range(0, 5):
            if time.time() > deadline - 10:
                break
            initial = list(range(192, 192 - start_size, -1))
            result = _greedy_with_priority(initial, deadline)
            if len(result) > len(best):
                best = result
    
    # Local search on best
    if time.time() < deadline - 15:
        best = _local_search(best, deadline)
    
    return best
