import time
import random
from itertools import combinations

def _num_divisors(n):
    """Count divisors of n."""
    if n <= 0:
        return 0
    count = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            count += 1
            if i * i != n:
                count += 1
        i += 1
    return count

def _any_triple_divisible(S_list, x):
    """Check if x divides the sum of any 3 distinct elements in S_list."""
    m = len(S_list)
    if m < 3:
        return False
    for i in range(m - 2):
        a = S_list[i]
        for j in range(i + 1, m - 1):
            b = S_list[j]
            ab = a + b
            for k in range(j + 1, m):
                if (ab + S_list[k]) % x == 0:
                    return True
    return False

def _can_add_fast(S_list, x):
    """
    Fast check if adding x to S_list maintains validity.
    Only checks new constraints involving x, not re-checking existing triples.
    """
    m = len(S_list)
    
    # 1) Check if x divides the sum of any 3 distinct elements in S_list.
    if m >= 3:
        for i in range(m - 2):
            a = S_list[i]
            for j in range(i + 1, m - 1):
                b = S_list[j]
                ab = a + b
                for k in range(j + 1, m):
                    if (ab + S_list[k]) % x == 0:
                        return False

    # 2) For each y in S_list, check if y divides (x + u + v) for any distinct u,v in S_list \ {y}
    for idx_y, y in enumerate(S_list):
        target = (-x) % y
        # Check pairs (u, v) from S_list where u != y, v != y, u != v
        for i in range(m):
            if i == idx_y:
                continue
            ui = S_list[i]
            for j in range(i + 1, m):
                if j == idx_y:
                    continue
                vj = S_list[j]
                if (ui + vj) % y == target:
                    return False
                    
    return True

def _is_valid(S):
    """Full validation."""
    S_list = list(S)
    m = len(S_list)
    for a in S_list:
        others = [v for v in S_list if v != a]
        if len(others) < 3:
            continue
        if _any_triple_divisible(others, a):
            return False
    return True

def _greedy(order, deadline):
    current_set = set()
    S_list = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add_fast(S_list, x):
            current_set.add(x)
            S_list.append(x)
    return current_set, S_list

def _build_conflict_graph(S_list):
    """
    Build a conflict graph: for each pair (i, j) where i < j,
    determine if removing element i or j would break some constraint.
    Returns a dict: element -> set of elements it conflicts with.
    """
    m = len(S_list)
    conflicts = {x: set() for x in S_list}
    
    # For each element y, find which triples it divides
    # If y divides sum of triple (a,b,c), then y conflicts with a, b, c
    for y in S_list:
        others = [v for v in S_list if v != y]
        if len(others) < 3:
            continue
        for i in range(len(others) - 2):
            a = others[i]
            for j in range(i + 1, len(others) - 1):
                b = others[j]
                ab = a + b
                for k in range(j + 1, len(others)):
                    c = others[k]
                    if (ab + c) % y == 0:
                        # y divides a+b+c, so y conflicts with a, b, c
                        conflicts[y].add(a)
                        conflicts[a].add(y)
                        conflicts[y].add(b)
                        conflicts[b].add(y)
                        conflicts[y].add(c)
                        conflicts[c].add(y)
    
    return conflicts

def _local_improve(best_set, deadline, rng):
    """Try to improve the set by adding elements or performing swaps."""
    n = 192
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        S_list = list(best_set)
        
        # Try adding any unselected element (prioritize those with fewer divisors)
        candidates = [x for x in range(1, n + 1) if x not in best_set]
        candidates.sort(key=lambda x: (_num_divisors(x), -x))
        
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if _can_add_fast(S_list, x):
                best_set.add(x)
                S_list.append(x)
                improved = True
                break
        
        # If no direct add, try 1-out 1-in swap
        if not improved:
            # Build conflict graph to identify which elements to remove
            conflicts = _build_conflict_graph(S_list)
            
            # Sort elements by conflict degree (most conflicted first)
            sorted_elements = sorted(S_list, key=lambda x: -len(conflicts.get(x, set())))
            
            for a in sorted_elements:
                if time.time() > deadline - 3.0:
                    break
                trial_set = best_set - {a}
                trial_list = [v for v in S_list if v != a]
                
                # Try to find a replacement
                candidates = [x for x in range(1, n + 1) if x not in trial_set and x != a]
                candidates.sort(key=lambda x: (_num_divisors(x), -x))
                
                for x in candidates:
                    if time.time() > deadline - 3.0:
                        break
                    if _can_add_fast(trial_list, x):
                        trial_set.add(x)
                        best_set = trial_set
                        improved = True
                        break
                if improved:
                    break
                    
        # If still no improvement, try 1-out 2-in swap
        if not improved:
            for _ in range(15):
                if time.time() > deadline - 3.0:
                    break
                if len(best_set) < 1:
                    break
                a = rng.choice(list(best_set))
                trial_set = best_set - {a}
                trial_list = [v for v in S_list if v != a]
                
                # Try to add two elements
                candidates = [x for x in range(1, n + 1) if x not in trial_set and x != a]
                candidates.sort(key=lambda x: (_num_divisors(x), -x))
                
                first_added = False
                for x in candidates:
                    if time.time() > deadline - 3.0:
                        break
                    if _can_add_fast(trial_list, x):
                        trial_set.add(x)
                        trial_list.append(x)
                        first_added = True
                        break
                        
                if first_added:
                    # Try to add a second element
                    candidates2 = [x for x in range(1, n + 1) if x not in trial_set]
                    candidates2.sort(key=lambda x: (_num_divisors(x), -x))
                    
                    for y in candidates2:
                        if time.time() > deadline - 3.0:
                            break
                        if _can_add_fast(trial_list, y):
                            trial_set.add(y)
                            best_set = trial_set
                            improved = True
                            break
                            
                if improved:
                    break
                S_list = list(best_set)

    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 192
    best_set = set()
    rng = random.Random(42)
    
    # Strategy 1: Greedy with divisor-based priority
    # Numbers with fewer divisors are less likely to divide sums
    order = list(range(1, n + 1))
    order.sort(key=lambda x: (_num_divisors(x), -x))
    current_set, _ = _greedy(order, deadline)
    if len(current_set) > len(best_set):
        best_set = current_set.copy()
        
    # Strategy 2: Descending Greedy (larger numbers first)
    order_desc = list(range(n, 0, -1))
    current_set, _ = _greedy(order_desc, deadline)
    if len(current_set) > len(best_set):
        best_set = current_set.copy()
        
    # Strategy 3: Randomized Greedy with multiple restarts
    for attempt in range(300):
        if time.time() > deadline - 10.0:
            break
        
        # Mix strategies: sometimes use divisor priority, sometimes descending, sometimes random
        strategy = attempt % 4
        if strategy == 0:
            order = list(range(1, n + 1))
            order.sort(key=lambda x: (_num_divisors(x), -x))
        elif strategy == 1:
            order = list(range(n, 0, -1))
        elif strategy == 2:
            order = list(range(1, n + 1))
            rng.shuffle(order)
        else:
            # Hybrid: start with high numbers, then fill with low-divisor numbers
            high = list(range(n, n // 2, -1))
            low = [x for x in range(1, n // 2 + 1) if x not in set(high)]
            low.sort(key=lambda x: _num_divisors(x))
            order = high + low
        
        current_set, _ = _greedy(order, deadline)
        
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
            
    # Local Search / Improvement
    if time.time() < deadline - 5.0:
        best_set = _local_improve(best_set, deadline, rng)

    # Final Validation
    if _is_valid(best_set):
        return list(best_set)
    
    # Fallback: return a small valid set
    fallback = [192, 191, 189, 187, 185]
    if _is_valid(fallback):
        return fallback
    
    return [192]
