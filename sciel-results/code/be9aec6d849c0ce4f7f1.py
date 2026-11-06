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

def _can_add(S, x):
    """
    Check if adding x to set S maintains the validity condition.
    Condition: No element in S U {x} divides the sum of any 3 distinct OTHER elements in S U {x}.
    """
    S_list = list(S)
    m = len(S_list)
    
    # 1) Check if x divides the sum of any 3 distinct elements in S.
    if m >= 3:
        if _any_triple_divisible(S_list, x):
            return False

    # 2) Check for each y in S, if y divides (x + u + v) for any distinct u,v in S \ {y}.
    #    The three summands are x, u, v where u,v are distinct elements from S\{y}
    #    and u != v. So we need: exists distinct u,v in S\{y} such that (x + u + v) % y == 0
    
    for y in S_list:
        # We need to check if there exist distinct u, v in S_list \ {y} such that
        # (x + u + v) % y == 0
        # i.e., (u + v) % y == (-x) % y
        
        target = (-x) % y
        others = [v for v in S_list if v != y]
        
        if len(others) < 2:
            continue
            
        # Check all pairs in others
        lo = len(others)
        for i in range(lo - 1):
            ui = others[i]
            for j in range(i + 1, lo):
                vj = others[j]
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
        if _can_add(current_set, x):
            current_set.add(x)
            S_list.append(x)
    return current_set, S_list

def _local_improve(best_set, deadline, rng):
    """Try to improve the set by adding elements or performing swaps."""
    n = 192
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        
        # Try adding any unselected element (prioritize those with fewer divisors)
        candidates = [x for x in range(1, n + 1) if x not in best_set]
        candidates.sort(key=lambda x: (_num_divisors(x), -x))
        
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if _can_add(best_set, x):
                best_set.add(x)
                improved = True
                break
        
        # If no direct add, try 1-out 1-in swap
        if not improved:
            for _ in range(30):
                if time.time() > deadline - 3.0:
                    break
                if len(best_set) < 1:
                    break
                a = rng.choice(list(best_set))
                trial_set = best_set - {a}
                
                # Try to find a replacement
                candidates = [x for x in range(1, n + 1) if x not in trial_set and x != a]
                candidates.sort(key=lambda x: (_num_divisors(x), -x))
                
                for x in candidates:
                    if time.time() > deadline - 3.0:
                        break
                    if _can_add(trial_set, x):
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
                
                # Try to add two elements
                candidates = [x for x in range(1, n + 1) if x not in trial_set and x != a]
                candidates.sort(key=lambda x: (_num_divisors(x), -x))
                
                first_added = False
                for x in candidates:
                    if time.time() > deadline - 3.0:
                        break
                    if _can_add(trial_set, x):
                        trial_set.add(x)
                        first_added = True
                        break
                        
                if first_added:
                    # Try to add a second element
                    candidates2 = [x for x in range(1, n + 1) if x not in trial_set]
                    candidates2.sort(key=lambda x: (_num_divisors(x), -x))
                    
                    for y in candidates2:
                        if time.time() > deadline - 3.0:
                            break
                        if _can_add(trial_set, y):
                            trial_set.add(y)
                            best_set = trial_set
                            improved = True
                            break
                            
                if improved:
                    break

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
    for attempt in range(200):
        if time.time() > deadline - 10.0:
            break
        
        # Mix strategies: sometimes use divisor priority, sometimes descending, sometimes random
        strategy = attempt % 3
        if strategy == 0:
            order = list(range(1, n + 1))
            order.sort(key=lambda x: (_num_divisors(x), -x))
        elif strategy == 1:
            order = list(range(n, 0, -1))
        else:
            order = list(range(1, n + 1))
            rng.shuffle(order)
        
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
