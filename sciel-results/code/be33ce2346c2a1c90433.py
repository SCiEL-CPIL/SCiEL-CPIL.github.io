import time
import random
import sys

DEADLINE_SECONDS = 85.0  # Safety margin for 90s limit

def _has_divisibility_conflict(sums):
    """
    Check if any two distinct values in `sums` divide each other.
    Uses sorted array + set for O(M * log(M)) worst case but often faster.
    """
    if len(sums) < 2:
        return False
    
    arr = sorted(sums)
    n = len(arr)
    max_val = arr[-1]
    
    # If the smallest element a is such that 2*a > max(arr), no conflict possible
    if 2 * arr[0] > max_val:
        return False
        
    sum_set = set(arr)
    
    for i in range(n):
        a = arr[i]
        # If 2*a > max_val, no multiples can exist for this or any larger a
        if 2 * a > max_val:
            break
        # Check multiples 2a, 3a, ...
        multiple = 2 * a
        while multiple <= max_val:
            if multiple in sum_set:
                return True
            multiple += a
            
    return False

def _can_add(current_sums_set, x):
    """
    Check if x can be added to the set defined by current_sums_set.
    Returns (new_sums_set) or None.
    We assume current_sums_set is valid.
    New sums are {x} U {s+x for s in current_sums_set}.
    """
    if x in current_sums_set:
        return None
        
    # Generate new sums: s + x for s in current_sums_set
    # Check no collision with existing sums
    new_sums_added = set()
    for s in current_sums_set:
        ns = s + x
        if ns in current_sums_set:
            return None
        if ns in new_sums_added:
            return None
        new_sums_added.add(ns)
    
    # Build new set and check divisibility
    new_sums = current_sums_set | new_sums_added | {x}
    
    if _has_divisibility_conflict(new_sums):
        return None
        
    return new_sums

def _greedy_construct(pool, order_key, start_time, deadline):
    """
    Greedy construction with adaptive ordering.
    """
    elements = []
    sums_set = set()
    
    candidates = list(pool)
    if order_key == 'asc':
        candidates.sort()
    elif order_key == 'desc':
        candidates.sort(reverse=True)
    elif order_key == 'random':
        random.shuffle(candidates)
    elif order_key == 'sparse':
        candidates.sort(key=lambda x: -x)
    elif order_key == 'prime_bias':
        def is_prime(n):
            if n < 2:
                return False
            if n < 4:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True
        candidates.sort(key=lambda x: (not is_prime(x), x))
    elif order_key == 'interaction':
        # Sort by how few divisibility conflicts each candidate would create
        # with the current sums. This is dynamic, so we do it iteratively.
        candidates.sort(key=lambda x: x)  # Initial sort, will be refined
    else:
        random.shuffle(candidates)
        
    for i, x in enumerate(candidates):
        if time.time() > deadline:
            break
        res = _can_add(sums_set, x)
        if res is not None:
            sums_set = res
            elements.append(x)
            
    return elements

def _local_search(elements, start_time, deadline):
    """
    Try to improve the set by adding elements and performing 1-for-2 swaps.
    Returns the best set found.
    """
    if not elements:
        return elements
    
    best_elements = list(elements)
    # Rebuild sums for best_elements
    best_sums = set()
    for e in best_elements:
        res = _can_add(best_sums, e)
        if res is None:
            return elements
        best_sums = res
        
    best_size = len(best_elements)
    
    # 1. Try adding more elements greedily
    used = set(best_elements)
    remaining = [x for x in range(1, 257) if x not in used]
    random.shuffle(remaining)
    
    for x in remaining:
        if time.time() > deadline:
            break
        res = _can_add(best_sums, x)
        if res is not None:
            best_sums = res
            best_elements.append(x)
            best_size += 1
            
    # 2. Try 1-for-2 swaps: remove one element, try to add two others
    max_swap_attempts = 30
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for _ in range(max_swap_attempts):
            if time.time() > deadline - 2:
                break
            if len(best_elements) < 2:
                break
            
            rem_idx = random.randint(0, len(best_elements) - 1)
            trial_elements = best_elements[:rem_idx] + best_elements[rem_idx+1:]
            
            trial_sums = set()
            valid = True
            for e in trial_elements:
                res = _can_add(trial_sums, e)
                if res is None:
                    valid = False
                    break
                trial_sums = res
            
            if not valid:
                continue
            
            used_trial = set(trial_elements)
            candidates = [x for x in range(1, 257) if x not in used_trial]
            random.shuffle(candidates)
            
            added_count = 0
            for x in candidates:
                if time.time() > deadline - 2:
                    break
                res = _can_add(trial_sums, x)
                if res is not None:
                    trial_sums = res
                    trial_elements.append(x)
                    added_count += 1
                    if added_count >= 2:
                        break
            
            if len(trial_elements) > len(best_elements):
                best_elements = trial_elements
                best_sums = trial_sums
                best_size = len(best_elements)
                improved = True
                break

    return best_elements

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best_elements = []
    
    # Pool strategies - focus on ranges that tend to work well
    pools = [
        list(range(1, 257)),
        list(range(2, 257)),
        list(range(3, 257)),
        list(range(5, 257)),
        list(range(10, 257)),
        list(range(15, 257)),
        list(range(20, 257)),
        list(range(30, 257)),
        list(range(50, 257)),
        list(range(80, 257)),
        list(range(100, 257)),
        list(range(128, 257)),
    ]
    
    orders = ['asc', 'desc', 'random', 'sparse', 'prime_bias']
    
    # Phase 1: Greedy Construction with multiple pool/order combos
    for pool in pools:
        for order in orders:
            if time.time() > deadline - 25:
                break
            elems = _greedy_construct(pool, order, start_time, deadline)
            if len(elems) > len(best_elements):
                best_elements = elems
                
    # Phase 2: Random Restarts with Local Search
    for _ in range(200):
        if time.time() > deadline - 5:
            break
            
        pool = random.choice(pools)
        if random.random() < 0.4:
            pool = random.sample(pool, k=min(60, len(pool)))
            
        order = random.choice(['random', 'desc', 'sparse', 'asc'])
        elems = _greedy_construct(pool, order, start_time, deadline)
        
        if len(elems) > 0:
            improved_elems = _local_search(elems, start_time, deadline)
            if len(improved_elems) > len(best_elements):
                best_elements = improved_elems
                
        if len(best_elements) > 0 and random.random() < 0.4:
            improved_best = _local_search(best_elements, start_time, deadline)
            if len(improved_best) > len(best_elements):
                best_elements = improved_best

    # Final polish: try to add any remaining elements
    if best_elements and time.time() < deadline - 1:
        used = set(best_elements)
        current_sums = set()
        for e in best_elements:
            res = _can_add(current_sums, e)
            if res:
                current_sums = res
        
        remaining = [x for x in range(1, 257) if x not in used]
        random.shuffle(remaining)
        
        for x in remaining:
            if time.time() > deadline - 0.5:
                break
            res = _can_add(current_sums, x)
            if res is not None:
                current_sums = res
                best_elements.append(x)
                
    return [int(x) for x in best_elements]

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}: {res}")
