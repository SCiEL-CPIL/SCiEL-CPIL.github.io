import time
import random
import sys
from collections import Counter

DEADLINE_SECONDS = 85.0  # Safety margin for 90s limit

def _has_divisibility_conflict(sums):
    """
    Check if any two distinct values in `sums` divide each other.
    Uses a sieve-like approach for efficiency.
    """
    if len(sums) < 2:
        return False
    
    max_val = max(sums)
    # Create a boolean array for quick lookup
    present = [False] * (max_val + 1)
    for s in sums:
        present[s] = True
    
    # For each possible divisor d, check if d and any multiple of d are both present
    for d in range(1, max_val // 2 + 1):
        if not present[d]:
            continue
        # Check multiples 2d, 3d, ...
        m = 2 * d
        while m <= max_val:
            if present[m]:
                return True
            m += d
    
    return False

def _can_add(current_sums_set, x):
    """
    Check if x can be added to the set defined by current_sums_set.
    Returns (new_sums_set) or None.
    """
    if x in current_sums_set:
        return None
        
    # Check if any new sum s+x already exists in current_sums_set
    # This would mean two different subsets have the same sum
    for s in current_sums_set:
        if s + x in current_sums_set:
            return None
    
    new_sums = set(current_sums_set)
    new_sums.add(x)
    for s in current_sums_set:
        new_sums.add(s + x)
    
    if _has_divisibility_conflict(new_sums):
        return None
        
    return new_sums

def _greedy_construct(pool, order_key, start_time, deadline):
    """
    Greedy construction.
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
    else:
        random.shuffle(candidates)
        
    for x in candidates:
        if time.time() > deadline:
            break
        res = _can_add(sums_set, x)
        if res is not None:
            sums_set = res
            elements.append(x)
            
    return elements

def _local_search(elements, start_time, deadline):
    """
    Try to improve the set by swapping elements or adding new ones.
    """
    if not elements:
        return elements
    
    best_elements = list(elements)
    best_sums = set()
    for e in best_elements:
        res = _can_add(best_sums, e)
        if res is None:
            return elements
        best_sums = res
        
    best_size = len(best_elements)
    
    # 1. Try adding more elements
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
            
    # 2. Local Search: Swap one element for another to potentially allow more additions
    max_swap_attempts = 100
    for _ in range(max_swap_attempts):
        if time.time() > deadline - 1:
            break
            
        if len(best_elements) < 2:
            break
            
        rem_idx = random.randint(0, len(best_elements) - 1)
        rem_elem = best_elements[rem_idx]
        
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
            if time.time() > deadline - 1:
                break
            res = _can_add(trial_sums, x)
            if res is not None:
                trial_sums = res
                trial_elements.append(x)
                added_count += 1
                if added_count > 1:
                    break
                
        if len(trial_elements) > len(best_elements):
            best_elements = trial_elements
            best_sums = trial_sums
            best_size = len(best_elements)
            max_swap_attempts = 100 

    return best_elements

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best_elements = []
    best_sums = set()
    
    pools = [
        list(range(1, 257)),
        list(range(2, 257)),
        list(range(3, 257)),
        list(range(5, 257)),
        list(range(10, 257)),
        list(range(20, 257)),
        list(range(50, 257)),
        list(range(100, 257)),
    ]
    
    orders = ['asc', 'desc', 'random', 'sparse', 'prime_bias']
    
    # Phase 1: Systematic greedy construction with various strategies
    for pool in pools:
        for order in orders:
            if time.time() > deadline - 30:
                break
            elems = _greedy_construct(pool, order, start_time, deadline)
            if len(elems) > len(best_elements):
                best_elements = elems
                best_sums = set()
                for e in best_elements:
                    res = _can_add(best_sums, e)
                    if res:
                        best_sums = res
    
    # Phase 2: Random restarts with local search
    max_restarts = 150
    for _ in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        pool = random.choice(pools)
        if random.random() < 0.4:
            pool = random.sample(pool, k=random.randint(30, 80))
            
        order = random.choice(['random', 'desc', 'sparse', 'asc'])
        elems = _greedy_construct(pool, order, start_time, deadline)
        
        if len(elems) > 0:
            improved_elems = _local_search(elems, start_time, deadline)
            if len(improved_elems) > len(best_elements):
                best_elements = improved_elems
                best_sums = set()
                for e in best_elements:
                    res = _can_add(best_sums, e)
                    if res:
                        best_sums = res
    
    # Phase 3: Intensive local search on the best solution
    if best_elements and time.time() < deadline - 3:
        improved_best = _local_search(best_elements, start_time, deadline)
        if len(improved_best) > len(best_elements):
            best_elements = improved_best
            best_sums = set()
            for e in best_elements:
                res = _can_add(best_sums, e)
                if res:
                    best_sums = res
    
    # Phase 4: Final polish - try to add any remaining elements
    if best_elements and time.time() < deadline - 0.5:
        used = set(best_elements)
        current_sums = set()
        for e in best_elements:
            res = _can_add(current_sums, e)
            if res:
                current_sums = res
        
        remaining = [x for x in range(1, 257) if x not in used]
        random.shuffle(remaining)
        
        for x in remaining:
            if time.time() > deadline - 0.2:
                break
            res = _can_add(current_sums, x)
            if res is not None:
                current_sums = res
                best_elements.append(x)
                
    return [int(x) for x in best_elements]

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}: {res}")
