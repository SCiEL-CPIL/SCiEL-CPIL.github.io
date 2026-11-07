import time
import random
import sys

DEADLINE_SECONDS = 82.0

def _has_divisibility_conflict(sums):
    """
    Check if any two distinct values in `sums` divide each other.
    Optimized for small sets of integers.
    """
    if len(sums) < 2:
        return False
    
    # Convert to list and sort for efficient bound checking
    arr = sorted(sums)
    n = len(arr)
    
    # If the smallest element a is such that 2*a > max(arr), no conflict possible
    if 2 * arr[0] > arr[-1]:
        return False
        
    # For each a, check if any multiple exists in arr
    # We can use a set for O(1) lookup of multiples
    sum_set = set(arr)
    max_val = arr[-1]
    
    for i in range(n):
        a = arr[i]
        # If 2*a > max_val, no multiples can exist
        if 2 * a > max_val:
            break
        # Check multiples 2a, 3a...
        multiple = 2 * a
        while multiple <= max_val:
            if multiple in sum_set:
                return True
            multiple += a
            
    return False

def _can_add(current_sums_set, current_sums_list, x):
    """
    Check if x can be added to the set defined by current_sums.
    Returns (new_sums_set, new_sums_list) or None.
    """
    # 1. Check distinctness
    # x must not be in current sums
    if x in current_sums_set:
        return None
        
    # New sums are {x} U {s+x for s in current_sums}
    # First check all new sums for collisions with existing sums
    new_sums_added = []
    for s in current_sums_list:
        ns = s + x
        if ns in current_sums_set:
            return None
        new_sums_added.append(ns)
    
    # Check if x collides with any new sum (shouldn't happen since s>0)
    # Check for collisions among new sums themselves
    # s1 + x == s2 + x implies s1 == s2, so no collision among new sums
    
    # Build new sets
    new_sums = set(current_sums_set)
    new_sums.add(x)
    for ns in new_sums_added:
        new_sums.add(ns)
    
    new_sums_list = list(current_sums_list)
    new_sums_list.append(x)
    new_sums_list.extend(new_sums_added)
        
    # 2. Check divisibility
    if _has_divisibility_conflict(new_sums):
        return None
        
    return new_sums, new_sums_list

def _validate_solution(elements):
    """
    Verify that the solution is valid:
    1. All elements are distinct integers in 1..256
    2. No two distinct subset sums divide each other
    """
    if not elements:
        return True
    
    # Check distinctness and range
    elem_set = set(elements)
    if len(elem_set) != len(elements):
        return False
    for e in elements:
        if not isinstance(e, int) or e < 1 or e > 256:
            return False
    
    # Generate all nonempty subset sums
    sums = set()
    n = len(elements)
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums.add(s)
    
    # Check divisibility constraint
    if _has_divisibility_conflict(sums):
        return False
    
    return True

def _greedy_construct(pool, order_key, start_time, deadline):
    """
    Greedy construction.
    order_key: 'asc', 'desc', 'random', 'prime', etc.
    """
    elements = []
    sums_set = set()
    sums_list = []
    
    # Prepare candidates
    candidates = list(pool)
    if order_key == 'asc':
        candidates.sort()
    elif order_key == 'desc':
        candidates.sort(reverse=True)
    elif order_key == 'random':
        random.shuffle(candidates)
    elif order_key == 'sparse':
        # Prioritize numbers that are less likely to create divisibility conflicts
        # Higher numbers tend to have fewer multiples in range
        candidates.sort(key=lambda x: -x)
    elif order_key == 'prime_bias':
        # Bias toward primes (fewer divisors)
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
    elif order_key == 'divisor_count':
        # Sort by number of divisors (ascending - fewer divisors first)
        def count_divisors(n):
            count = 0
            i = 1
            while i * i <= n:
                if n % i == 0:
                    count += 1
                    if i * i != n:
                        count += 1
                i += 1
            return count
        candidates.sort(key=lambda x: count_divisors(x))
    else:
        random.shuffle(candidates)
        
    for x in candidates:
        if time.time() > deadline:
            break
        res = _can_add(sums_set, sums_list, x)
        if res is not None:
            sums_set, sums_list = res
            elements.append(x)
            
    return elements

def _local_search(elements, start_time, deadline, max_iterations=1000):
    """
    Local search: try to add elements or swap elements to improve the solution.
    """
    if not elements:
        return elements
    
    elements = list(elements)
    used = set(elements)
    
    # Rebuild sums
    current_sums_set = set()
    current_sums_list = []
    for e in elements:
        res = _can_add(current_sums_set, current_sums_list, e)
        if res:
            current_sums_set, current_sums_list = res
    
    # Try to add elements
    remaining = [x for x in range(1, 257) if x not in used]
    random.shuffle(remaining)
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # Try adding elements
        for x in remaining:
            if time.time() > deadline - 2:
                break
            res = _can_add(current_sums_set, current_sums_list, x)
            if res is not None:
                current_sums_set, current_sums_list = res
                elements.append(x)
                used.add(x)
                remaining.remove(x)
                improved = True
        
        # Try swapping elements: remove one, try to add two
        if time.time() > deadline - 2:
            break
            
        elements_copy = list(elements)
        for i in range(len(elements_copy)):
            if time.time() > deadline - 2:
                break
            
            # Remove element i
            removed = elements_copy[i]
            new_elements = [e for j, e in enumerate(elements_copy) if j != i]
            
            # Rebuild sums without removed
            test_sums_set = set()
            test_sums_list = []
            valid_rebuild = True
            for e in new_elements:
                res = _can_add(test_sums_set, test_sums_list, e)
                if res is None:
                    valid_rebuild = False
                    break
                test_sums_set, test_sums_list = res
            
            if not valid_rebuild:
                continue
            
            # Try to add two new elements
            candidates = [x for x in range(1, 257) if x not in set(new_elements) and x != removed]
            random.shuffle(candidates)
            
            for x1 in candidates[:20]:  # Limit search
                if time.time() > deadline - 2:
                    break
                res1 = _can_add(test_sums_set, test_sums_list, x1)
                if res1 is None:
                    continue
                sums_set1, sums_list1 = res1
                
                for x2 in candidates[:20]:
                    if x2 == x1:
                        continue
                    if time.time() > deadline - 2:
                        break
                    res2 = _can_add(sums_set1, sums_list1, x2)
                    if res2 is not None:
                        # Success: we can replace removed with x1, x2
                        elements = new_elements + [x1, x2]
                        used = set(elements)
                        current_sums_set, current_sums_list = res2
                        remaining = [x for x in range(1, 257) if x not in used]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return elements

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best_elements = []
    
    # Pool strategies
    pools = [
        list(range(1, 257)),
        list(range(2, 257)),
        list(range(3, 257)),
        list(range(5, 257)),
        list(range(10, 257)),
        list(range(20, 257)),
        list(range(30, 257)),
        list(range(50, 257)),
    ]
    
    orders = ['asc', 'desc', 'random', 'sparse', 'prime_bias', 'divisor_count']
    
    # Run Greedy for each pool and order
    for pool in pools:
        for order in orders:
            if time.time() > deadline - 15:
                break
            elems = _greedy_construct(pool, order, start_time, deadline)
            if len(elems) > len(best_elements):
                best_elements = elems
                
    # Random Restarts with mixed pools
    max_restarts = 5000
    for _ in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        # Pick a random pool with bias toward larger ranges
        r = random.random()
        if r < 0.2:
            pool = list(range(1, 257))
        elif r < 0.4:
            pool = list(range(2, 257))
        elif r < 0.55:
            pool = list(range(5, 257))
        elif r < 0.7:
            pool = list(range(10, 257))
        elif r < 0.85:
            pool = list(range(20, 257))
        else:
            pool = list(range(30, 257))
        
        # Sometimes use a random subset to encourage diversity
        if random.random() < 0.5:
            pool = random.sample(pool, k=min(30, len(pool)))
            
        # Random order selection
        order = random.choice(['random', 'desc', 'sparse', 'divisor_count', 'prime_bias'])
        elems = _greedy_construct(pool, order, start_time, deadline)
        if len(elems) > len(best_elements):
            best_elements = elems
            
    # Local Search / Improvement
    if best_elements and time.time() < deadline - 3:
        improved_elements = _local_search(best_elements, start_time, deadline)
        if len(improved_elements) > len(best_elements):
            best_elements = improved_elements
    
    # Final verification
    if best_elements:
        if not _validate_solution(best_elements):
            # If invalid, fall back to a simple valid solution
            best_elements = []
            # Build a simple valid solution
            test_sums_set = set()
            test_sums_list = []
            for x in range(256, 0, -1):
                res = _can_add(test_sums_set, test_sums_list, x)
                if res is not None:
                    test_sums_set, test_sums_list = res
                    best_elements.append(x)
    
    return list(best_elements)
