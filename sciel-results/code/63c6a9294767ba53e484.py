import time
import random
import sys

N = 256

# EVOLVE-BLOCK-START

def _is_valid_solution(arr):
    """
    Verifies:
    1. All elements are distinct integers in [1, N].
    2. It is a Sidon set (all unordered pair sums a+b with a<=b are distinct).
    3. It is maximal (adding any missing element from [1, N] destroys the Sidon property).
    """
    if not arr:
        return False
    
    arr_set = set(arr)
    if len(arr_set) != len(arr):
        return False
    
    for x in arr:
        if x < 1 or x > N:
            return False
            
    # Compute all sums
    sums = set()
    for i in range(len(arr)):
        a = arr[i]
        for j in range(i, len(arr)):
            b = arr[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
            
    # Check maximality
    for x in range(1, N + 1):
        if x in arr_set:
            continue
        
        # Try to add x
        # New sums are {x+a for a in arr} U {2x}
        # If any of these are in `sums`, or if there's a conflict among them, x cannot be added.
        # If x CAN be added (no conflicts), the set is NOT maximal.
        
        # Check conflicts with existing sums
        conflict = False
        for a in arr:
            if (x + a) in sums:
                conflict = True
                break
        if not conflict:
            if (2 * x) in sums:
                conflict = True
        
        if not conflict:
            # x can be added, so the set is not maximal.
            return False
        
    return True


def _greedy_construct(order):
    """
    Greedily constructs a Sidon set by iterating through `order`.
    """
    selected = []
    sums = set()
    
    for x in order:
        conflict = False
        for a in selected:
            if (x + a) in sums:
                conflict = True
                break
        if conflict:
            continue
            
        if (2 * x) in sums:
            continue
            
        selected.append(x)
        for a in selected:
            sums.add(x + a)
        sums.add(2 * x)
        
    return selected


def _make_maximal(arr):
    """
    Given a Sidon set `arr`, add the smallest possible elements from [1, N]
    to make it maximal. Returns the new list (sorted).
    """
    arr = sorted(arr)
    sums = set()
    for i in range(len(arr)):
        a = arr[i]
        for j in range(i, len(arr)):
            b = arr[j]
            sums.add(a + b)
    
    arr_set = set(arr)
    
    # We want to make it maximal. We greedily add the smallest available x that doesn't conflict.
    # This is not guaranteed to be minimal size, but it's a valid heuristic.
    while True:
        is_maximal = True
        for x in range(1, N + 1):
            if x in arr_set:
                continue
            conflict = False
            for a in arr:
                if (x + a) in sums:
                    conflict = True
                    break
            if not conflict:
                if (2 * x) in sums:
                    conflict = True
            if not conflict:
                # x can be added, so we add it (to block this gap)
                arr.append(x)
                arr_set.add(x)
                for a in arr:
                    sums.add(x + a)
                sums.add(2 * x)
                is_maximal = False
                break
        
        if is_maximal:
            break
    
    return sorted(arr)


def _local_search_reduce(current_set, rng, deadline):
    """
    Attempts to reduce the size of a maximal Sidon set by:
    1. Trying to remove an element and repairing maximality.
    2. Comparing sizes.
    """
    current = sorted(current_set)
    best = current[:]
    best_size = len(current)
    
    def get_sums(arr):
        sums = set()
        for i in range(len(arr)):
            a = arr[i]
            for j in range(i, len(arr)):
                b = arr[j]
                s = a + b
                if s in sums:
                    return None  # Not Sidon
                sums.add(s)
        return sums
    
    def is_maximal(arr, sums):
        arr_set = set(arr)
        for x in range(1, N + 1):
            if x in arr_set:
                continue
            conflict = False
            for a in arr:
                if (x + a) in sums:
                    conflict = True
                    break
            if not conflict and (2 * x) in sums:
                conflict = True
            if not conflict:
                return False
        return True
    
    # Main loop: try to remove elements
    max_attempts = 500
    attempts = 0
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        arr = sorted(best)
        sums = get_sums(arr)
        if sums is None:
            continue
        
        if not is_maximal(arr, sums):
            # Should not happen if best is maintained validly, but just in case
            best = _make_maximal(arr)
            best_size = len(best)
            continue
        
        # Try removing a random element
        if len(arr) > 2:
            idx = rng.randrange(len(arr))
            removed = arr[idx]
            new_arr = [x for x in arr if x != removed]
            
            # Check if new_arr is still Sidon
            new_sums = get_sums(new_arr)
            if new_sums is None:
                continue
            
            # Try to make it maximal again
            repaired = _make_maximal(new_arr)
            
            if len(repaired) < best_size:
                best = repaired
                best_size = len(repaired)
                
            # If repaired is same size, we might keep it for diversity, but best_size won't improve
            # If repaired is larger, discard
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best_set = []
    best_size = float('inf')
    
    rng = random.Random(12345)
    
    # Phase 1: Deterministic Greedy Seeds
    # Descending order tends to produce larger elements, which might block more sums?
    # Or ascending produces small elements which block many sums involving small numbers?
    # Let's try both.
    
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        s = _greedy_construct(order)
        s = _make_maximal(s)
        if len(s) < best_size:
            best_size = len(s)
            best_set = s[:]
            
    # Phase 2: Randomized Greedy
    # Use random shuffles to find diverse starting points.
    while time.time() < deadline - 10:  # Leave 10s for local search
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        s = _greedy_construct(order)
        s = _make_maximal(s)
        
        if len(s) < best_size:
            best_size = len(s)
            best_set = s[:]
            
            # If we find a very small set, we can break or continue.
            # Let's continue to see if we can do better.
            if best_size <= 12:
                # 12 is quite small, maybe stop early? 
                # But let's use the remaining time for local search on this best.
                # Actually, let's just break to local search if we've found something very good.
                # But maybe we can find 11?
                # Let's keep searching a bit more.
                pass
    
    # Phase 3: Local Search to Reduce
    if time.time() < deadline and best_set:
        # Run local search on the best set found
        improved = _local_search_reduce(best_set, rng, deadline)
        if len(improved) < best_size:
            best_size = len(improved)
            best_set = improved[:]
            
    # Final Verification
    if best_set and _is_valid_solution(best_set):
        return best_set
    else:
        # Fallback: Return the result of the first greedy run
        s = _greedy_construct(list(range(1, N + 1)))
        s = _make_maximal(s)
        return s

# EVOLVE-BLOCK-END
