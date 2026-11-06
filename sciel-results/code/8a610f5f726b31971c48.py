import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit (using 55s to be safe)

    MASK = (1 << 201) - 1
    TARGET_BIT = 1 << 200

    def is_valid(s):
        """
        Check if any subset of s sums to 200 using bitset DP.
        Returns True if valid (no sum 200), False otherwise.
        """
        reachable = 1
        for x in s:
            reachable |= (reachable << x)
            if reachable & TARGET_BIT:
                return False
        return True

    def find_conflict(s, target=200):
        """
        Find one subset of s that sums to target.
        Returns a list of elements in the subset, or None if no such subset exists.
        """
        dp = [False] * (target + 1)
        parent = [None] * (target + 1)
        dp[0] = True
        
        for x in s:
            for j in range(target, x - 1, -1):
                if dp[j - x] and not dp[j]:
                    dp[j] = True
                    parent[j] = x
        
        if not dp[target]:
            return None
        
        conflict = []
        j = target
        while j > 0:
            x = parent[j]
            if x is None:
                break
            conflict.append(x)
            j -= x
        return conflict

    # --- Phase 1: Greedy Construction ---
    # Try multiple orderings and keep the best result
    best_set = set()
    best_size = 0
    
    # Ordering 1: Descending (100, 99, ..., 1)
    # This tends to pick large numbers which are less likely to form small sums, 
    # but can form 200 with other large numbers.
    # Actually, for target 200, picking large numbers > 100 first is safe (sum > 200 with any other positive).
    # Numbers 100 down to 1.
    
    def greedy_construct(order):
        current = set()
        reachable = 1
        for x in order:
            if time.time() > deadline:
                break
            new_reachable = reachable | (reachable << x)
            if not (new_reachable & TARGET_BIT):
                current.add(x)
                reachable = new_reachable & MASK
        return current

    # Try a few different orderings
    orderings = [
        list(range(100, 0, -1)),       # Descending
        list(range(1, 101)),           # Ascending
        list(range(100, 0, -2)) + list(range(99, 0, -2)), # Large evens, large odds
        list(range(2, 101, 2)) + list(range(1, 100, 2)),  # Evens then odds
    ]
    
    for order in orderings:
        if time.time() > deadline - 5.0:
            break
        candidate = greedy_construct(order)
        if len(candidate) > best_size:
            best_size = len(candidate)
            best_set = candidate

    # --- Phase 2: Local Search / Swap ---
    # Start with the best greedy result
    current_set = best_set
    current_size = best_size
    
    # Try to add back excluded elements
    all_elements = set(range(1, 101))
    
    # Rebuild reachable set for current_set
    def get_reachable(s):
        r = 1
        for x in s:
            r |= (r << x)
        return r & MASK

    reachable = get_reachable(current_set)
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        # Try to add any excluded element
        for x in sorted(all_elements - current_set):
            if time.time() > deadline - 2.0:
                break
            new_reachable = reachable | (reachable << x)
            if not (new_reachable & TARGET_BIT):
                current_set.add(x)
                reachable = new_reachable
                current_size += 1
                improved = True
        
        if not improved:
            # Try swaps: remove y, add x
            # We want to maximize size, so swaps keep size constant. 
            # However, a swap might allow a subsequent addition that wasn't possible before.
            # So we perform swaps to reach a different local optimum, then try adding again.
            
            # To avoid infinite loops, we limit the number of swaps or use a specific heuristic.
            # Let's try to find a swap that allows adding a new element.
            # Or simply do a limited number of random/deterministic swaps to escape local minima.
            
            # Deterministic swap heuristic: 
            # Try to swap a larger element y in S for a smaller element x not in S.
            # This might free up "sum space" for smaller elements.
            
            swapped = False
            for x in sorted(all_elements - current_set):
                if time.time() > deadline - 2.0:
                    break
                for y in sorted(current_set, reverse=True):
                    if time.time() > deadline - 2.0:
                        break
                    if x >= y:
                        continue
                    # Try swap
                    candidate_set = (current_set - {y}) | {x}
                    # Check validity
                    # Since we just swapped, we can check quickly?
                    # Just use is_valid for safety, or update reachable.
                    # is_valid is O(N * 200 / word_size) which is very fast.
                    if is_valid(candidate_set):
                        # Accept swap
                        current_set = candidate_set
                        reachable = get_reachable(current_set)
                        swapped = True
                        improved = True
                        break
                if swapped:
                    break
            
    # --- Phase 3: Final Verification ---
    if not is_valid(current_set):
        # Fallback to a known valid set
        # {100, 99, ..., 66} is valid?
        # Min 3-sum: 66+67+68 = 201. Max 2-sum: 100+99=199.
        # Yes, {66..100} is valid. Size 35.
        fallback = set(range(66, 101))
        if len(fallback) > len(current_set):
            current_set = fallback
        else:
            # Try to find any valid set
            # Single element is always valid
            current_set = {1}
            
    return list(current_set)
