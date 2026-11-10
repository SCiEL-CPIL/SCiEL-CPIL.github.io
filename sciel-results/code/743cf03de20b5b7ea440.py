import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    limit = 180
    
    # Precompute conflicts: for each a in [1, limit], for each b in (a, limit],
    # compute c = ab/(b-a). If c is integer, c > a, c != b, c <= limit,
    # then (a, b, c) is a forbidden triple.
    # conflicts[a][b] = c means if a and b are in S, then c must NOT be in S.
    
    conflicts = {}
    for a in range(1, limit + 1):
        conflicts[a] = {}
        for b in range(a + 1, limit + 1):
            num = a * b
            den = b - a
            if num % den == 0:
                c = num // den
                if c > a and c != b and c <= limit:
                    conflicts[a][b] = c
    
    # Reverse index: for each element x, store list of (a, b) pairs where x is the forbidden c
    # i.e., if a and b are both in S, then x cannot be in S
    reverse_conflicts = {x: [] for x in range(1, limit + 1)}
    for a in range(1, limit + 1):
        for b, c in conflicts[a].items():
            reverse_conflicts[c].append((a, b))
    
    def can_add(x, S_set):
        """Check if x can be added to S_set without creating a forbidden triple."""
        # For x to be addable, no pair (a, b) in S_set should have x as their forbidden c
        # AND no pair involving x should have the third element in S_set
        for a, b in reverse_conflicts[x]:
            if a in S_set and b in S_set:
                return False
        # Also check: for each y in S_set, if x and y form a pair, check the third element
        for y in S_set:
            if y > x:
                if y in conflicts[x] and conflicts[x][y] in S_set:
                    return False
            elif y < x:
                if x in conflicts[y] and conflicts[y][x] in S_set:
                    return False
        return True
    
    # Multiple greedy restarts with different orderings
    best_set = set()
    best_size = 0
    
    orderings = [
        list(range(limit, 0, -1)),  # largest to smallest
        list(range(1, limit + 1)),  # smallest to largest
    ]
    
    # Build alternating ordering
    left, right = 1, limit
    alt_order = []
    while left <= right:
        alt_order.append(left)
        left += 1
        if left <= right:
            alt_order.append(right)
            right -= 1
    orderings.append(alt_order)
    
    # Random-ish orderings using deterministic pseudo-random
    import random
    rng = random.Random(42)
    for _ in range(8):
        order = list(range(1, limit + 1))
        rng.shuffle(order)
        orderings.append(order)
    
    # Weighted orderings: bias toward certain ranges
    for seed in [1, 2, 3, 4]:
        rng2 = random.Random(seed)
        order = list(range(1, limit + 1))
        # Sort by random key with bias
        rng2.shuffle(order)
        orderings.append(order)
    
    for ordering in orderings:
        if time.time() > deadline - 10:
            break
        
        S_set = set()
        for a in ordering:
            if can_add(a, S_set):
                S_set.add(a)
        
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_set = set(S_set)
        
        # Try to add any remaining elements (fill in gaps)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for a in range(1, limit + 1):
                if a in S_set:
                    continue
                if can_add(a, S_set):
                    S_set.add(a)
                    improved = True
        
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_set = set(S_set)
    
    # Local search on the best set found
    S_set = set(best_set)
    
    # Phase 1: Add any element that can be added
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for a in range(1, limit + 1):
            if a in S_set:
                continue
            if can_add(a, S_set):
                S_set.add(a)
                improved = True
    
    # Phase 2: 1-for-1 swaps
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        current_size = len(S_set)
        
        for remove_val in list(S_set):
            if time.time() > deadline - 10:
                break
            S_set.remove(remove_val)
            # Try to add a different element
            for a in range(1, limit + 1):
                if a in S_set or a == remove_val:
                    continue
                if can_add(a, S_set):
                    S_set.add(a)
                    improved = True
                    break
            
            if not improved:
                S_set.add(remove_val)
            
            if improved:
                break
    
    # Phase 3: 1-for-2 swaps (remove one, try to add two)
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        current_size = len(S_set)
        
        for remove_val in list(S_set):
            if time.time() > deadline - 15:
                break
            S_set.remove(remove_val)
            
            # Try to find two elements that can both be added
            added = []
            for a1 in range(1, limit + 1):
                if a1 in S_set or a1 == remove_val:
                    continue
                if can_add(a1, S_set):
                    S_set.add(a1)
                    added.append(a1)
                    # Now try to add one more
                    for a2 in range(1, limit + 1):
                        if a2 in S_set or a2 == remove_val or a2 == a1:
                            continue
                        if can_add(a2, S_set):
                            S_set.add(a2)
                            added.append(a2)
                            improved = True
                            break
                    break
            
            if not improved:
                # Undo additions
                for a in added:
                    S_set.remove(a)
                S_set.add(remove_val)
            
            if improved:
                break
    
    # Phase 4: Try 2-for-1 swaps (remove two, add one) - only if it increases size
    # This won't increase size, so skip unless we're stuck
    
    # Final cleanup: verify the set is valid by rebuilding
    final_set = set()
    for a in sorted(S_set):
        if can_add(a, final_set):
            final_set.add(a)
    
    # Try to add more after cleanup
    for a in range(1, limit + 1):
        if a not in final_set and can_add(a, final_set):
            final_set.add(a)
    
    return list(final_set)
