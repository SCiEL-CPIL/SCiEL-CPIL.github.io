import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s kill
    
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
    reverse_conflicts = {x: [] for x in range(1, limit + 1)}
    for a in range(1, limit + 1):
        for b, c in conflicts[a].items():
            reverse_conflicts[c].append((a, b))
    
    def can_add(x, S_set):
        """Check if x can be added to S_set without creating a forbidden triple."""
        # Check if any pair (a,b) in S_set has x as forbidden third element
        for a, b in reverse_conflicts[x]:
            if a in S_set and b in S_set:
                return False
        # Check pairs involving x: for each y in S_set, verify no conflict
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
    
    # Random orderings
    rng = random.Random(42)
    for _ in range(8):
        order = list(range(1, limit + 1))
        rng.shuffle(order)
        orderings.append(order)
    
    # Additional structured orderings
    # Even numbers first, then odd
    evens = list(range(2, limit + 1, 2))
    odds = list(range(1, limit + 1, 2))
    orderings.append(evens + odds)
    orderings.append(odds + evens)
    
    # Multiples of 3 first
    mult3 = list(range(3, limit + 1, 3))
    non3 = [x for x in range(1, limit + 1) if x % 3 != 0]
    orderings.append(mult3 + non3)
    
    for ordering in orderings:
        if time.time() > deadline - 10:
            break
        
        S_set = set()
        for a in ordering:
            if can_add(a, S_set):
                S_set.add(a)
        
        # Fill in gaps
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
    
    # Phase 2: 1-for-1 swaps (Optimized)
    # We limit the number of swaps to prevent long runs
    max_swaps = 50
    swaps_done = 0
    while swaps_done < max_swaps and time.time() < deadline - 5:
        improved = False
        # Shuffle to avoid bias
        remove_candidates = list(S_set)
        rng.shuffle(remove_candidates)
        for remove_val in remove_candidates:
            if time.time() > deadline - 5:
                break
            S_set.remove(remove_val)
            # Try to add any element that fits
            add_candidates = [a for a in range(1, limit + 1) if a not in S_set]
            rng.shuffle(add_candidates)
            for a in add_candidates:
                if can_add(a, S_set):
                    S_set.add(a)
                    improved = True
                    swaps_done += 1
                    break
            if not improved:
                S_set.add(remove_val)
            if improved:
                break
    
    # Final verification and cleanup
    final_set = set()
    for a in sorted(S_set):
        if can_add(a, final_set):
            final_set.add(a)
    
    # Try to add more after cleanup
    for a in range(1, limit + 1):
        if a not in final_set and can_add(a, final_set):
            final_set.add(a)
    
    return list(final_set)
