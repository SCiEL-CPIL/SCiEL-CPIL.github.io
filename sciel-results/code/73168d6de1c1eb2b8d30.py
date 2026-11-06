import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    fixed_elements = [1, 2, 5, 11]
    fixed_set = set(fixed_elements)
    
    def compute_sums(a):
        s = set()
        for i, x in enumerate(a):
            for y in a[i:]:
                s.add(x + y)
        return s
        
    def can_add(a, sums, x):
        # Check if x can be added to a
        # New sums will be x+y for y in a, and 2x
        # Conflict if any new sum is in sums, or if new sums collide with each other (only 2x can collide with x+y if 2x == x+y => x==y, but x not in a)
        # Wait, x+y1 == x+y2 => y1==y2, so no internal collision among x+y's.
        # Only collision is 2x == x+y => x==y (impossible) or 2x in sums.
        # And x+y in sums.
        for y in a:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    def greedy_expand(initial_list, order):
        a = list(initial_list)
        sums = compute_sums(a)
        a_set = set(a)
        
        for x in order:
            if x in a_set:
                continue
            if can_add(a, sums, x):
                a.append(x)
                a_set.add(x)
                # Update sums
                for y in a:
                    sums.add(x + y)
                sums.add(2 * x)
                
        return a

    def local_search(a, max_rounds=1000, deadline=None):
        a = list(a)
        sums = compute_sums(a)
        a_set = set(a)
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            improved = False
            
            # 1. Try to add any valid element
            added = False
            # Randomize candidate order to avoid bias
            candidates = [x for x in range(1, 285) if x not in a_set]
            random.shuffle(candidates)
            
            for x in candidates:
                if can_add(a, sums, x):
                    a.append(x)
                    a_set.add(x)
                    for y in a:
                        sums.add(x + y)
                    sums.add(2 * x)
                    added = True
                    improved = True
                    break
                    
            if added:
                continue
            
            # 2. Try 1-for-1 swap (remove y, add x) to potentially unblock additions
            # Or just try to remove one element and see if we can add two
            # Actually, if no single add works, we are in a local max for size.
            # To increase size, we must remove some and add more.
            # Try removing 1 non-fixed element y, then try to add as many as possible.
            # If we can add 2 or more, net gain > 0.
            
            non_fixed = [x for x in a if x not in fixed_set]
            if not non_fixed:
                break
                
            random.shuffle(non_fixed)
            best_swap = None
            best_size = len(a)
            
            # Try a few random removals
            for y in non_fixed[:10]: # Limit trials per round for speed
                temp_a = [x for x in a if x != y]
                temp_sums = compute_sums(temp_a)
                temp_set = set(temp_a)
                
                # Try to refill
                added_count = 0
                candidates = [x for x in range(1, 285) if x not in temp_set]
                random.shuffle(candidates)
                
                for x in candidates:
                    if can_add(temp_a, temp_sums, x):
                        temp_a.append(x)
                        temp_set.add(x)
                        for z in temp_a:
                            temp_sums.add(x + z)
                        temp_sums.add(2 * x)
                        added_count += 1
                        if len(temp_a) > best_size:
                            best_size = len(temp_a)
                            best_swap = (temp_a, temp_sums, temp_set)
                            # If we already gained, we can keep trying to add more
                            # But for speed, maybe stop if we got +2?
                            # Let's just let it run, it's fast.
                
                if best_size > len(a):
                    a, sums, a_set = best_swap
                    improved = True
                    break
            
            if not improved:
                # Try 2-for-3? Too slow.
                # Try random restart of local search logic?
                break
                
        return a

    # Main search logic
    best = []
    
    # Strategy 1: Greedy Ascending
    a1 = greedy_expand(fixed_elements, range(12, 285))
    if len(a1) > len(best):
        best = a1
        
    # Strategy 2: Greedy Descending
    a2 = greedy_expand(fixed_elements, range(284, 11, -1))
    if len(a2) > len(best):
        best = a2
        
    # Strategy 3: Randomized Greedy + Local Search
    random.seed(42)
    
    # Run multiple restarts
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Random order
        candidates = list(range(12, 285))
        random.shuffle(candidates)
        
        a3 = greedy_expand(fixed_elements, candidates)
        
        # Polish with local search
        a3 = local_search(a3, max_rounds=50, deadline=deadline)
        
        if len(a3) > len(best):
            best = a3
            if len(best) >= 20:
                break
                
    # Final deep polish
    if time.time() < deadline - 10:
        best = local_search(best, max_rounds=500, deadline=deadline)
        
    return best
