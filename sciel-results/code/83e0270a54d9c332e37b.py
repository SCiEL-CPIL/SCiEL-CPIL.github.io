import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    fixed_elements = [1, 2, 5, 11]
    fixed_set = set(fixed_elements)
    
    def compute_sums(a):
        s = set()
        for i, x in enumerate(a):
            for y in a[i:]:
                s.add(x + y)
        return s
        
    def can_add(a, sums, x):
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
            non_fixed = [x for x in a if x not in fixed_set]
            if not non_fixed:
                break
                
            random.shuffle(non_fixed)
            best_swap = None
            best_size = len(a)
            
            for y in non_fixed[:15]:
                temp_a = [x for x in a if x != y]
                temp_sums = compute_sums(temp_a)
                temp_set = set(temp_a)
                
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
                            best_swap = (list(temp_a), set(temp_sums), set(temp_set))
                
                if best_size > len(a):
                    a, sums, a_set = best_swap
                    improved = True
                    break
            
            if not improved:
                # Try 2-for-3? Too slow.
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
    
    for restart in range(200):
        if time.time() > deadline:
            break
            
        candidates = list(range(12, 285))
        random.shuffle(candidates)
        
        a3 = greedy_expand(fixed_elements, candidates)
        a3 = local_search(a3, max_rounds=80, deadline=deadline)
        
        if len(a3) > len(best):
            best = a3
            if len(best) >= 20:
                break
                
    # Final deep polish
    if time.time() < deadline - 5:
        best = local_search(best, max_rounds=1000, deadline=deadline)
        
    return best
