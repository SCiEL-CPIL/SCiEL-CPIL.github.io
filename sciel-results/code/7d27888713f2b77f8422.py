import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Prescribed elements
    fixed_elements = [1, 2, 5, 11]
    fixed_set = set(fixed_elements)
    
    # Helper: check if we can add element x to set a (with existing sums)
    def can_add(a, sums, x):
        for y in a:
            s = x + y
            if s in sums:
                return False
        if 2 * x in sums:
            return False
        return True
    
    def get_new_sums(a, x):
        return {x + y for y in a} | {2 * x}
    
    def compute_sums(a):
        s = set()
        for i, x in enumerate(a):
            for y in a[i:]:
                s.add(x + y)
        return s
    
    # Greedy expansion: try to add elements in various orders
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
                sums |= get_new_sums(a, x)
        return a
    
    # Strong local search: try removing 1-2 elements and greedily refilling
    def strong_local_search(a, max_rounds=50):
        a = list(a)
        sums = compute_sums(a)
        a_set = set(a)
        
        for round_num in range(max_rounds):
            if time.time() > deadline:
                break
            
            improved = False
            
            # First, try to add any possible elements
            for cand in range(1, 285):
                if cand in a_set:
                    continue
                if can_add(a, sums, cand):
                    a.append(cand)
                    a_set.add(cand)
                    sums |= get_new_sums(a, cand)
                    improved = True
            
            if not improved:
                # Try removing 1 non-fixed element and refilling
                non_fixed_indices = [i for i in range(len(a)) if a[i] not in fixed_set]
                random.shuffle(non_fixed_indices)
                
                for i in non_fixed_indices[:10]:  # Try up to 10 removals
                    if time.time() > deadline:
                        break
                    
                    removed = a[i]
                    temp_a = [x for j, x in enumerate(a) if j != i]
                    temp_sums = compute_sums(temp_a)
                    temp_set = set(temp_a)
                    
                    # Greedily refill
                    for cand in range(1, 285):
                        if cand in temp_set:
                            continue
                        if can_add(temp_a, temp_sums, cand):
                            temp_a.append(cand)
                            temp_set.add(cand)
                            temp_sums |= get_new_sums(temp_a, cand)
                    
                    if len(temp_a) > len(a):
                        a = temp_a
                        a_set = temp_set
                        sums = temp_sums
                        improved = True
                        break
            
            if not improved:
                # Try removing 2 non-fixed elements and refilling
                non_fixed_indices = [i for i in range(len(a)) if a[i] not in fixed_set]
                if len(non_fixed_indices) >= 2:
                    random.shuffle(non_fixed_indices)
                    # Try a few pairs
                    for trial in range(min(5, len(non_fixed_indices))):
                        if time.time() > deadline:
                            break
                        
                        i1 = non_fixed_indices[trial]
                        # Pick a second index different from i1
                        i2_candidates = [j for j in non_fixed_indices if j != i1]
                        if not i2_candidates:
                            continue
                        i2 = i2_candidates[trial % len(i2_candidates)]
                        
                        # Ensure i1 != i2 after potential index shift
                        # We remove by value to be safe
                        v1 = a[i1]
                        v2 = a[i2]
                        if v1 == v2:
                            continue
                        
                        temp_a = [x for x in a if x != v1 and x != v2]
                        temp_sums = compute_sums(temp_a)
                        temp_set = set(temp_a)
                        
                        # Greedily refill
                        for cand in range(1, 285):
                            if cand in temp_set:
                                continue
                            if can_add(temp_a, temp_sums, cand):
                                temp_a.append(cand)
                                temp_set.add(cand)
                                temp_sums |= get_new_sums(temp_a, cand)
                        
                        if len(temp_a) > len(a):
                            a = temp_a
                            a_set = temp_set
                            sums = temp_sums
                            improved = True
                            break
            
            if not improved:
                break
        
        return a
    
    # Try multiple greedy strategies with different orderings
    best = []
    
    # Strategy 1: Ascending order
    a1 = greedy_expand(fixed_elements, range(12, 285))
    if len(a1) > len(best):
        best = a1
    
    # Strategy 2: Descending order
    a2 = greedy_expand(fixed_elements, range(284, 11, -1))
    if len(a2) > len(best):
        best = a2
    
    # Strategy 3: Randomized greedy with many restarts and varied biases
    random.seed(42)
    n_restarts = 300
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        candidates = list(range(12, 285))
        
        # Different bias strategies
        if restart % 6 == 0:
            # Pure random
            random.shuffle(candidates)
        elif restart % 6 == 1:
            # Bias toward larger values
            candidates.sort(key=lambda x: random.random() * 50 + x)
        elif restart % 6 == 2:
            # Bias toward smaller values
            candidates.sort(key=lambda x: random.random() * 50 - x)
        elif restart % 6 == 3:
            # Gaussian-ish bias around middle
            mid = 148
            candidates.sort(key=lambda x: abs(x - mid) + random.random() * 100)
        elif restart % 6 == 4:
            # Two-phase: first small, then large
            small = [x for x in candidates if x < 150]
            large = [x for x in candidates if x >= 150]
            random.shuffle(small)
            random.shuffle(large)
            candidates = small + large
        else:
            # Two-phase: first large, then small
            small = [x for x in candidates if x < 150]
            large = [x for x in candidates if x >= 150]
            random.shuffle(small)
            random.shuffle(large)
            candidates = large + small
        
        a3 = greedy_expand(fixed_elements, candidates)
        
        # Apply local search to each result
        a3_improved = strong_local_search(a3, max_rounds=20)
        
        if len(a3_improved) > len(best):
            best = a3_improved
    
    # Final polish: apply strong local search to the best
    best = strong_local_search(best, max_rounds=100)
    
    # One more round of random restarts focusing on the neighborhood of the best
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Start from best, remove 1-2 random non-fixed elements, then greedy refill
        a = list(best)
        non_fixed = [x for x in a if x not in fixed_set]
        
        if non_fixed:
            # Remove 1 random element
            idx = random.randint(0, len(non_fixed) - 1)
            removed_val = non_fixed[idx]
            a = [x for x in a if x != removed_val]
        
        # Greedy refill
        sums = compute_sums(a)
        a_set = set(a)
        candidates = list(range(12, 285))
        random.shuffle(candidates)
        for cand in candidates:
            if cand in a_set:
                continue
            if can_add(a, sums, cand):
                a.append(cand)
                a_set.add(cand)
                sums |= get_new_sums(a, cand)
        
        a_improved = strong_local_search(a, max_rounds=30)
        
        if len(a_improved) > len(best):
            best = a_improved
    
    return best
