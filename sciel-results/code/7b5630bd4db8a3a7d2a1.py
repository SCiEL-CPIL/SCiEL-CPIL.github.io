def construct():
    from fractions import Fraction
    import time
    import random
    
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    fractions = [Fraction(1, d) for d in denominators]
    
    def compute_sum(indices):
        s = Fraction(0, 1)
        for i in indices:
            s += fractions[i]
        return s
    
    def deficit(indices):
        return 1 - compute_sum(indices)
    
    # Greedy initialization: add in order while sum < 1
    def greedy():
        selected = []
        total = Fraction(0, 1)
        for i in range(n):
            if total + fractions[i] < 1:
                selected.append(i)
                total += fractions[i]
        return selected
    
    best = greedy()
    best_def = deficit(best)
    best_sum = 1 - best_def
    
    # Local search
    def local_search(current, max_iters=500000):
        current = list(current)
        cur_sum = compute_sum(current)
        cur_def = 1 - cur_sum
        current_set = set(current)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try single swaps: remove one, add one
            for i in list(current):
                new_sum = cur_sum - fractions[i]
                for j in range(n):
                    if j not in current_set:
                        candidate_sum = new_sum + fractions[j]
                        if candidate_sum < 1:
                            candidate_def = 1 - candidate_sum
                            if candidate_def < cur_def:
                                # Apply swap
                                current.remove(i)
                                current.append(j)
                                current_set.remove(i)
                                current_set.add(j)
                                cur_sum = candidate_sum
                                cur_def = candidate_def
                                improved = True
                                break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try double swaps: remove two, add two
            found_double = False
            for i_idx in range(len(current)):
                i = current[i_idx]
                for j_idx in range(i_idx + 1, len(current)):
                    j = current[j_idx]
                    new_sum = cur_sum - fractions[i] - fractions[j]
                    for a in range(n):
                        if a not in current_set:
                            for b in range(a + 1, n):
                                if b not in current_set and b != a:
                                    candidate_sum = new_sum + fractions[a] + fractions[b]
                                    if candidate_sum < 1:
                                        candidate_def = 1 - candidate_sum
                                        if candidate_def < cur_def:
                                            # Apply double swap
                                            current.remove(i)
                                            current.remove(j)
                                            current.append(a)
                                            current.append(b)
                                            current_set.remove(i)
                                            current_set.remove(j)
                                            current_set.add(a)
                                            current_set.add(b)
                                            cur_sum = candidate_sum
                                            cur_def = candidate_def
                                            found_double = True
                                            break
                                if found_double:
                                    break
                            if found_double:
                                break
                    if found_double:
                        break
                if found_double:
                    break
            
            if found_double:
                continue
            
            # Try single addition
            added = False
            for j in range(n):
                if j not in current_set:
                    candidate_sum = cur_sum + fractions[j]
                    if candidate_sum < 1:
                        candidate_def = 1 - candidate_sum
                        if candidate_def < cur_def:
                            current.append(j)
                            current_set.add(j)
                            cur_sum = candidate_sum
                            cur_def = candidate_def
                            added = True
                            break
            if added:
                continue
            
            # No improvement
            break
        
        return current, cur_def
    
    best, best_def = local_search(best)
    
    # Try a few random restarts
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Random subset initialization
        random.shuffle(denominators)
        selected = []
        total = Fraction(0, 1)
        for i in range(n):
            if total + fractions[i] < 1:
                selected.append(i)
                total += fractions[i]
        
        if len(selected) > 0:
            candidate_def = 1 - total
            if candidate_def < best_def:
                best, best_def = local_search(selected)
    
    return best
