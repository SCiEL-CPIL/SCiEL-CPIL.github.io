def construct():
    import time
    from fractions import Fraction
    
    # Deadline guard: 90s wall-clock limit, use 85s for safety
    deadline = time.time() + 85
    
    denominators = list(range(2, 49))
    n = len(denominators)
    # Exact reciprocals
    recip = [Fraction(1, d) for d in denominators]
    ONE = Fraction(1, 1)
    
    # Greedy initialization: add largest reciprocals first
    # Sort indices by reciprocal value descending
    order = sorted(range(n), key=lambda i: -recip[i])
    selected = set()
    current_sum = Fraction(0, 1)
    
    for i in order:
        # Check if adding i keeps sum strictly below 1
        if current_sum + recip[i] < ONE:
            selected.add(i)
            current_sum += recip[i]
    
    best_sum = current_sum
    best_set = frozenset(selected)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1-exchange: add one item
        for i in range(n):
            if i not in selected:
                new_sum = current_sum + recip[i]
                if new_sum < ONE and new_sum > best_sum:
                    selected.add(i)
                    current_sum = new_sum
                    best_sum = new_sum
                    best_set = frozenset(selected)
                    improved = True
        
        # 2-exchange: remove one, add one
        found_swap = False
        for i in list(selected):
            if time.time() > deadline:
                break
            for j in range(n):
                if j not in selected:
                    new_sum = current_sum - recip[i] + recip[j]
                    if new_sum < ONE and new_sum > best_sum:
                        selected.discard(i)
                        selected.add(j)
                        current_sum = new_sum
                        best_sum = new_sum
                        best_set = frozenset(selected)
                        improved = True
                        found_swap = True
                        break
            if found_swap:
                break
        
        # 3-exchange: remove two, add two
        # Only do this if we haven't improved in previous steps and have time.
        if not improved and time.time() < deadline - 5:
            found_2swap = False
            sel_list = list(selected)
            for i_idx in range(len(sel_list)):
                if time.time() > deadline - 5:
                    break
                i = sel_list[i_idx]
                for j_idx in range(i_idx + 1, len(sel_list)):
                    j = sel_list[j_idx]
                    sum_after_rem = current_sum - recip[i] - recip[j]
                    
                    for k in range(n):
                        if k in selected:
                            continue
                        sum_after_add1 = sum_after_rem + recip[k]
                        if sum_after_add1 >= ONE:
                            continue
                        
                        for l in range(n):
                            if l in selected or l == k:
                                continue
                            new_sum = sum_after_add1 + recip[l]
                            if new_sum < ONE and new_sum > best_sum:
                                selected.discard(i)
                                selected.discard(j)
                                selected.add(k)
                                selected.add(l)
                                current_sum = new_sum
                                best_sum = new_sum
                                best_set = frozenset(selected)
                                improved = True
                                found_2swap = True
                                break
                        if found_2swap:
                            break
                    if found_2swap:
                        break
                if found_2swap:
                    break

    # Final safety check: ensure the returned set is valid
    final_sum = sum((recip[i] for i in best_set), Fraction(0, 1))
    if final_sum >= ONE or len(best_set) == 0:
        # Fallback: return a known valid small set, e.g., {2, 3} -> 1/2+1/3=5/6 < 1
        # Indices: 2 is index 0, 3 is index 1
        return [0, 1]
        
    return sorted(best_set)
