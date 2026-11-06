def construct():
    import time
    from fractions import Fraction
    
    start_time = time.time()
    deadline = start_time + 75
    
    denominators = list(range(2, 49))
    n = len(denominators)
    recip = [Fraction(1, d) for d in denominators]
    
    best_set = None
    best_sum = Fraction(0)
    
    def update_best(s, total):
        nonlocal best_set, best_sum
        if total < 1 and (best_set is None or total > best_sum):
            best_set = list(s)
            best_sum = total
    
    # Suffix sums
    suffix = [Fraction(0)] * (n + 1)
    for i in range(n-1, -1, -1):
        suffix[i] = suffix[i+1] + recip[i]
    
    # Greedy initial
    s = []
    total = Fraction(0)
    for i in range(n):
        if total + recip[i] < 1:
            s.append(i)
            total += recip[i]
    if s:
        update_best(s, total)
    
    # Branch and bound
    def dfs(idx, current_set, current_sum):
        nonlocal best_set, best_sum
        if time.time() >= deadline:
            return
        
        if idx == n:
            if current_sum < 1:
                update_best(current_set, current_sum)
            return
        
        # Prune: can't beat best even with all remaining
        if current_sum + suffix[idx] <= best_sum:
            return
        
        # Take all remaining if stays below 1
        if current_sum + suffix[idx] < 1:
            full_set = current_set + list(range(idx, n))
            full_sum = current_sum + suffix[idx]
            update_best(full_set, full_sum)
            return
        
        # Branch: try taking first (for early good solution)
        new_sum = current_sum + recip[idx]
        if new_sum < 1:
            dfs(idx+1, current_set + [idx], new_sum)
        
        # Skip
        dfs(idx+1, current_set, current_sum)
    
    dfs(0, [], Fraction(0))
    
    # Local search refinement
    if best_set and time.time() < deadline - 5:
        s = set(best_set)
        total = best_sum
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # 1-add
            for j in range(n):
                if j not in s:
                    nt = total + recip[j]
                    if nt < 1 and nt > total:
                        s.add(j)
                        total = nt
                        improved = True
                        update_best(s, total)
                        break
            if improved:
                continue
            # 1-swap
            best_swap = None
            best_nt = total
            for i in s:
                ri = recip[i]
                for j in range(n):
                    if j not in s:
                        nt = total - ri + recip[j]
                        if nt < 1 and nt > best_nt:
                            best_nt = nt
                            best_swap = (i, j)
            if best_swap:
                i, j = best_swap
                s.remove(i)
                s.add(j)
                total = best_nt
                improved = True
                update_best(s, total)
            if improved:
                continue
            # 2-swap
            s_list = list(s)
            found = False
            for i1_idx in range(len(s_list)):
                for i2_idx in range(i1_idx+1, len(s_list)):
                    i1, i2 = s_list[i1_idx], s_list[i2_idx]
                    base = total - recip[i1] - recip[i2]
                    best_pair = None
                    best_nt2 = total
                    for j1 in range(n):
                        if j1 in s:
                            continue
                        for j2 in range(j1+1, n):
                            if j2 in s:
                                continue
                            nt = base + recip[j1] + recip[j2]
                            if nt < 1 and nt > best_nt2:
                                best_nt2 = nt
                                best_pair = (j1, j2)
                    if best_pair:
                        j1, j2 = best_pair
                        s.discard(i1)
                        s.discard(i2)
                        s.add(j1)
                        s.add(j2)
                        total = best_nt2
                        found = True
                        update_best(s, total)
                        break
                if found:
                    break
            if found:
                continue
        
        best_set = list(s)
    
    if best_set is None:
        return [0]
    
    return sorted(best_set)
