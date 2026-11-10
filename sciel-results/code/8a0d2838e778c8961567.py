import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    N = 256
    recip = [0.0] + [1.0 / i for i in range(1, N + 1)]
    
    # Precompute conflicts: for each x, list of (y, z) such that {x,y,z} is a 3-AP
    conflict_pairs = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            c = 2 * b - a
            if c > N or c <= b:
                continue
            conflict_pairs[a].append((b, c))
            conflict_pairs[b].append((a, c))
            conflict_pairs[c].append((a, b))
    
    # Convert to sets for faster lookup
    conflict_sets = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        conflict_sets[x] = set(conflict_pairs[x])
    
    def can_add(x, S_mask):
        """Check if x can be added to set S_mask."""
        for y, z in conflict_pairs[x]:
            if (S_mask >> y) & 1 and (S_mask >> z) & 1:
                return False
        return True

    def get_sum(S_mask):
        """Compute reciprocal sum from bitmask."""
        total = 0.0
        for x in range(1, N + 1):
            if (S_mask >> x) & 1:
                total += recip[x]
        return total

    def greedy_weighted(forbidden_mask, S_mask, order):
        """Fill S by adding elements in given order."""
        S = S_mask
        for x in order:
            if (forbidden_mask >> x) & 1:
                continue
            if (S >> x) & 1:
                continue
            if can_add(x, S):
                S |= (1 << x)
        return S

    def local_search_2swap(S_mask, max_time):
        """Local search with 1-swap and 2-swap moves."""
        S = S_mask
        sum_val = get_sum(S)
        
        while (time.time() - start_time) < max_time:
            improved = False
            
            # Try 1-swap: remove x from S, add y not in S
            best_gain_1 = 0.0
            best_x_1 = -1
            best_y_1 = -1
            
            for y in range(1, N + 1):
                if (S >> y) & 1:
                    continue
                for x in range(N, 0, -1):
                    if not ((S >> x) & 1):
                        continue
                    gain = recip[y] - recip[x]
                    if gain <= 1e-15:
                        continue
                    S_test = S & ~(1 << x)
                    if can_add(y, S_test):
                        if gain > best_gain_1:
                            best_gain_1 = gain
                            best_x_1 = x
                            best_y_1 = y
            
            if best_y_1 != -1:
                S = S & ~(1 << best_x_1)
                S |= (1 << best_y_1)
                sum_val += best_gain_1
                improved = True
                continue
            
            # Try 2-swap: remove x1, x2 from S, add y1, y2 not in S
            best_gain_2 = 0.0
            best_x1_2 = -1
            best_x2_2 = -1
            best_y1_2 = -1
            best_y2_2 = -1
            
            # Only try 2-swap if we have time
            if (time.time() - start_time) < max_time - 2:
                # Get list of elements in S and not in S
                in_S = [x for x in range(1, N + 1) if (S >> x) & 1]
                not_in_S = [x for x in range(1, N + 1) if not ((S >> x) & 1)]
                
                # Try promising pairs: small y's and large x's
                for y1 in not_in_S[:30]:  # Only small y's
                    for y2 in not_in_S[:30]:
                        if y1 >= y2:
                            continue
                        gain_y = recip[y1] + recip[y2]
                        
                        for x1 in in_S[-30:]:  # Only large x's
                            for x2 in in_S[-30:]:
                                if x1 <= x2:
                                    continue
                                gain_x = recip[x1] + recip[x2]
                                gain = gain_y - gain_x
                                if gain <= 1e-15:
                                    continue
                                
                                S_test = S & ~(1 << x1) & ~(1 << x2)
                                if can_add(y1, S_test) and can_add(y2, S_test):
                                    if gain > best_gain_2:
                                        best_gain_2 = gain
                                        best_x1_2 = x1
                                        best_x2_2 = x2
                                        best_y1_2 = y1
                                        best_y2_2 = y2
            
            if best_y1_2 != -1:
                S = S & ~(1 << best_x1_2) & ~(1 << best_x2_2)
                S |= (1 << best_y1_2) | (1 << best_y2_2)
                sum_val += best_gain_2
                improved = True
                continue
            
            # Try remove-and-regreedy
            best_remove_gain = -1.0
            best_remove_S = None
            best_remove_x = -1
            
            in_S_list = [x for x in range(1, N + 1) if (S >> x) & 1]
            # Try removing larger elements first (they have smaller reciprocal)
            for x in sorted(in_S_list, reverse=True)[:20]:
                S_no_x = S & ~(1 << x)
                # Regreedy: add small elements first
                order = sorted([i for i in range(1, N + 1) if not ((S_no_x >> i) & 1)])
                S_new = greedy_weighted(1 << x, S_no_x, order)
                sum_new = get_sum(S_new)
                gain = sum_new - sum_val
                if gain > 1e-15 and gain > best_remove_gain:
                    best_remove_gain = gain
                    best_remove_S = S_new
                    best_remove_x = x
            
            if best_remove_x != -1:
                S = best_remove_S
                sum_val += best_remove_gain
                improved = True
                continue
            
            if not improved:
                break
        
        return S, sum_val

    # Step 1: Multiple initializations
    initializations = []
    
    # Weighted greedy: prioritize small elements (high reciprocal)
    order_asc = list(range(1, N + 1))
    S1 = greedy_weighted(0, 0, order_asc)
    initializations.append(S1)
    
    # Descending greedy
    order_desc = list(range(N, 0, -1))
    S2 = greedy_weighted(0, 0, order_desc)
    initializations.append(S2)
    
    # Randomized greedy with different skip probabilities
    for trial in range(15):
        if (time.time() - start_time) > deadline - 25:
            break
        random.seed(trial * 7919 + 104729)
        S_rand = 0
        for x in range(1, N + 1):
            skip_prob = 0.05 if x <= 20 else (0.1 if x <= 50 else 0.15)
            if random.random() < skip_prob:
                continue
            if can_add(x, S_rand):
                S_rand |= (1 << x)
        initializations.append(S_rand)
    
    # Step 2: Local search from each initialization
    best_S = 0
    best_sum = 0.0
    
    remaining_time = deadline - 10 - (time.time() - start_time)
    if remaining_time < 10:
        # Not enough time, just use best initialization
        for S_init in initializations:
            s = get_sum(S_init)
            if s > best_sum:
                best_sum = s
                best_S = S_init
    else:
        time_per_ls = max(3.0, remaining_time / max(1, len(initializations)))
        
        for i, S_init in enumerate(initializations):
            if (time.time() - start_time) > deadline - 8:
                break
            
            time_left = deadline - 8 - (time.time() - start_time)
            if time_left < 2:
                break
            
            S_ls, sum_ls = local_search_2swap(S_init, min(time_per_ls, time_left))
            
            if sum_ls > best_sum:
                best_sum = sum_ls
                best_S = S_ls
    
    # Step 3: Final polish
    if best_S != 0 and (time.time() - start_time) < deadline - 5:
        remaining = deadline - 5 - (time.time() - start_time)
        if remaining > 5:
            S_final, sum_final = local_search_2swap(best_S, remaining)
            if sum_final > best_sum:
                best_S = S_final
                best_sum = sum_final
    
    # Convert bitmask to list
    result = [x for x in range(1, N + 1) if (best_S >> x) & 1]
    
    # Validate
    S_set = set(result)
    valid = True
    for a in S_set:
        for b in S_set:
            if a >= b:
                continue
            s = a + b
            if s % 2 == 0:
                mid = s // 2
                if mid in S_set and mid != a and mid != b:
                    valid = False
                    break
        if not valid:
            break
    
    if not valid:
        result = [x for x in range(1, N + 1) if (S1 >> x) & 1]
    
    return result
