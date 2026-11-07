import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 256
    
    # Precompute prime factors
    def prime_factors(x):
        factors = set()
        d = 2
        while d * d <= x:
            while x % d == 0:
                factors.add(d)
                x //= d
            d += 1
        if x > 1:
            factors.add(x)
        return factors

    factors = [set() for _ in range(n + 1)]
    for i in range(2, n + 1):
        factors[i] = prime_factors(i)
        
    # Identify large primes (> 128)
    large_primes = []
    for p in range(129, n + 1):
        if len(factors[p]) == 1 and list(factors[p])[0] == p:
            large_primes.append(p)
            
    base_solution = [1] + large_primes
    base_sum = sum(base_solution)
    
    candidates = [i for i in range(2, n + 1) if i not in set(base_solution)]
    
    small_primes = sorted(set(p for i in candidates for p in factors[i]))
    prime_to_bit = {p: i for i, p in enumerate(small_primes)}
    
    cand_bitmasks = {}
    for x in candidates:
        mask = 0
        for p in factors[x]:
            mask |= (1 << prime_to_bit[p])
        cand_bitmasks[x] = mask
        
    sorted_candidates = sorted(candidates, reverse=True)
    
    selected = []
    selected_mask = 0
    
    for x in sorted_candidates:
        if (cand_bitmasks[x] & selected_mask) == 0:
            selected.append(x)
            selected_mask |= cand_bitmasks[x]
            
    def local_search(selected, time_limit):
        selected_set = set(selected)
        current_sum = sum(selected)
        selected_mask = 0
        for x in selected:
            selected_mask |= cand_bitmasks[x]
            
        improved = True
        no_improve_count = 0
        max_no_improve = 20
        
        while time.time() < time_limit:
            improved = False
            
            # Try pure addition
            for j in sorted_candidates:
                if j in selected_set:
                    continue
                if (cand_bitmasks[j] & selected_mask) == 0:
                    selected.append(j)
                    selected_set.add(j)
                    selected_mask |= cand_bitmasks[j]
                    current_sum += j
                    improved = True
                    break
            
            if improved:
                no_improve_count = 0
                continue
                
            # Try 1-Swap
            found_swap = False
            sel_list = list(selected)
            random.shuffle(sel_list)
            
            for i in sel_list:
                if time.time() > time_limit - 1.0:
                    break
                others_mask = selected_mask & ~cand_bitmasks[i]
                best_j = -1
                best_gain = 0
                for j in sorted_candidates:
                    if j in selected_set:
                        continue
                    if (cand_bitmasks[j] & others_mask) == 0:
                        gain = j - i
                        if gain > best_gain:
                            best_gain = gain
                            best_j = j
                        # Since sorted desc, first compatible is best?
                        # Yes, j is decreasing. If gain > 0, we want max j.
                        # So break on first compatible if gain > 0?
                        # If gain <= 0, we might find a better one? No, j is decreasing.
                        # So if gain <= 0, all subsequent j will have gain < 0.
                        # So we can break.
                        if gain <= 0:
                            break
                        
                if best_j != -1 and best_gain > 0:
                    selected.remove(i)
                    selected_set.remove(i)
                    selected.append(best_j)
                    selected_set.add(best_j)
                    selected_mask = others_mask | cand_bitmasks[best_j]
                    current_sum += best_gain
                    improved = True
                    found_swap = True
                    break
            
            if improved:
                no_improve_count = 0
                continue
                
            # Try 2-Swap (limited)
            if time.time() < time_limit - 10.0 and len(selected) >= 2:
                found_2swap = False
                sel_list = list(selected)
                random.shuffle(sel_list)
                # Limit pairs to check
                max_pairs = 100
                pairs_checked = 0
                
                for idx_i in range(len(sel_list)):
                    if time.time() > time_limit - 2.0 or pairs_checked > max_pairs:
                        break
                    i = sel_list[idx_i]
                    for idx_k in range(idx_i + 1, len(sel_list)):
                        if time.time() > time_limit - 2.0 or pairs_checked > max_pairs:
                            break
                        pairs_checked += 1
                        k = sel_list[idx_k]
                        others_mask = selected_mask & ~cand_bitmasks[i] & ~cand_bitmasks[k]
                        
                        valid_j = []
                        for j in sorted_candidates:
                            if j in selected_set:
                                continue
                            if (cand_bitmasks[j] & others_mask) == 0:
                                valid_j.append(j)
                                
                        best_pair = None
                        best_gain = 0
                        for j_idx, j in enumerate(valid_j):
                            if j + valid_j[j_idx+1] - i - k <= 0: # Rough bound
                                break
                            for l in valid_j[j_idx+1:]:
                                if (cand_bitmasks[j] & cand_bitmasks[l]) == 0:
                                    gain = j + l - i - k
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_pair = (j, l)
                                    # Can break if gain is decreasing?
                                    # l is decreasing, so gain is decreasing for fixed j.
                                    break
                        
                        if best_pair and best_gain > 0:
                            j, l = best_pair
                            selected.remove(i)
                            selected.remove(k)
                            selected_set.remove(i)
                            selected_set.remove(k)
                            selected.append(j)
                            selected.append(l)
                            selected_set.add(j)
                            selected_set.add(l)
                            selected_mask = 0
                            for x in selected:
                                selected_mask |= cand_bitmasks[x]
                            current_sum += best_gain
                            improved = True
                            found_2swap = True
                            break
                    if found_2swap:
                        break
                        
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Perturb
                    if len(selected) >= 2:
                        to_remove = random.sample(selected, 2)
                        for x in to_remove:
                            selected.remove(x)
                            selected_set.remove(x)
                        selected_mask = 0
                        for x in selected:
                            selected_mask |= cand_bitmasks[x]
                        current_sum = sum(selected)
                        
                        shuffled_cands = list(candidates)
                        random.shuffle(shuffled_cands)
                        added = 0
                        for x in shuffled_cands:
                            if added >= 2:
                                break
                            if x in selected_set:
                                continue
                            if (cand_bitmasks[x] & selected_mask) == 0:
                                selected.append(x)
                                selected_set.add(x)
                                selected_mask |= cand_bitmasks[x]
                                current_sum += x
                                added += 1
                    no_improve_count = 0
        
        return selected

    best = local_search(selected, deadline)
    
    best_set = set(best)
    best_mask = 0
    for x in best:
        best_mask |= cand_bitmasks[x]
        
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for j in sorted_candidates:
            if j in best_set:
                continue
            if (cand_bitmasks[j] & best_mask) == 0:
                best.append(j)
                best_set.add(j)
                best_mask |= cand_bitmasks[j]
                changed = True
                break

    result = base_solution + best
    return [int(x) for x in result]
