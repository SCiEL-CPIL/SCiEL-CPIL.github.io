import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 256
    
    # Precompute prime factors for each number 1..256
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
    
    # factors[i] = set of prime factors of i (for i in 1..256)
    # factors[1] = empty set
    factors = [set() for _ in range(n + 1)]
    for i in range(2, n + 1):
        factors[i] = prime_factors(i)
    
    # Build bitmask representation for fast compatibility checks
    # Assign each prime a bit position
    all_primes = sorted(set(p for i in range(2, n + 1) for p in factors[i]))
    prime_to_bit = {p: i for i, p in enumerate(all_primes)}
    
    # bitmask[i] = bitmask of prime factors of i
    bitmask = [0] * (n + 1)
    for i in range(2, n + 1):
        mask = 0
        for p in factors[i]:
            mask |= (1 << prime_to_bit[p])
        bitmask[i] = mask
    
    # Fast compatibility check using bitmasks
    def compatible_fast(a, b):
        return (bitmask[a] & bitmask[b]) == 0
    
    # Greedy initial solution: sort by value descending, add if compatible with all current
    def greedy_solution():
        selected = [1]  # 1 is always compatible
        selected_mask = 0
        for candidate in range(n, 1, -1):
            if (bitmask[candidate] & selected_mask) == 0:
                selected.append(candidate)
                selected_mask |= bitmask[candidate]
        return selected
    
    # Local search with bitmask optimization
    def local_search(selected, time_limit):
        selected_set = set(selected)
        current_sum = sum(selected)
        
        # Precompute combined mask for selected elements
        def compute_mask(lst):
            m = 0
            for x in lst:
                m |= bitmask[x]
            return m
        
        selected_mask = compute_mask(selected)
        improved = True
        
        while improved and time.time() < time_limit:
            improved = False
            
            # Try single swaps: remove i, add j if compatible with all others and improves sum
            for i in list(selected):
                others = [s for s in selected if s != i]
                others_mask = selected_mask & ~bitmask[i]
                for j in range(n, 1, -1):
                    if j in selected_set:
                        continue
                    if (bitmask[j] & others_mask) == 0:
                        new_sum = current_sum - i + j
                        if new_sum > current_sum:
                            selected = others + [j]
                            selected_set = set(selected)
                            selected_mask = others_mask | bitmask[j]
                            current_sum = new_sum
                            improved = True
                            break
                if improved:
                    break
            
            # Try adding a single element that's compatible with all current
            if not improved and time.time() < time_limit:
                for j in range(n, 1, -1):
                    if j in selected_set:
                        continue
                    if (bitmask[j] & selected_mask) == 0:
                        selected.append(j)
                        selected_set.add(j)
                        selected_mask |= bitmask[j]
                        current_sum += j
                        improved = True
                        break
            
            # Try double swaps: remove i, j, add k, l if compatible
            if not improved and time.time() < time_limit - 2.0:
                for i_idx in range(len(selected)):
                    if time.time() > time_limit - 1.0:
                        break
                    for j_idx in range(i_idx + 1, len(selected)):
                        if time.time() > time_limit - 1.0:
                            break
                        i, j = selected[i_idx], selected[j_idx]
                        others = [s for k_idx, s in enumerate(selected) if k_idx != i_idx and k_idx != j_idx]
                        others_mask = selected_mask & ~bitmask[i] & ~bitmask[j]
                        
                        for k in range(n, 1, -1):
                            if time.time() > time_limit - 1.0:
                                break
                            if k in selected_set:
                                continue
                            if (bitmask[k] & others_mask) != 0:
                                continue
                            for l in range(n, 1, -1):
                                if time.time() > time_limit - 1.0:
                                    break
                                if l in selected_set or l == k:
                                    continue
                                if (bitmask[l] & others_mask) != 0 or (bitmask[k] & bitmask[l]) != 0:
                                    continue
                                new_sum = current_sum - i - j + k + l
                                if new_sum > current_sum:
                                    selected = others + [k, l]
                                    selected_set = set(selected)
                                    selected_mask = compute_mask(selected)
                                    current_sum = new_sum
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return selected
    
    # Start with greedy
    best = greedy_solution()
    
    # Local search
    best = local_search(best, deadline)
    
    # Final polish: try to add any remaining compatible numbers
    best_set = set(best)
    best_mask = 0
    for x in best:
        best_mask |= bitmask[x]
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for j in range(n, 1, -1):
            if j in best_set:
                continue
            if (bitmask[j] & best_mask) == 0:
                best.append(j)
                best_set.add(j)
                best_mask |= bitmask[j]
                changed = True
                break
    
    # Convert to list of Python ints
    result = [int(x) for x in best]
    return result
