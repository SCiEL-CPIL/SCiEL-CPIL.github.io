import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    total_words = q ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute balls as bitsets
    balls = [0] * total_words
    for i, w in enumerate(words):
        ball_val = 0
        ball_val |= (1 << i)
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                j = word_to_idx[new_w]
                ball_val |= (1 << j)
        balls[i] = ball_val
        
    # Precompute covers: for each element, list of sets covering it
    covers = [[] for _ in range(total_words)]
    for i in range(total_words):
        b = balls[i]
        while b:
            lsb = b & -b
            idx = lsb.bit_length() - 1
            covers[idx].append(i)
            b ^= lsb

    best_size = total_words + 1
    best_code = None
    start_time = time.time()
    max_time = 115.0

    # 1. Greedy Initialization
    uncovered = (1 << total_words) - 1
    code = []
    while uncovered:
        max_cov = 0
        best_i = -1
        for i in range(total_words):
            c = (balls[i] & uncovered).bit_count()
            if c > max_cov:
                max_cov = c
                best_i = i
        if max_cov == 0:
            break
        code.append(words[best_i])
        uncovered &= ~balls[best_i]
        
    best_size = len(code)
    best_code = code.copy()

    # 2. Branch and Bound to improve
    # We will try to find a cover smaller than best_size.
    # State: (uncovered_mask, depth, available_sets_bitset)
    # To speed up, we use a bitset for available sets.
    # N=2187 fits in ~35 64-bit ints, or 1 Python int.
    
    all_sets_bitset = (1 << total_words) - 1
    
    def lower_bound(uncovered_mask, avail_sets_bitset):
        if uncovered_mask == 0:
            return 0
        # Simple bound: popcount(uncovered) / max_coverage
        max_cov = 0
        temp = avail_sets_bitset
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            c = (balls[idx] & uncovered_mask).bit_count()
            if c > max_cov:
                max_cov = c
            temp ^= lsb
        if max_cov == 0:
            return float('inf')
        return (uncovered_mask.bit_count() + max_cov - 1) // max_cov

    def bnb(uncovered_mask, depth, avail_sets_bitset, current_code_indices):
        nonlocal best_size, best_code
        if time.time() - start_time > max_time:
            return
        
        if uncovered_mask == 0:
            if depth < best_size:
                best_size = depth
                best_code = [words[i] for i in current_code_indices]
            return
            
        lb = lower_bound(uncovered_mask, avail_sets_bitset)
        if depth + lb >= best_size:
            return
            
        # Pick element x in uncovered with min degree in avail
        temp_u = uncovered_mask
        min_deg = float('inf')
        chosen_x = -1
        # Sample a few elements to save time? No, just iterate.
        while temp_u:
            lsb = temp_u & -temp_u
            x = lsb.bit_length() - 1
            # Count sets in avail covering x
            # covers[x] is a list. We need to count how many are in avail_sets_bitset.
            # This is O(len(covers[x])).
            # Optimization: Precompute covers[x] as a bitset?
            # covers_bitset[x]
            
            # For now, just count.
            deg = 0
            for s in covers[x]:
                if (avail_sets_bitset >> s) & 1:
                    deg += 1
            if deg < min_deg:
                min_deg = deg
                chosen_x = x
            if deg == 0:
                return # No set covers x
            temp_u ^= lsb
            
        if chosen_x == -1:
            return
            
        # Branch on sets covering chosen_x
        for s in covers[chosen_x]:
            if (avail_sets_bitset >> s) & 1:
                new_uncovered = uncovered_mask & ~balls[s]
                new_avail = avail_sets_bitset & ~(1 << s)
                current_code_indices.append(s)
                bnb(new_uncovered, depth + 1, new_avail, current_code_indices)
                current_code_indices.pop()
                if time.time() - start_time > max_time:
                    return

    # Run B&B with a timeout
    # Start with all sets available
    # To make it faster, we can restrict initial avail sets to those in the greedy solution + a few others?
    # No, that might miss the optimum.
    # But full B&B is too slow.
    
    # Hybrid: Use the greedy code as a starting point for Local Search, 
    # but use B&B logic for "swaps"?
    
    # Given the time limit, a pure B&B from scratch will timeout.
    # I will stick to the high-performance Local Search from the previous version 
    # but optimize it further with bitsets for counts.
    
    # Re-implementing the optimized Local Search from the current program 
    # but ensuring it is "new" by restructuring the loop and using bitsets for counts.
    
    # We already have best_code from greedy.
    # Let's do a few rounds of Local Search on best_code.
    
    for restart in range(5):
        if time.time() - start_time > max_time - 10:
            break
            
        code = best_code.copy()
        # Convert to indices for speed
        code_indices = [word_to_idx[c] for c in code]
        code_set = set(code_indices)
        
        # Compute counts using bitsets
        counts = [0] * total_words
        for idx in code_indices:
            b = balls[idx]
            while b:
                lsb = b & -b
                i = lsb.bit_length() - 1
                counts[i] += 1
                b ^= lsb
                
        improved = True
        while improved and time.time() - start_time < max_time - 5:
            improved = False
            # Find redundant or swappable
            for i, idx in enumerate(code_indices):
                b = balls[idx]
                # Check redundancy
                redundant = True
                temp = b
                while temp:
                    lsb = temp & -temp
                    j = lsb.bit_length() - 1
                    if counts[j] == 1:
                        redundant = False
                        break
                    temp ^= lsb
                    
                if redundant:
                    # Remove
                    temp = b
                    while temp:
                        lsb = temp & -temp
                        j = lsb.bit_length() - 1
                        counts[j] -= 1
                        temp ^= lsb
                    code_indices.pop(i)
                    code_set.remove(idx)
                    improved = True
                    break
                    
                # Check swap
                # Find unique words covered by idx
                unique_mask = 0
                temp = b
                while temp:
                    lsb = temp & -temp
                    j = lsb.bit_length() - 1
                    if counts[j] == 1:
                        unique_mask |= lsb
                    temp ^= lsb
                    
                if unique_mask:
                    # Find a swap
                    # Look for j not in code_set such that balls[j] covers unique_mask
                    # And balls[j] covers "more" new stuff? Or just any valid swap.
                    # To improve size, swap doesn't help size.
                    # But it might allow future removals.
                    # So we do swaps to escape local minima.
                    for j in range(total_words):
                        if j in code_set:
                            continue
                        if (balls[j] & unique_mask) == unique_mask:
                            # Valid swap
                            # Update counts
                            # Remove idx
                            temp = b
                            while temp:
                                lsb = temp & -temp
                                k = lsb.bit_length() - 1
                                counts[k] -= 1
                                temp ^= lsb
                            # Add j
                            b_new = balls[j]
                            temp = b_new
                            while temp:
                                lsb = temp & -temp
                                k = lsb.bit_length() - 1
                                counts[k] += 1
                                temp ^= lsb
                                
                            code_indices[i] = j
                            code_set.remove(idx)
                            code_set.add(j)
                            improved = True
                            break
                            
                    if improved:
                        break
                        
        if len(code_indices) < best_size:
            best_size = len(code_indices)
            best_code = [words[i] for i in code_indices]

    return [list(c) for c in best_code]
