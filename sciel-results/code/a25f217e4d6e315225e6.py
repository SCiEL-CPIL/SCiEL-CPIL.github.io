"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0  # safety margin before 90s kill
    
    N = 256
    
    # Precompute conflict_mask: conflict_mask[i] is a bitmask of indices j where i and j conflict
    # (a*b) % (a+b) == 0
    conflict_mask = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    def count_bits(mask):
        return bin(mask).count('1')
    
    # Phase 1: Initialize with all odds
    current_mask = 0
    for i in range(0, N, 2):
        current_mask |= (1 << i)
    
    # Phase 2: Greedy addition of evens, sorted by conflict count with odds
    odds_mask = current_mask
    even_indices = [i for i in range(1, N, 2)]
    
    def conflict_with_odds(idx):
        return count_bits(conflict_mask[idx] & odds_mask)
    
    even_indices.sort(key=conflict_with_odds)
    
    for idx in even_indices:
        if (current_mask >> idx) & 1:
            continue
        if (conflict_mask[idx] & current_mask) == 0:
            current_mask |= (1 << idx)
    
    best_mask = current_mask
    best_size = count_bits(current_mask)
    
    def local_search(start_mask, deadline, max_iters=50000):
        cur_m = start_mask
        cur_sz = count_bits(cur_m)
        best_m = start_mask
        best_sz = cur_sz
        
        # Track temperature for simulated annealing on same-size moves
        temperature = 0.3
        min_temp = 0.05
        cooling = 0.99995
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            move_type = random.random()
            
            if move_type < 0.35:
                # Try to add a random element not in current
                idx = random.randint(0, N - 1)
                if (cur_m >> idx) & 1:
                    continue
                if (conflict_mask[idx] & cur_m) == 0:
                    new_m = cur_m | (1 << idx)
                    new_sz = cur_sz + 1
                    if new_sz > best_sz:
                        best_m = new_m
                        best_sz = new_sz
                    cur_m = new_m
                    cur_sz = new_sz
            elif move_type < 0.75:
                # Try to swap: remove one, add one
                idx_rm = random.randint(0, N - 1)
                if not ((cur_m >> idx_rm) & 1):
                    continue
                
                idx_add = random.randint(0, N - 1)
                if idx_add == idx_rm:
                    continue
                if (cur_m >> idx_add) & 1:
                    continue
                
                trial_base = cur_m & ~(1 << idx_rm)
                if (conflict_mask[idx_add] & trial_base) == 0:
                    new_m = trial_base | (1 << idx_add)
                    # Accept same-size moves with probability based on temperature
                    if random.random() < temperature:
                        cur_m = new_m
                    else:
                        # Still update best if it's a different configuration (might help later)
                        pass
            else:
                # Try 1-remove, 2-add move
                idx_rm = random.randint(0, N - 1)
                if not ((cur_m >> idx_rm) & 1):
                    continue
                
                trial_base = cur_m & ~(1 << idx_rm)
                
                found = 0
                new_m = trial_base
                # Try more attempts to find 2 compatible additions
                for _ in range(15):
                    idx_add = random.randint(0, N - 1)
                    if (new_m >> idx_add) & 1:
                        continue
                    if (conflict_mask[idx_add] & new_m) == 0:
                        new_m |= (1 << idx_add)
                        found += 1
                        if found == 2:
                            break
                
                if found == 2:
                    new_sz = cur_sz + 1
                    if new_sz > best_sz:
                        best_m = new_m
                        best_sz = new_sz
                    cur_m = new_m
                    cur_sz = new_sz
                elif found == 1:
                    # Accept with low probability to explore
                    if random.random() < temperature * 0.5:
                        cur_m = new_m
                        cur_sz = cur_sz  # size unchanged (removed 1, added 1)
            
            # Cool down
            temperature = max(min_temp, temperature * cooling)
        
        return best_m
    
    # Run local search on the greedy result
    best_mask = local_search(best_mask, DEADLINE - 5, max_iters=25000)
    best_size = count_bits(best_mask)
    
    # Multi-start with randomized greedy
    for restart in range(6):
        if time.time() > DEADLINE - 8:
            break
        
        current_mask = 0
        for i in range(0, N, 2):
            current_mask |= (1 << i)
        
        even_indices = [i for i in range(1, N, 2)]
        random.shuffle(even_indices)
        
        for idx in even_indices:
            if (current_mask >> idx) & 1:
                continue
            if (conflict_mask[idx] & current_mask) == 0:
                current_mask |= (1 << idx)
        
        improved = local_search(current_mask, DEADLINE - 5, max_iters=12000)
        sz = count_bits(improved)
        if sz > best_size:
            best_mask = improved
            best_size = sz
    
    # Final polish: systematic swap attempts on the best solution
    if time.time() < DEADLINE - 2:
        cur_m = best_mask
        cur_sz = best_size
        improved = False
        max_polish_iters = 5000
        for _ in range(max_polish_iters):
            if time.time() > DEADLINE - 1:
                break
            
            # Try all possible single adds
            added = False
            for idx in range(N):
                if (cur_m >> idx) & 1:
                    continue
                if (conflict_mask[idx] & cur_m) == 0:
                    cur_m |= (1 << idx)
                    cur_sz += 1
                    added = True
                    break
            if added:
                continue
            
            # Try swaps: for each element in set, try swapping with each element not in set
            swap_found = False
            in_set = [i for i in range(N) if (cur_m >> i) & 1]
            not_in_set = [i for i in range(N) if not ((cur_m >> i) & 1)]
            
            for idx_rm in in_set:
                if swap_found:
                    break
                trial_base = cur_m & ~(1 << idx_rm)
                for idx_add in not_in_set:
                    if (conflict_mask[idx_add] & trial_base) == 0:
                        # Check if this swap enables any new adds
                        new_m = trial_base | (1 << idx_add)
                        # Try to add more after swap
                        extra = 0
                        for idx_extra in not_in_set:
                            if idx_extra == idx_add:
                                continue
                            if (new_m >> idx_extra) & 1:
                                continue
                            if (conflict_mask[idx_extra] & new_m) == 0:
                                new_m |= (1 << idx_extra)
                                extra += 1
                        if extra > 0:
                            cur_m = new_m
                            cur_sz = cur_sz + extra
                            swap_found = True
                            break
            if not swap_found and not added:
                break
        
        if cur_sz > best_size:
            best_mask = cur_m
            best_size = cur_sz
    
    # Convert back to numbers
    result = sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
    
    return result
# EVOLVE-BLOCK-END
