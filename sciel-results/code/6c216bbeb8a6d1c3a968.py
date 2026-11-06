import time
import random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0  # Safety margin before 90s kill
    
    N = 256
    all_nums = list(range(1, N + 1))
    
    # Precompute conflict_mask: conflict_mask[i] is a bitmask of indices j where i and j conflict
    conflict_mask = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflict_mask[i] |= (1 << j)
    
    # All odds are indices 0,2,4,...,254 (numbers 1,3,5,...,255)
    odds_indices = [i for i in range(N) if (i + 1) % 2 == 1]
    evens_indices = [i for i in range(N) if (i + 1) % 2 == 0]
    
    def count_bits(x):
        return bin(x).count('1')
    
    def get_conflict_count(idx, current_mask):
        """Count conflicts between idx and elements in current_mask using bit operations."""
        cm = conflict_mask[idx] & current_mask
        return count_bits(cm)
    
    # Phase 1: Start with all odds, greedily add compatible evens
    current_mask = 0
    for i in odds_indices:
        current_mask |= (1 << i)
    
    best_mask = current_mask
    best_size = count_bits(current_mask)
    
    # Greedy add evens sorted by conflict count (fewest conflicts first)
    evens_sorted = sorted(evens_indices, key=lambda i: count_bits(conflict_mask[i] & current_mask))
    
    for e in evens_sorted:
        if (current_mask >> e) & 1:
            continue
        if (conflict_mask[e] & current_mask) == 0:
            current_mask |= (1 << e)
            sz = count_bits(current_mask)
            if sz > best_size:
                best_mask = current_mask
                best_size = sz
    
    # Phase 2: Local search with bitmask operations
    rng = random.Random(42)
    
    # Precompute for speed
    all_indices = list(range(N))
    
    max_iters = 500000
    for it in range(max_iters):
        if time.time() > DEADLINE - 10:
            break
        
        r = rng.random()
        
        if r < 0.25:
            # Try to add a random even
            e = rng.choice(evens_indices)
            if (current_mask >> e) & 1:
                continue
            if (conflict_mask[e] & current_mask) == 0:
                current_mask |= (1 << e)
                sz = count_bits(current_mask)
                if sz > best_size:
                    best_mask = current_mask
                    best_size = sz
        
        elif r < 0.5:
            # Try 1-swap: Remove one element, add one even
            to_remove = rng.choice(all_indices)
            if not ((current_mask >> to_remove) & 1):
                continue
            trial_mask = current_mask & ~(1 << to_remove)
            
            e = rng.choice(evens_indices)
            if (trial_mask >> e) & 1:
                continue
            if (conflict_mask[e] & trial_mask) == 0:
                new_mask = trial_mask | (1 << e)
                # Accept size-neutral moves with low probability for exploration
                if rng.random() < 0.05:
                    current_mask = new_mask
        
        elif r < 0.75:
            # Try 1-Remove, 2-Add (size +1)
            to_remove = rng.choice(all_indices)
            if not ((current_mask >> to_remove) & 1):
                continue
            trial_mask = current_mask & ~(1 << to_remove)
            
            e1 = rng.choice(evens_indices)
            e2 = rng.choice(evens_indices)
            if e1 == e2:
                continue
            if (trial_mask >> e1) & 1 or (trial_mask >> e2) & 1:
                continue
            
            if (conflict_mask[e1] & trial_mask) == 0 and (conflict_mask[e2] & (trial_mask | (1 << e1))) == 0:
                new_mask = trial_mask | (1 << e1) | (1 << e2)
                sz = count_bits(new_mask)
                if sz > count_bits(current_mask):
                    current_mask = new_mask
                    if sz > best_size:
                        best_mask = current_mask
                        best_size = sz
                elif rng.random() < 0.02:
                    current_mask = new_mask
        else:
            # Try 2-Remove, 3-Add (size +1)
            to_remove1 = rng.choice(all_indices)
            to_remove2 = rng.choice(all_indices)
            if to_remove1 == to_remove2:
                continue
            if not ((current_mask >> to_remove1) & 1) or not ((current_mask >> to_remove2) & 1):
                continue
            
            trial_mask = current_mask & ~(1 << to_remove1) & ~(1 << to_remove2)
            
            e1 = rng.choice(evens_indices)
            e2 = rng.choice(evens_indices)
            e3 = rng.choice(evens_indices)
            if e1 == e2 or e1 == e3 or e2 == e3:
                continue
            if (trial_mask >> e1) & 1 or (trial_mask >> e2) & 1 or (trial_mask >> e3) & 1:
                continue
            
            mask_check = trial_mask
            valid = True
            for e in [e1, e2, e3]:
                if (conflict_mask[e] & mask_check) != 0:
                    valid = False
                    break
                mask_check |= (1 << e)
            
            if valid:
                new_mask = trial_mask | (1 << e1) | (1 << e2) | (1 << e3)
                sz = count_bits(new_mask)
                if sz > count_bits(current_mask):
                    current_mask = new_mask
                    if sz > best_size:
                        best_mask = current_mask
                        best_size = sz
    
    # Phase 3: Tomita Branch and Bound for final polish
    # Focus on improving beyond best_size
    # Use the current best as starting point and try to find larger sets
    
    def greedy_bound(U):
        if U == 0:
            return 0
        color = 0
        u = U
        while u:
            color += 1
            v_idx = (u & -u).bit_length() - 1
            v = v_idx + 1
            u &= ~(conflict_mask[v_idx] | (1 << v_idx))
        return color
    
    best_size_bb = best_size
    best_mask_bb = best_mask
    
    def search(U, current_size, current_mask):
        nonlocal best_size_bb, best_mask_bb
        if time.time() > DEADLINE - 2:
            raise TimeoutError()
            
        if U == 0:
            if current_size > best_size_bb:
                best_size_bb = current_size
                best_mask_bb = current_mask
            return
            
        bound = greedy_bound(U)
        if current_size + bound <= best_size_bb:
            return
            
        # Pick vertex with highest degree in U
        u_temp = U
        best_v_idx = -1
        max_deg = -1
        while u_temp:
            v_idx = (u_temp & -u_temp).bit_length() - 1
            u_temp &= u_temp - 1
            deg = count_bits(U & conflict_mask[v_idx])
            if deg > max_deg:
                max_deg = deg
                best_v_idx = v_idx
                
        if best_v_idx == -1:
            return
            
        v_idx = best_v_idx
        bit = 1 << v_idx
        
        # Branch 1: Include v
        new_U_include = U & ~(conflict_mask[v_idx] | bit)
        search(new_U_include, current_size + 1, current_mask | bit)
        
        # Branch 2: Exclude v
        new_U_exclude = U & ~bit
        search(new_U_exclude, current_size, current_mask)
    
    # Start B&B from the complement of best_mask (vertices not in best set)
    # But we need to be careful: we can only add vertices that don't conflict with best_mask
    # Actually, let's run B&B on a restricted graph: vertices that could potentially be added
    
    # Better approach: Run B&B on the full graph but start with a good lower bound
    # and use the time limit to explore promising branches
    
    # Initialize with best found so far
    # Try B&B starting from scratch but with strong pruning
    try:
        search((1 << N) - 1, 0, 0)
    except TimeoutError:
        pass
    except RecursionError:
        pass
    
    # Use the best from either local search or B&B
    if best_size_bb > best_size:
        best_mask = best_mask_bb
        best_size = best_size_bb
    
    # Convert best_mask to list of integers
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    # Verify
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                # Invalid, fallback to greedy result
                return sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
                
    return result
