import time
import math
import random

def construct():
    start_time = time.time()
    timeout = 58.0
    
    n = 256
    
    # Build conflict masks
    conflict = [0] * n
    
    for i in range(n):
        a = i + 1
        mask_i = 0
        for j in range(n):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                mask_i |= (1 << j)
        conflict[i] = mask_i
    
    # Precompute degrees
    degrees = [conflict[i].bit_count() for i in range(n)]
    
    best_size = 0
    best_mask = 0
    
    def popcount(x):
        return x.bit_count()
    
    # Greedy coloring upper bound on the induced subgraph of candidates
    def color_bound(candidates):
        """Return upper bound on MIS size in the induced subgraph of candidates."""
        if candidates == 0:
            return 0
        
        uncolored = candidates
        colors = {}
        color_count = 0
        
        while uncolored:
            # Pick vertex with max degree in uncolored subgraph
            v_idx = -1
            max_deg = -1
            c = uncolored
            while c:
                lsb = c & -c
                i = lsb.bit_length() - 1
                deg = (conflict[i] & uncolored).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v_idx = i
                c ^= lsb
            
            if v_idx == -1:
                break
            
            # Find smallest color not used by colored neighbors
            used_colors = set()
            neighbors_colored = conflict[v_idx] & ~uncolored
            nc = neighbors_colored
            while nc:
                lsb = nc & -nc
                j = lsb.bit_length() - 1
                if j in colors:
                    used_colors.add(colors[j])
                nc ^= lsb
            
            for col in range(color_count + 1):
                if col not in used_colors:
                    colors[v_idx] = col
                    if col == color_count:
                        color_count += 1
                    break
            
            uncolored &= ~(1 << v_idx)
        
        return color_count
    
    def solve(candidates, current_size, current_mask, depth):
        nonlocal best_size, best_mask
        
        if time.time() - start_time > timeout:
            return
            
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
            
        # Pruning with greedy coloring bound
        if current_size + popcount(candidates) <= best_size:
            return
        
        # Use coloring bound for tighter pruning
        if popcount(candidates) <= 80:
            ub = color_bound(candidates)
            if current_size + ub <= best_size:
                return
        
        # Select vertex v from candidates using max-degree heuristic
        v_idx = -1
        max_deg = -1
        c = candidates
        while c:
            lsb = c & -c
            i = lsb.bit_length() - 1
            deg = (conflict[i] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                v_idx = i
            c ^= lsb
        
        if v_idx == -1:
            return
            
        v_bit = 1 << v_idx
        
        # Branch 1: Include v_idx
        new_cand_inc = (candidates & ~v_bit) & ~conflict[v_idx]
        new_mask_inc = current_mask | v_bit
        solve(new_cand_inc, current_size + 1, new_mask_inc, depth + 1)
        
        if time.time() - start_time > timeout:
            return
        
        # Branch 2: Exclude v_idx
        new_cand_exc = candidates & ~v_bit
        solve(new_cand_exc, current_size, current_mask, depth + 1)
    
    # Phase 1: Branch and bound to find a good initial solution
    initial_cand = (1 << n) - 1
    solve(initial_cand, 0, 0, 0)
    
    if best_size == 0:
        return [1]
    
    # Convert best_mask to a set for fast lookup
    best_set = set()
    for i in range(n):
        if (best_mask >> i) & 1:
            best_set.add(i)
    
    # Phase 2: Simulated annealing to try to improve
    if time.time() - start_time < timeout - 10:
        current_mask = best_mask
        current_size = best_size
        current_set = set(best_set)
        
        # Temperature parameters
        T = 1.0
        cooling_rate = 0.9995
        min_T = 0.001
        
        # Precompute list of indices
        indices = list(range(n))
        
        while T > min_T and time.time() - start_time < timeout - 2:
            # Pick a random vertex to potentially add or remove
            idx = random.choice(indices)
            bit = 1 << idx
            
            if current_mask & bit:
                # Try removing idx
                new_mask = current_mask & ~bit
                new_size = current_size - 1
                
                # Accept if it doesn't violate constraints (removal always valid)
                # But we want to keep size high, so only accept with probability
                # based on size decrease
                delta_size = -1
                prob = math.exp(delta_size / T) if T > 0 else 0
                if random.random() < prob:
                    current_mask = new_mask
                    current_size = new_size
                    current_set.discard(idx)
            else:
                # Try adding idx
                # Check if idx conflicts with any current member
                can_add = True
                conflicts_with_current = conflict[idx] & current_mask
                if conflicts_with_current:
                    can_add = False
                
                if can_add:
                    new_mask = current_mask | bit
                    new_size = current_size + 1
                    current_mask = new_mask
                    current_size = new_size
                    current_set.add(idx)
                    
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask
                        best_set = set(current_set)
                else:
                    # Try a swap: remove one conflicting element and add idx
                    # Pick a random conflicting element to remove
                    conflicting_bits = conflict[idx] & current_mask
                    if conflicting_bits:
                        # Get a random conflicting index
                        c = conflicting_bits
                        lsb = c & -c
                        # Randomly select among conflicting bits
                        num_conflicts = c.bit_count()
                        target = random.randrange(num_conflicts)
                        temp_c = c
                        for _ in range(target):
                            lsb = temp_c & -temp_c
                            temp_c ^= lsb
                        remove_idx = lsb.bit_length() - 1
                        remove_bit = 1 << remove_idx
                        
                        # Check if after removal, we can still add idx
                        # (idx doesn't conflict with remove_idx's neighbors that remain)
                        new_mask = (current_mask & ~remove_bit) | bit
                        # Verify no conflicts in new_mask
                        valid = True
                        # Quick check: idx should not conflict with any remaining
                        if conflict[idx] & (current_mask & ~remove_bit):
                            valid = False
                        
                        if valid:
                            current_mask = new_mask
                            current_size = current_size  # Size stays same
                            current_set.discard(remove_idx)
                            current_set.add(idx)
            
            T *= cooling_rate
        
        # Update best if improved
        if current_size > best_size:
            best_size = current_size
            best_mask = current_mask
    
    # Convert best_mask back to list of integers
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
