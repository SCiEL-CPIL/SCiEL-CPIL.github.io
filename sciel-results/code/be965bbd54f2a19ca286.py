import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    n = 256
    # 1. Identify forbidden nodes (2*a is a perfect square)
    forbidden = set()
    for a in range(1, n + 1):
        s = 2 * a
        r = int(math.isqrt(s))
        if r * r == s:
            forbidden.add(a)
            
    valid_nodes = [i for i in range(1, n + 1) if i not in forbidden]
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    num_nodes = len(valid_nodes)
    
    # 2. Build conflict masks
    # conflict_masks[i] is a bitmask of indices j such that valid_nodes[i] + valid_nodes[j] is a square
    conflict_masks = [0] * num_nodes
    
    # Precompute squares
    max_sum = 2 * n
    squares = set()
    k = 1
    while k * k <= max_sum:
        squares.add(k * k)
        k += 1
        
    for i in range(num_nodes):
        a = valid_nodes[i]
        mask = 0
        for s in squares:
            b = s - a
            if b in node_to_idx:
                j = node_to_idx[b]
                if j != i:
                    mask |= (1 << j)
        conflict_masks[i] = mask
        
    # 3. Initial Solution: Greedy
    # Try multiple random orders or specific heuristics
    best_set_mask = 0
    best_size = 0
    
    # Heuristic 1: Mod 3 class 1
    # This is known to be 86.
    # Let's try a greedy approach on the graph.
    
    def greedy_select(order):
        mask = 0
        # order is a list of indices
        for idx in order:
            # Check if idx conflicts with current mask
            # (mask & conflict_masks[idx]) == 0
            if (mask & conflict_masks[idx]) == 0:
                mask |= (1 << idx)
        return mask

    # Try a few random permutations
    random.seed(42)
    indices = list(range(num_nodes))
    
    for _ in range(10):
        random.shuffle(indices)
        m = greedy_select(indices)
        size = bin(m).count('1')
        if size > best_size:
            best_size = size
            best_set_mask = m
            
    # Also try the mod 3 strategy explicitly to ensure we have a good baseline
    # Nodes v where v % 3 == 1
    mod3_mask = 0
    for v in valid_nodes:
        if v % 3 == 1:
            idx = node_to_idx[v]
            mod3_mask |= (1 << idx)
    # Verify validity of mod3_mask
    is_valid = True
    temp_mask = mod3_mask
    while temp_mask:
        lsb = temp_mask & -temp_mask
        idx = lsb.bit_length() - 1
        if (mod3_mask & conflict_masks[idx]) != 0:
             is_valid = False
             break
        temp_mask ^= lsb
    if is_valid:
        size = bin(mod3_mask).count('1')
        if size > best_size:
            best_size = size
            best_set_mask = mod3_mask

    # 4. Local Search / Simulated Annealing
    # Objective: Maximize size, subject to 0 conflicts.
    # We will use a penalty-based objective for SA: Score = Size - Penalty * Conflicts
    # But we want to stay in the valid space if possible.
    # Let's try Hill Climbing on valid moves first, then SA.
    
    current_mask = best_set_mask
    current_size = bin(current_mask).count('1')
    
    def count_conflicts(mask):
        count = 0
        temp = mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            # Count bits in (mask & conflict_masks[idx])
            # This counts each conflict twice (once for i, once for j).
            count += bin(mask & conflict_masks[idx]).count('1')
            temp ^= lsb
        return count // 2

    # Initial conflicts should be 0
    # assert count_conflicts(current_mask) == 0
    
    # SA Parameters
    T = 100.0
    T_min = 0.01
    alpha = 0.99
    max_iters = 50000
    
    best_mask_sa = current_mask
    best_score_sa = current_size
    
    for i in range(max_iters):
        if time.time() - start_time > deadline:
            break
            
        # Pick a random node to flip
        idx = random.randint(0, num_nodes - 1)
        bit = 1 << idx
        
        new_mask = current_mask ^ bit
        
        # Calculate delta
        # Delta Size
        if current_mask & bit:
            delta_size = -1
        else:
            delta_size = 1
            
        # Delta Conflicts
        # Old conflicts involving idx: 
        # If idx was in set, conflicts were with (current_mask & conflict_masks[idx]).
        # If idx was not in set, adding it creates conflicts with (current_mask & conflict_masks[idx]).
        
        old_conflicts_involving_idx = bin(current_mask & conflict_masks[idx]).count('1')
        
        # New conflicts:
        # If we add idx (was 0):
        #   New conflicts = old_conflicts_involving_idx (with new neighbors? No, neighbors are fixed)
        #   Wait, if we add idx, the new conflicts are exactly the number of neighbors of idx already in the set.
        #   So delta_conflicts = + old_conflicts_involving_idx
        # If we remove idx (was 1):
        #   We remove conflicts between idx and its neighbors in the set.
        #   So delta_conflicts = - old_conflicts_involving_idx
        
        if current_mask & bit:
            # Removing idx
            delta_conflicts = -old_conflicts_involving_idx
        else:
            # Adding idx
            delta_conflicts = old_conflicts_involving_idx
            
        # We want to maximize Size.
        # If we stay valid (Conflicts == 0), we accept if Size increases.
        # If we become invalid, we accept with probability exp(-DeltaCost/T).
        
        # Let's define Cost = -Size + M * Conflicts
        # We want to minimize Cost.
        # Delta Cost = -Delta_Size + M * Delta_Conflicts
        
        M = 100 # Penalty
        
        delta_cost = -delta_size + M * delta_conflicts
        
        # If new state is valid (Conflicts == 0) and old was valid, and size increases, accept.
        # If new state is invalid, accept with prob exp(-delta_cost/T) if delta_cost > 0?
        # Actually, standard SA:
        # If delta_cost < 0 (improvement), accept.
        # If delta_cost > 0 (worsening), accept with prob exp(-delta_cost/T).
        
        if delta_cost < 0:
            accept = True
        else:
            if T > 0:
                accept = random.random() < math.exp(-delta_cost / T)
            else:
                accept = False
                
        if accept:
            current_mask = new_mask
            current_size = bin(current_mask).count('1')
            
            # Track best valid solution
            # We only update best if the new state is VALID
            # Checking validity every step is expensive if we use count_conflicts.
            # But we can track current_conflicts.
            
            # Let's maintain current_conflicts count.
            # This is tricky with the simple delta calculation above because 
            # "old_conflicts_involving_idx" only counts edges incident to idx.
            # The total number of conflicts in the graph changes by exactly delta_conflicts.
            
            # Let's re-verify the delta logic.
            # Let C(S) be number of conflict edges in S.
            # If we add v: C(S+{v}) = C(S) + deg_S(v).
            # If we remove v: C(S-{v}) = C(S) - deg_S(v).
            # So delta_conflicts is exactly the change in total conflict count.
            
            # We need to track current_conflicts.
            # Let's initialize it.
            # This requires a full count at start.
            
            # I will refactor the loop to track current_conflicts.
            
        T *= alpha
        
    # The above loop structure is a bit messy with state tracking. 
    # Let's write a cleaner SA loop.
    
    # Reset for clean SA
    current_mask = best_set_mask
    current_conflicts = count_conflicts(current_mask) # Should be 0
    current_size = bin(current_mask).count('1')
    
    best_mask_final = current_mask
    best_size_final = current_size
    
    T = 50.0
    T_min = 0.1
    alpha = 0.995
    
    # Estimate max iterations based on time
    # ~100k iterations should be fast enough in Python?
    # Bit operations are fast.
    
    while time.time() - start_time < deadline - 5:
        idx = random.randint(0, num_nodes - 1)
        bit = 1 << idx
        
        # Calculate delta
        if current_mask & bit:
            # Remove
            delta_size = -1
            # Conflicts removed: number of neighbors in current_mask
            delta_conflicts = -bin(current_mask & conflict_masks[idx]).count('1')
        else:
            # Add
            delta_size = 1
            # Conflicts added: number of neighbors in current_mask
            delta_conflicts = bin(current_mask & conflict_masks[idx]).count('1')
            
        # Cost = -Size + M * Conflicts
        M = 100
        delta_cost = -delta_size + M * delta_conflicts
        
        accept = False
        if delta_cost < 0:
            accept = True
        else:
            if T > T_min:
                if random.random() < math.exp(-delta_cost / T):
                    accept = True
            
        if accept:
            current_mask ^= bit
            current_size += delta_size
            current_conflicts += delta_conflicts
            
            # If we have a valid solution (0 conflicts), update best
            if current_conflicts == 0 and current_size > best_size_final:
                best_size_final = current_size
                best_mask_final = current_mask
                
        T *= alpha
        
    # Convert best_mask_final to list
    result = []
    temp = best_mask_final
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        result.append(valid_nodes[idx])
        temp ^= lsb
        
    if not result:
        # Fallback
        result = [1]
        
    return result
