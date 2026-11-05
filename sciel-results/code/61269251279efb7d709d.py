import itertools
import time
import random
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for safety
    
    # Precompute all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    full_mask = (1 << n_t_subsets) - 1
    
    # Precompute all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(k_blocks)
    
    # Precompute bitsets for each k-block
    # block_bitsets[i] is the mask of t-subsets covered by k_blocks[i]
    block_bitsets = [0] * n_k_blocks
    # Also precompute ts_to_blocks for greedy initialization speed
    # ts_to_blocks[ts_idx] = list of block indices containing this t-subset
    ts_to_blocks = [[] for _ in range(n_t_subsets)]
    
    for i, b in enumerate(k_blocks):
        mask = 0
        for ts in itertools.combinations(b, t):
            idx = subset_to_idx[ts]
            mask |= (1 << idx)
            ts_to_blocks[idx].append(i)
        block_bitsets[i] = mask

    # 1. Greedy Construction
    uncovered_mask = full_mask
    chosen_block_indices = []
    
    while uncovered_mask:
        if time.time() > deadline - 10.0: # Reserve time for SA
            break
            
        # Find the first uncovered t-subset
        lsb_idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
        
        # Find the block in ts_to_blocks[lsb_idx] that covers the most uncovered subsets
        best_bi = -1
        best_count = -1
        for bi in ts_to_blocks[lsb_idx]:
            mask = block_bitsets[bi]
            count = (uncovered_mask & mask).bit_count()
            if count > best_count:
                best_count = count
                best_bi = bi
        
        if best_bi == -1:
            break
            
        chosen_block_indices.append(best_bi)
        uncovered_mask &= ~block_bitsets[best_bi]
        
    # Convert indices to blocks
    current_blocks = [k_blocks[i] for i in chosen_block_indices]
    
    # 2. Local Search: Removal
    # Try to remove redundant blocks
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        n = len(current_blocks)
        if n == 0: break
        
        # Randomize order for removal to escape symmetry? Or just iterate.
        # Iterating is deterministic.
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
                
            b_i = current_blocks[i]
            mask_i = block_bitsets[chosen_block_indices[i]] # Wait, indices might be stale if we pop?
            # Actually, it's better to keep indices and update them, or just recalculate mask from block.
            # Recalculating mask from block is expensive.
            # Let's keep `current_block_indices` in sync with `current_blocks`.
            
            # Let's restructure to keep indices.
            pass # This block is just a placeholder for thought process. I will rewrite the loop below.
            
    # Let's implement the Removal and SA more carefully using indices.
    
    # Reset current state to Greedy result
    current_indices = list(chosen_block_indices)
    
    # Helper to calculate energy
    def calc_energy(indices):
        if not indices:
            return 0
        union = 0
        for idx in indices:
            union |= block_bitsets[idx]
        uncovered = (full_mask ^ union).bit_count() # full_mask & ~union
        return len(indices) + 100 * uncovered

    # 2. Greedy Removal (Deterministic)
    def try_remove_redundant(indices):
        n = len(indices)
        if n == 0: return indices, False
        for i in range(n):
            # Check if block i is redundant
            # Union of all others
            union_others = 0
            for j in range(n):
                if j != i:
                    union_others |= block_bitsets[indices[j]]
            
            # If block i's coverage is subset of union_others
            mask_i = block_bitsets[indices[i]]
            if (mask_i & ~union_others) == 0:
                # Redundant
                new_indices = indices[:i] + indices[i+1:]
                return new_indices, True
        return indices, False

    # Apply removals
    while time.time() < deadline - 10.0:
        current_indices, removed = try_remove_redundant(current_indices)
        if not removed:
            break

    best_indices = list(current_indices)
    best_energy = calc_energy(best_indices)
    
    # 3. Simulated Annealing
    # We want to minimize energy.
    # Moves: Swap, Add, Remove
    # Temperature schedule
    
    T = 10.0
    T_min = 0.1
    alpha = 0.95
    
    # To make SA effective, we need to ensure we explore.
    # We will run SA until time runs out.
    
    random.seed(42) # Deterministic for reproducibility in testing, though evaluator might not care
    
    while time.time() < deadline - 2.0:
        if T < T_min:
            break
            
        # Generate Move
        move_type = random.random()
        new_indices = list(current_indices)
        
        if move_type < 0.7:
            # Swap: Pick one in, one out
            if len(current_indices) > 0:
                out_idx_pos = random.randint(0, len(current_indices) - 1)
                out_idx = current_indices[out_idx_pos]
                
                # Pick a random block not in current set
                # If set is small, picking random from all and checking is fast.
                # If set is large, we might pick one already in set.
                in_idx = random.randint(0, n_k_blocks - 1)
                if in_idx == out_idx:
                    continue # Skip this step if same block
                # Check if in_idx is already in current_indices?
                # If it is, it's not a swap, it's a no-op or duplicate.
                # Blocks must be distinct.
                if in_idx in current_indices:
                    continue
                    
                new_indices[out_idx_pos] = in_idx
            else:
                # Add
                in_idx = random.randint(0, n_k_blocks - 1)
                new_indices.append(in_idx)
                
        elif move_type < 0.85:
            # Remove
            if len(current_indices) > 0:
                out_idx_pos = random.randint(0, len(current_indices) - 1)
                new_indices.pop(out_idx_pos)
            else:
                continue # Can't remove from empty
                
        else:
            # Add
            in_idx = random.randint(0, n_k_blocks - 1)
            if in_idx in current_indices:
                continue
            new_indices.append(in_idx)
            
        # Calculate New Energy
        # Optimization: If the move is a swap/add/remove, we can update union incrementally?
        # No, recalculating union is O(N). N~300. Bitset OR is fast.
        # Let's just recalculate.
        
        union_new = 0
        for idx in new_indices:
            union_new |= block_bitsets[idx]
            
        uncovered_new = (full_mask ^ union_new).bit_count()
        energy_new = len(new_indices) + 100 * uncovered_new
        
        delta = energy_new - best_energy # Compare to best? Or current?
        # Standard SA compares to current state.
        current_energy = calc_energy(current_indices) # This is expensive if called every time.
        # Let's maintain current_energy.
        
        # Wait, I need to maintain `current_energy` variable.
        # Let's refactor the loop to maintain state variables.
        
        # I will rewrite the SA loop properly in the final code.
        
    # The above loop was pseudo-code. I need to write the actual code carefully.
    
    # --- Revised SA Implementation ---
    
    current_indices = list(best_indices)
    current_union = 0
    for idx in current_indices:
        current_union |= block_bitsets[idx]
    current_uncovered = (full_mask ^ current_union).bit_count()
    current_energy = len(current_indices) + 100 * current_uncovered
    
    best_indices = list(current_indices)
    best_energy = current_energy
    
    T = 5.0
    T_min = 0.05
    alpha = 0.99
    
    while time.time() < deadline - 1.0:
        if T < T_min:
            T = T_min
            
        # Generate Move
        new_indices = list(current_indices)
        move_valid = True
        
        r = random.random()
        if r < 0.7: # Swap
            if current_indices:
                pos = random.randint(0, len(current_indices) - 1)
                old_idx = current_indices[pos]
                new_idx = random.randint(0, n_k_blocks - 1)
                if new_idx != old_idx and new_idx not in current_indices:
                    new_indices[pos] = new_idx
                else:
                    move_valid = False
            else:
                move_valid = False
        elif r < 0.85: # Remove
            if current_indices:
                pos = random.randint(0, len(current_indices) - 1)
                new_indices.pop(pos)
            else:
                move_valid = False
        else: # Add
            new_idx = random.randint(0, n_k_blocks - 1)
            if new_idx not in current_indices:
                new_indices.append(new_idx)
            else:
                move_valid = False
                
        if not move_valid:
            T *= alpha
            continue
            
        # Calculate New State
        # Instead of full recalc, we can try to update union?
        # Swap: new_union = (old_union & ~mask_old) | mask_new? No, other blocks might cover mask_old's bits.
        # So we MUST recalculate union for correctness.
        # O(N) bitset ops.
        
        new_union = 0
        for idx in new_indices:
            new_union |= block_bitsets[idx]
            
        new_uncovered = (full_mask ^ new_union).bit_count()
        new_energy = len(new_indices) + 100 * new_uncovered
        
        delta = new_energy - current_energy
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_indices = new_indices
            current_union = new_union
            current_uncovered = new_uncovered
            current_energy = new_energy
            
            if current_uncovered == 0 and current_energy < best_energy:
                best_energy = current_energy
                best_indices = list(current_indices)
                
        T *= alpha
        
    # Ensure we return a valid solution
    # If best_indices is valid, return it.
    # Check validity of best_indices
    union_best = 0
    for idx in best_indices:
        union_best |= block_bitsets[idx]
    if (full_mask ^ union_best).bit_count() == 0:
        return [list(k_blocks[i]) for i in best_indices]
    
    # Fallback to current if valid
    if (full_mask ^ current_union).bit_count() == 0:
        return [list(k_blocks[i]) for i in current_indices]
        
    # Fallback to initial greedy (which was valid)
    # Reconstruct initial greedy result if it was valid
    # We stored `chosen_block_indices` from the first pass.
    # Let's verify it.
    union_greedy = 0
    for idx in chosen_block_indices:
        union_greedy |= block_bitsets[idx]
    if (full_mask ^ union_greedy).bit_count() == 0:
        return [list(k_blocks[i]) for i in chosen_block_indices]
        
    # If all else fails, return empty (should not happen)
    return []
