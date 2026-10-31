import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer for final checks and return
    
    # 1. Precomputation
    # Generate all t-subsets as bitmasks
    tsub_masks = []
    for sub in itertools.combinations(range(v), t):
        m = 0
        for p in sub:
            m |= (1 << p)
        tsub_masks.append(m)
    n_tsub = len(tsub_masks)
    idx_of_tsub = {m: i for i, m in enumerate(tsub_masks)}
    
    # Generate all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    k_sub_masks = []
    k_sub_tsub_indices = [] # List of lists of t-subset indices covered by this k-subset
    
    for comb in all_k_subsets:
        mask = 0
        for p in comb:
            mask |= (1 << p)
        k_sub_masks.append(mask)
        
        subs = []
        for sub in itertools.combinations(comb, t):
            sm = 0
            for p in sub:
                sm |= (1 << p)
            subs.append(idx_of_tsub[sm])
        k_sub_tsub_indices.append(subs)
        
    n_ksub = len(all_k_subsets)
    
    # Map from bitmask to index in all_k_subsets for quick lookup
    mask_to_ksub_idx = {m: i for i, m in enumerate(k_sub_masks)}

    # 2. Initial Solution: Greedy Set Cover (Fast)
    # We use a simple greedy to get a valid starting point, but we allow it to be slightly suboptimal
    # to save time, as SA will fix it.
    # Actually, a random valid solution is hard to find. Greedy is safer.
    
    # Optimized Greedy using dynamic updates
    uncovered = np.ones(n_tsub, dtype=bool)
    selected_indices = []
    selected_set = set()
    
    # Precompute tsub_to_blocks for dynamic gain updates
    tsub_to_blocks = [[] for _ in range(n_tsub)]
    for i, indices in enumerate(k_sub_tsub_indices):
        for idx in indices:
            tsub_to_blocks[idx].append(i)
            
    gains = np.full(n_ksub, 35, dtype=np.int32) # C(7,4)=35
    selected_mask = np.zeros(n_ksub, dtype=bool)
    
    # Greedy Phase
    while uncovered.any() and time.time() < deadline - 60.0:
        # Find max gain among unselected
        temp_gains = gains.copy()
        temp_gains[selected_mask] = -1
        best_idx = np.argmax(temp_gains)
        
        if gains[best_idx] <= 0:
            break
            
        selected_indices.append(best_idx)
        selected_set.add(k_sub_masks[best_idx])
        selected_mask[best_idx] = True
        gains[best_idx] = -1
        
        block_tsubs = k_sub_tsub_indices[best_idx]
        for s_idx in block_tsubs:
            if uncovered[s_idx]:
                uncovered[s_idx] = False
                for b_idx in tsub_to_blocks[s_idx]:
                    if not selected_mask[b_idx]:
                        gains[b_idx] -= 1

    # Ensure validity
    if uncovered.any():
        uncovered_indices = np.where(uncovered)[0]
        for idx in uncovered_indices:
            if time.time() > deadline - 55.0:
                break
            tsub_mask = tsub_masks[idx]
            have = [p for p in range(v) if (tsub_mask >> p) & 1]
            need = k - len(have)
            remaining_points = [p for p in range(v) if not ((tsub_mask >> p) & 1)]
            if need > 0:
                extra = random.sample(remaining_points, need)
            else:
                extra = []
            block = have + extra
            block.sort()
            mask = 0
            for p in block:
                mask |= (1 << p)
            if mask not in selected_set:
                selected_set.add(mask)
                b_idx = mask_to_ksub_idx[mask]
                selected_indices.append(b_idx)
                for s_idx in k_sub_tsub_indices[b_idx]:
                    if uncovered[s_idx]:
                        uncovered[s_idx] = False

    # Convert to list of indices
    current_blocks = np.array(selected_indices, dtype=np.int32)
    
    # 3. Simulated Annealing
    # Objective: Minimize cost = n_blocks + lambda * n_uncovered
    # We maintain a fixed-size vector of blocks. Initially size = len(current_blocks).
    # We can try to reduce size by allowing "deletion" moves if coverage remains valid?
    # Standard SA on set cover is tricky with variable size.
    # Strategy: Fix size N. If N is too large, we waste time. If too small, we can't cover.
    # Let's start with N = len(current_blocks) and try to reduce N if we find a valid solution with fewer blocks?
    # No, SA usually operates on a fixed dimension. 
    # Alternative: "Swap and Delete" local search within SA.
    # Let's stick to a fixed size N = len(current_blocks) initially.
    # We will perform swaps: replace block i with block j.
    # We will also perform "deletions" if the solution remains valid? 
    # If we delete a block, N decreases. We can restart SA with smaller N?
    # Better: Allow the vector to shrink.
    
    # Let's implement a hybrid: 
    # 1. SA on fixed size N to improve packing (minimize redundancy).
    # 2. If valid, try to remove blocks.
    # 3. Repeat with smaller N.
    
    # Actually, a simpler robust approach:
    # Run SA to minimize (number of blocks) is hard because it's discrete.
    # Let's minimize a penalty: P = sum(1 for uncovered) * 1000 + len(blocks).
    # But len(blocks) is fixed in the vector.
    
    # Let's try this:
    # Vector B of size M. 
    # Cost = M + 1000 * (number of uncovered t-subsets).
    # Moves:
    # 1. Swap B[i] with random block j.
    # 2. If valid (0 uncovered), try to remove a redundant block. If successful, M decreases.
    
    # Initialize M
    M = len(current_blocks)
    if M == 0:
        return []
        
    best_blocks = current_blocks.copy()
    best_cost = M + 1000 * np.sum(uncovered)
    
    # Precompute cover counts for current solution
    cover_count = np.zeros(n_tsub, dtype=np.int32)
    for b_idx in best_blocks:
        for s_idx in k_sub_tsub_indices[b_idx]:
            cover_count[s_idx] += 1
            
    n_uncovered = np.sum(cover_count == 0)
    current_cost = M + 1000 * n_uncovered
    
    T = 1000.0
    T_min = 1.0
    alpha = 0.995
    
    # To speed up swap evaluation, we need to know the delta in coverage.
    # Removing block i:
    #   -1 to cover_count[s] for s in block i.
    #   If any cover_count[s] becomes 0, n_uncovered increases.
    # Adding block j:
    #   +1 to cover_count[s] for s in block j.
    #   If any cover_count[s] was 0, n_uncovered decreases.
    
    # We can precompute the list of t-subsets for each block (already have k_sub_tsub_indices).
    
    # SA Loop
    while T > T_min and time.time() < deadline - 10.0:
        # Pick a random index to modify
        i = random.randint(0, M - 1)
        old_block = best_blocks[i]
        
        # Pick a new random block
        new_block = random.randint(0, n_ksub - 1)
        
        if new_block == old_block:
            continue
            
        # Calculate delta cost
        # 1. Effect of removing old_block
        #    Delta_uncovered_remove = 0
        #    For s in old_block:
        #       if cover_count[s] == 1: Delta_uncovered_remove += 1
        #       (conceptually cover_count[s] -= 1)
        
        # 2. Effect of adding new_block
        #    Delta_uncovered_add = 0
        #    For s in new_block:
        #       # Note: we must consider the state AFTER removal
        #       # Let's simulate the intermediate state or calculate carefully.
        
        # Efficient calculation:
        # Let S_old = set of tsubs in old_block
        # Let S_new = set of tsubs in new_block
        
        # Uncovered after removal (before add):
        # U_rem = { s in S_old | cover_count[s] == 1 }
        
        # Uncovered before removal: U_cur = { s | cover_count[s] == 0 }
        
        # After removal:
        # cover_count'[s] = cover_count[s] - 1 if s in S_old else cover_count[s]
        # U_rem = { s in S_old | cover_count[s] == 1 }
        # Note: s in U_cur cannot be in S_old if cover_count[s]==1? 
        # If s in U_cur, cover_count[s]=0. If s in S_old, cover_count[s]>=1. Contradiction.
        # So U_rem and U_cur are disjoint.
        
        # After adding new_block:
        # cover_count''[s] = cover_count'[s] + 1 if s in S_new else cover_count'[s]
        # U_final = { s | cover_count''[s] == 0 }
        
        # Let's trace states for specific s:
        # Case 1: s not in S_old and not in S_new.
        #   cc'' = cc. State unchanged.
        
        # Case 2: s in S_old, not in S_new.
        #   cc' = cc - 1. cc'' = cc - 1.
        #   If cc=1: cc''=0. Uncovered increases by 1.
        #   If cc>1: cc''>0. No change.
        
        # Case 3: s not in S_old, in S_new.
        #   cc' = cc. cc'' = cc + 1.
        #   If cc=0: cc''=1. Uncovered decreases by 1.
        #   If cc>0: cc''>0. No change.
        
        # Case 4: s in S_old, in S_new.
        #   cc' = cc - 1. cc'' = cc.
        #   State unchanged.
        
        # So:
        # Delta_uncovered = (# of s in S_old \ S_new with cc[s]==1) - (# of s in S_new \ S_old with cc[s]==0)
        
        # We need sets S_old and S_new.
        # S_old = k_sub_tsub_indices[old_block]
        # S_new = k_sub_tsub_indices[new_block]
        
        # To make this fast, we can't use Python sets for 35 elements every step if SA runs millions of times.
        # But 35 is small. 
        # Let's use lists and a helper.
        
        S_old = k_sub_tsub_indices[old_block]
        S_new = k_sub_tsub_indices[new_block]
        
        # Optimization: Precompute a bitmask or boolean array for each block's t-subsets?
        # We have k_sub_tsub_indices as lists.
        # Let's create a boolean array for S_old and S_new? Too slow to allocate.
        # Let's just iterate.
        
        # 1. Count s in S_old where cc[s]==1 and s not in S_new.
        #    To check "s not in S_new", we can use a set for S_new? 
        #    Creating a set of 35 ints is ~1us. 
        #    SA steps: 100k steps * 1us = 0.1s. Feasible.
        
        set_new = set(S_new)
        
        inc_uncov = 0
        for s in S_old:
            if s not in set_new:
                if cover_count[s] == 1:
                    inc_uncov += 1
                    
        dec_uncov = 0
        # 2. Count s in S_new where cc[s]==0 and s not in S_old.
        #    We need to check s not in S_old.
        #    We can reuse a set for S_old? Or just check cover_count?
        #    If cc[s]==0, it's uncovered. 
        #    If s is in S_old, cc[s] >= 1 (since old_block covers it). 
        #    So if cc[s]==0, s CANNOT be in S_old.
        #    So we just check cc[s]==0 for all s in S_new.
        
        for s in S_new:
            if cover_count[s] == 0:
                dec_uncov += 1
                
        delta_uncov = inc_uncov - dec_uncov
        delta_cost = 1000 * delta_uncov
        
        # Acceptance
        if delta_cost <= 0 or random.random() < np.exp(-delta_cost / T):
            # Apply move
            best_blocks[i] = new_block
            
            # Update cover_count
            for s in S_old:
                cover_count[s] -= 1
            for s in S_new:
                cover_count[s] += 1
                
            current_cost += delta_cost
            if current_cost < best_cost:
                best_cost = current_cost
                # We could save best_blocks here, but best_blocks IS the current state.
                # We just need to track the minimum cost seen.
                
        T *= alpha
        
        # Periodic Pruning / Size Reduction
        # If we have a valid solution (n_uncovered == 0), try to remove redundant blocks.
        # This reduces M.
        # We do this every N steps or if cost is low.
        if current_cost == M and time.time() < deadline - 15.0:
            # Valid solution found. Try to shrink.
            # Check for redundant blocks.
            # A block is redundant if all its t-subsets have cover_count > 1.
            # If we remove it, cover_count drops by 1, but stays > 0.
            
            # We need to be careful: removing a block changes cover_counts.
            # We can try to remove one block. If valid, keep it.
            
            # Find a candidate
            candidate_idx = -1
            for idx in range(M - 1, -1, -1):
                b = best_blocks[idx]
                is_redundant = True
                for s in k_sub_tsub_indices[b]:
                    if cover_count[s] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    candidate_idx = idx
                    break
                    
            if candidate_idx != -1:
                # Remove it
                b_to_remove = best_blocks[candidate_idx]
                for s in k_sub_tsub_indices[b_to_remove]:
                    cover_count[s] -= 1
                    
                # Update M
                M -= 1
                best_blocks = np.delete(best_blocks, candidate_idx)
                current_cost = M # Since it was valid and remains valid
                if current_cost < best_cost:
                    best_cost = current_cost
                    # Reset T slightly to explore new space?
                    # T = max(T, 10.0)
                
    # 4. Final Cleanup
    # Ensure the best solution found is valid.
    # Rebuild cover_count for best_blocks to be sure.
    cover_count_final = np.zeros(n_tsub, dtype=np.int32)
    for b_idx in best_blocks:
        for s_idx in k_sub_tsub_indices[b_idx]:
            cover_count_final[s_idx] += 1
            
    if np.any(cover_count_final == 0):
        # If SA ended in an invalid state (which can happen if we don't enforce validity strictly 
        # or if the "best" was tracked incorrectly), we must repair.
        # The SA above minimizes cost = M + 1000*uncov. 
        # If best_cost < M_final, it means we found a valid solution earlier?
        # No, best_cost tracks the cost of the *current* state if it improves.
        # If current state is invalid, cost > M.
        # We should keep the best VALID solution separately.
        
        # Let's re-evaluate: 
        # My SA loop updates best_cost only if current_cost < best_cost.
        # current_cost includes the penalty.
        # So if best_cost corresponds to a valid solution, best_cost <= M_at_that_time.
        # However, I didn't store the blocks for the best valid solution separately from the current state.
        # If the current state is invalid, I should revert to the last known valid state?
        # Or just repair the current state.
        
        # Repairing is safer and faster than reverting.
        uncovered_indices = np.where(cover_count_final == 0)[0]
        for idx in uncovered_indices:
            if time.time() > deadline - 2.0:
                break
            tsub_mask = tsub_masks[idx]
            have = [p for p in range(v) if (tsub_mask >> p) & 1]
            need = k - len(have)
            remaining_points = [p for p in range(v) if not ((tsub_mask >> p) & 1)]
            if need > 0:
                extra = random.sample(remaining_points, need)
            else:
                extra = []
            block = have + extra
            block.sort()
            mask = 0
            for p in block:
                mask |= (1 << p)
            b_idx = mask_to_ksub_idx[mask]
            # Add to best_blocks
            best_blocks = np.append(best_blocks, b_idx)
            for s_idx in k_sub_tsub_indices[b_idx]:
                cover_count_final[s_idx] += 1

    # Final Pruning on the repaired solution
    changed = True
    while changed and time.time() < deadline - 1.0:
        changed = False
        # Shuffle to break symmetry
        random.shuffle(best_blocks)
        
        for i in range(len(best_blocks) - 1, -1, -1):
            if time.time() > deadline - 1.0:
                break
            b_idx = best_blocks[i]
            is_redundant = True
            for s_idx in k_sub_tsub_indices[b_idx]:
                if cover_count_final[s_idx] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                for s_idx in k_sub_tsub_indices[b_idx]:
                    cover_count_final[s_idx] -= 1
                best_blocks = np.delete(best_blocks, i)
                changed = True

    # Convert to result
    result = []
    for idx in best_blocks:
        result.append(list(all_k_subsets[idx]))
        
    return result

# EVOLVE-BLOCK-END
