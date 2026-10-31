import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for final checks and return
    
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
    mask_to_ksub_idx = {m: i for i, m in enumerate(k_sub_masks)}
    
    # Precompute intersection counts? 
    # For the "quasi-symmetric" initialization, we want blocks that intersect others in a "balanced" way.
    # A simpler heuristic for initialization: Random Sampling with Conflict Avoidance.
    # Or: Start with a random large set of blocks, then prune.
    
    # Let's try a "Randomized Greedy with Diversity" initialization.
    # 1. Pick a random block.
    # 2. For subsequent blocks, prefer those that cover many *new* t-subsets 
    #    but also penalize those that overlap too much with existing blocks (to avoid redundancy).
    
    # Actually, a better structural approach for covering designs:
    # Start with a random set of ~1.2 * estimated_blocks blocks.
    # Then perform local search (swaps) to minimize count while maintaining coverage.
    
    # Estimate blocks: C(v,t) / C(k,t) = 27132 / 35 ~ 775. But lower bound is 131.
    # Let's start with a slightly higher number to ensure coverage, then prune.
    # Or use the greedy set cover to get a valid (but maybe suboptimal) solution, 
    # then apply local search. The current program IS greedy set cover but slow.
    
    # Key Improvement: Vectorized Gain Calculation.
    # We can store the coverage state as a bitset or boolean array.
    # To speed up greedy:
    # Instead of checking all 50k blocks every step (which is 50k * 35 ops),
    # we can maintain a "candidate pool" or use a heap.
    
    # However, the most effective structural change suggested is **Simulated Annealing** 
    # or **Local Search** on a fixed-size vector.
    
    # Let's implement:
    # Phase 1: Fast Greedy Construction (optimized) to get a valid covering.
    # Phase 2: Local Search (1-for-1 swaps and deletions) to reduce size.
    
    # Phase 1: Optimized Greedy
    # We need to find the block with max gain quickly.
    # Gain(block) = number of t-subsets in block that are currently uncovered.
    
    # State:
    # uncovered: np.bool array of size n_tsub
    # selected_blocks: list of indices into all_k_subsets
    # block_to_count: dict or array to track if a block is selected? 
    # Actually, we just need the list of selected blocks to check duplicates.
    
    uncovered = np.ones(n_tsub, dtype=bool)
    selected_indices = []
    selected_set = set()
    
    # To speed up the greedy step, we can precompute the list of t-subset indices for each k-subset.
    # We already have k_sub_tsub_indices.
    
    # Optimization: 
    # Instead of iterating all 50k blocks every time, we can use a "bucket" approach or just accept the cost if we can make the inner loop fast.
    # 50k blocks * 35 t-subsets = 1.75M operations. 
    # In Python, a loop of 1.75M is ~0.1s. 
    # 150 blocks * 0.1s = 15s. This is feasible!
    # The previous program was slow because it likely used inefficient structures or didn't vectorize the sum.
    # Let's use numpy sum for the gain calculation.
    
    # Convert k_sub_tsub_indices to a list of numpy arrays for faster slicing/summing?
    # Or just use a flat array with offsets as in the previous attempt, but ensure the sum is fast.
    
    # Let's try the flat array approach again, but carefully.
    all_tsub_indices_flat = []
    offsets = [0]
    for indices in k_sub_tsub_indices:
        all_tsub_indices_flat.extend(indices)
        offsets.append(len(all_tsub_indices_flat))
    
    all_tsub_indices_arr = np.array(all_tsub_indices_flat, dtype=np.int32)
    offsets_arr = np.array(offsets, dtype=np.int32)
    
    # Greedy Loop
    while uncovered.any() and time.time() < deadline - 30.0: # Leave 30s for local search
        # Calculate gains
        # We can't easily vectorize variable length sums in pure numpy without padding.
        # But we can do it in a loop that is slightly optimized.
        
        # Alternative: Use a "score" array that is updated incrementally?
        # When we add a block, we decrement the "potential gain" of other blocks that share t-subsets?
        # This is the standard "lazy deletion" or "dynamic update" for greedy set cover.
        
        # Let's implement dynamic updates.
        # gain[i] = sum(1 for j in k_sub_tsub_indices[i] if uncovered[j])
        # Initially, gain[i] = len(k_sub_tsub_indices[i]) = 35.
        
        # When we select block b:
        # For each t-subset s in block b:
        #   If s was uncovered:
        #     For every block i that covers s:
        #       gain[i] -= 1
        #     Mark s as covered.
        
        # This requires knowing, for each t-subset, which blocks cover it.
        # tsub_to_blocks: list of lists of block indices.
        
        # Precomputation for dynamic update:
        # tsub_to_blocks = [[] for _ in range(n_tsub)]
        # for i, indices in enumerate(k_sub_tsub_indices):
        #     for idx in indices:
        #         tsub_to_blocks[idx].append(i)
        
        # This precomputation is 50k * 35 = 1.75M appends. Fast enough.
        
        if 'tsub_to_blocks' not in locals():
            tsub_to_blocks = [[] for _ in range(n_tsub)]
            for i, indices in enumerate(k_sub_tsub_indices):
                for idx in indices:
                    tsub_to_blocks[idx].append(i)
            
            gains = np.full(n_ksub, 35, dtype=np.int32) # C(7,4)=35
            # Wait, gains should be 0 if the block is already selected? No, we don't re-select.
            # We just need to pick the max gain among unselected blocks.
            # We can maintain a set of selected blocks and skip them, or set their gain to -1.
            
        # In the loop:
        # Find max gain among unselected blocks.
        # To avoid scanning all 50k blocks to find max (which is 50k ops), 
        # we can use a heap. 
        # But gains change dynamically. 
        # A heap with lazy deletion is good.
        
        # Let's stick to the simple scan first, but optimize the inner loop of the scan.
        # Scanning 50k ints to find max is ~0.005s in numpy? 
        # np.max(gains) is fast. 
        # The bottleneck is updating gains.
        
        # Update cost: 
        # Selected block covers 35 t-subsets.
        # Each t-subset is covered by C(v-t, k-t) / C(v, k) * n_ksub? 
        # Number of blocks covering a specific t-subset: C(19-4, 7-4) = C(15, 3) = 455.
        # So updating gains involves touching 35 * 455 ~ 16k entries.
        # 16k updates per step. 150 steps * 16k = 2.4M updates. Very fast.
        
        # So the algorithm is:
        # 1. Init gains = 35 for all.
        # 2. Loop:
        #    a. Find block b with max gain (among unselected).
        #    b. Select b. Set gain[b] = -1.
        #    c. For each t-subset s in b:
        #       If s is uncovered:
        #          Mark s covered.
        #          For each block i in tsub_to_blocks[s]:
        #             If i is not selected:
        #                gains[i] -= 1
    
        # We need to know if a block is selected. 
        # selected_mask: np.bool array of size n_ksub.
        
        selected_mask = np.zeros(n_ksub, dtype=bool)
        
        while uncovered.any() and time.time() < deadline - 30.0:
            # Find max gain
            # We can use np.argmax on a masked array? 
            # Or just loop? 
            # np.max(gains[~selected_mask]) is fast.
            
            valid_gains = gains[~selected_mask]
            if valid_gains.size == 0:
                break
            max_gain = np.max(valid_gains)
            if max_gain == 0:
                break
            
            # Find the index
            # Get indices of all unselected blocks
            unselected_indices = np.where(~selected_mask)[0]
            # Find which of these has the max gain
            # This is a bit slow if done naively.
            
            # Better: 
            # Create a temporary array with -1 for selected, gains for others.
            temp_gains = gains.copy()
            temp_gains[selected_mask] = -1
            best_idx = np.argmax(temp_gains)
            
            # Add block
            selected_indices.append(best_idx)
            selected_mask[best_idx] = True
            gains[best_idx] = -1 # Prevent re-selection
            
            # Update gains for blocks sharing t-subsets with best_idx
            block_tsubs = k_sub_tsub_indices[best_idx]
            for s_idx in block_tsubs:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    # Update all blocks covering s_idx
                    for b_idx in tsub_to_blocks[s_idx]:
                        if not selected_mask[b_idx]:
                            gains[b_idx] -= 1
            
            # Check time
            if time.time() > deadline - 30.0:
                break
    
    # Phase 2: Repair
    # If we exited early, cover remaining uncovered t-subsets.
    uncovered_indices = np.where(uncovered)[0]
    for idx in uncovered_indices:
        if time.time() > deadline - 5.0:
            break
        tsub_mask = tsub_masks[idx]
        have = [p for p in range(v) if (tsub_mask >> p) & 1]
        need = k - len(have)
        remaining_points = [p for p in range(v) if not ((tsub_mask >> p) & 1)]
        
        # Pick random extra points
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
            # We don't strictly need to update gains/state here as we are done with greedy,
            # but we must ensure the final list is valid.
            
    # Phase 3: Local Search (1-for-1 Swaps and Deletions)
    # Goal: Reduce the number of blocks while maintaining coverage.
    
    # Current state:
    # selected_indices: list of block indices.
    # We need to track coverage counts to know if a block can be removed or swapped.
    
    # Rebuild coverage counts from scratch for the current selected set.
    cover_count = np.zeros(n_tsub, dtype=np.int32)
    for b_idx in selected_indices:
        for s_idx in k_sub_tsub_indices[b_idx]:
            cover_count[s_idx] += 1
            
    # Check if valid
    if np.any(cover_count == 0):
        # Repair failed or incomplete. Try to fix by adding blocks for uncovered.
        # This shouldn't happen if Phase 1/2 worked, but just in case.
        uncovered_indices = np.where(cover_count == 0)[0]
        for idx in uncovered_indices:
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
                     cover_count[s_idx] += 1

    # Local Search Loop
    # Try to remove a block if it doesn't uncover any t-subset (i.e., all its t-subsets are covered by others).
    # Try to swap a block with another if it reduces count? No, swap keeps count same.
    # Swap is useful if it allows *subsequent* removals.
    # Simpler: Just try to remove redundant blocks repeatedly.
    
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        # Shuffle to break symmetry
        random.shuffle(selected_indices)
        
        for i in range(len(selected_indices) - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break
                
            b_idx = selected_indices[i]
            # Check if this block is essential
            # Essential if there exists a t-subset in it with cover_count == 1.
            
            # Optimization: Precompute which t-subsets are "critical" (count == 1)?
            # We can just check the t-subsets of this block.
            
            is_redundant = True
            for s_idx in k_sub_tsub_indices[b_idx]:
                if cover_count[s_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove it
                selected_indices.pop(i)
                selected_set.remove(k_sub_masks[b_idx])
                for s_idx in k_sub_tsub_indices[b_idx]:
                    cover_count[s_idx] -= 1
                changed = True
    
    # Final Conversion
    result = []
    for idx in selected_indices:
        result.append(list(all_k_subsets[idx]))
        
    return result

# EVOLVE-BLOCK-END
