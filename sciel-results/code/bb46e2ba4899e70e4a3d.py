import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Leave 2s buffer for final checks and return
    
    # 1. Precomputation
    # Generate all t-subsets as bitmasks
    tsub_masks = []
    for sub in itertools.combinations(range(v), t):
        m = 0
        for p in sub:
            m |= (1 << p)
        tsub_masks.append(m)
    n_tsub = len(tsub_masks)
    
    # Generate all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_ksub = len(all_k_subsets)
    
    # Map from bitmask to index in all_k_subsets
    k_sub_masks = []
    mask_to_ksub_idx = {}
    for i, comb in enumerate(all_k_subsets):
        mask = 0
        for p in comb:
            mask |= (1 << p)
        k_sub_masks.append(mask)
        mask_to_ksub_idx[mask] = i
        
    # Precompute t-subset indices covered by each k-subset
    # k_sub_tsub_indices[i] is a list of indices into tsub_masks
    k_sub_tsub_indices = []
    # Also precompute a boolean matrix or bitset? 
    # For v=19, k=7, t=4: n_tsub = C(19,4) = 3876. n_ksub = C(19,7) = 50388.
    # A boolean matrix of size n_ksub x n_tsub is 50388 * 3876 ~ 195M bits ~ 24MB. Feasible.
    # But we can also just store lists. Lists are sparse (35 entries per block).
    
    # Let's use lists for memory efficiency and cache locality if we access sequentially.
    # However, for fast "is this tsub covered by block X" checks, a set or bitset is better.
    # Let's stick to lists of indices for the main logic, as 35 is small.
    
    # We need to map a t-subset mask to its index quickly.
    idx_of_tsub = {m: i for i, m in enumerate(tsub_masks)}
    
    for comb in all_k_subsets:
        subs = []
        for sub in itertools.combinations(comb, t):
            sm = 0
            for p in sub:
                sm |= (1 << p)
            subs.append(idx_of_tsub[sm])
        k_sub_tsub_indices.append(subs)
        
    # Precompute "conflict" or "overlap" info?
    # For swap moves, we need to know how coverage changes.
    # We maintain a global cover_count array.
    
    # Helper function to calculate delta uncovered for a swap (old_block -> new_block)
    # Given current cover_count.
    def calc_swap_delta(old_idx, new_idx, cover_count):
        # Returns: (delta_uncovered, list_of_changes)
        # We need to simulate the effect.
        # old_tsubs = k_sub_tsub_indices[old_idx]
        # new_tsubs = k_sub_tsub_indices[new_idx]
        
        # To be fast, we can iterate through the smaller list? They are both size 35.
        # We need to count:
        # 1. tsubs in old but not in new, where cover_count == 1 (will become uncovered)
        # 2. tsubs in new but not in old, where cover_count == 0 (will become covered)
        
        # Using sets for intersection/difference is fast for size 35.
        set_old = set(k_sub_tsub_indices[old_idx])
        set_new = set(k_sub_tsub_indices[new_idx])
        
        # tsubs that will lose coverage
        # s in set_old - set_new and cover_count[s] == 1
        # tsubs that will gain coverage
        # s in set_new - set_old and cover_count[s] == 0
        
        inc_uncov = 0
        dec_uncov = 0
        
        # Optimization: Iterate over the difference sets
        # set_old - set_new
        for s in set_old:
            if s not in set_new:
                if cover_count[s] == 1:
                    inc_uncov += 1
                    
        for s in set_new:
            if s not in set_old:
                if cover_count[s] == 0:
                    dec_uncov += 1
                    
        delta = inc_uncov - dec_uncov
        return delta

    # Greedy Construction with Randomization (GRASP)
    def greedy_construct(alpha):
        """
        Builds a solution using randomized greedy.
        alpha controls randomness: 0 = deterministic greedy, 1 = random.
        """
        uncovered = np.ones(n_tsub, dtype=bool)
        selected = []
        selected_set = set()
        
        # Precompute gains? 
        # In GRASP, we pick from a candidate list of best alpha-fraction.
        # Since n_ksub is 50k, sorting every step is expensive.
        # We can maintain a heap or just scan if we do it smartly.
        # But we only do this for a few iterations (e.g., 10-20 starts).
        # Let's just do a standard greedy but with random tie-breaking or random selection from top candidates.
        
        # To speed up, we'll maintain a list of uncovered tsubs and their potential blocks?
        # No, let's just use the same gain update mechanism as before, but it's O(N) per step.
        # 50k * 150 steps = 7.5M operations per construct. 20 constructs = 150M. Might be tight in 118s.
        # Let's optimize the greedy.
        
        # Alternative: Use a simpler heuristic for the start.
        # Pick random blocks until valid? No, too slow to reach valid.
        
        # Let's stick to a fast greedy.
        # We can precompute for each tsub, which blocks cover it.
        tsub_to_blocks = [[] for _ in range(n_tsub)]
        for i, indices in enumerate(k_sub_tsub_indices):
            for idx in indices:
                tsub_to_blocks[idx].append(i)
                
        # Initial gains
        gains = np.full(n_ksub, 35, dtype=np.int32)
        selected_mask = np.zeros(n_ksub, dtype=bool)
        
        while uncovered.any() and time.time() < deadline - 30.0:
            # Find candidates with max gain
            # To allow randomness, we can pick randomly from the top K% of gains?
            # Or just pick the max gain, but break ties randomly?
            # Better: Pick randomly from blocks that cover the "hardest" uncovered tsub?
            
            # Let's find the uncovered tsub with the fewest available covering blocks (most constrained).
            # If any uncovered tsub has 0 available blocks, we are stuck (shouldn't happen if we add blocks).
            
            # Actually, standard greedy is: pick block with max new coverage.
            # Let's add noise: gain' = gain + random_noise.
            
            # To make it fast, we'll just pick the argmax of (gains + noise) among unselected.
            # But scanning 50k items every step is slow.
            
            # Optimization: Maintain a max-heap of (gain, index).
            # But gains change.
            
            # Let's just do a linear scan but in C (numpy).
            # temp_gains = gains.copy()
            # temp_gains[selected_mask] = -1
            # best_idx = np.argmax(temp_gains)
            
            # This is O(N) per step. 150 steps * 50k = 7.5M.
            # 20 starts * 7.5M = 150M ops. In Python/Numpy, this is ~1-2 seconds per start?
            # 20 starts = 20-40s. Feasible.
            
            # Add randomness:
            # Instead of argmax, pick from top 10%?
            # Or add random noise to gains?
            
            # Let's add small random noise to gains to diversify.
            noise = np.random.uniform(0, 5, size=n_ksub) # Scale noise relative to max gain 35
            # But we only want to consider unselected.
            # We can just set selected gains to -1 after adding noise?
            # No, we need to update gains dynamically.
            
            # Let's just use the standard greedy for the first few, then randomize?
            # Actually, a common GRASP strategy:
            # 1. Calculate gains.
            # 2. Find max gain G_max.
            # 3. Select a block uniformly at random from those with gain >= alpha * G_max.
            
            # This requires finding all blocks with gain >= threshold.
            # We can do this with numpy masking.
            
            # But first, we need to update gains.
            # The current code updates gains dynamically. That's good.
            
            # Let's implement the alpha-GRASP selection.
            # 1. Find max gain among unselected.
            # We need to track max gain efficiently.
            # Since we update gains by decrementing, the max gain can only decrease or stay same?
            # No, if we select a block, we decrement gains of other blocks.
            # So max gain is non-increasing.
            
            # We can maintain a list of blocks sorted by gain?
            # Or just scan.
            
            # Let's scan to find max_gain and candidate list.
            max_gain = -1
            candidates = []
            
            # This loop is in Python, so it's slow.
            # Let's use numpy to find indices where gain > 0 and not selected.
            # valid_mask = ~selected_mask
            # active_gains = np.where(valid_mask, gains, -1)
            # max_g = np.max(active_gains)
            
            # If max_g < 0, break.
            
            # Candidates = np.where(active_gains >= alpha * max_g)[0]
            # Pick random from Candidates.
            
            # This is O(N) per step.
            
            active_gains = np.where(selected_mask, -1, gains)
            max_g = np.max(active_gains)
            
            if max_g < 0:
                break
                
            threshold = alpha * max_g
            candidates = np.where(active_gains >= threshold)[0]
            
            # Pick random candidate
            best_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            selected.append(best_idx)
            selected_mask[best_idx] = True
            gains[best_idx] = -1
            
            # Update gains and uncovered
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
                if time.time() > deadline - 25.0:
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
                    selected.append(b_idx)
                    for s_idx in k_sub_tsub_indices[b_idx]:
                        if uncovered[s_idx]:
                            uncovered[s_idx] = False

        return np.array(selected, dtype=np.int32)

    # Local Search: Swap and Prune
    def local_search(blocks):
        M = len(blocks)
        if M == 0:
            return blocks, 0
            
        # Calculate cover counts
        cover_count = np.zeros(n_tsub, dtype=np.int32)
        for b_idx in blocks:
            for s_idx in k_sub_tsub_indices[b_idx]:
                cover_count[s_idx] += 1
                
        n_uncovered = np.sum(cover_count == 0)
        if n_uncovered > 0:
            return blocks, n_uncovered # Invalid
            
        # Try to improve
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # 1. Prune redundant blocks
            # A block is redundant if all its tsubs have cover_count > 1.
            # We can shuffle to break symmetry.
            random.shuffle(blocks)
            
            for i in range(len(blocks) - 1, -1, -1):
                b_idx = blocks[i]
                is_redundant = True
                for s_idx in k_sub_tsub_indices[b_idx]:
                    if cover_count[s_idx] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for s_idx in k_sub_tsub_indices[b_idx]:
                        cover_count[s_idx] -= 1
                    blocks = np.delete(blocks, i)
                    improved = True
                    # After pruning, we might be able to prune more, so restart loop?
                    # Or continue scanning?
                    # Let's restart to ensure we find all redundancies in the new configuration.
                    break # Break inner loop, restart outer while
            
            # 2. Try swaps to reduce redundancy or enable further pruning?
            # Swaps in a valid solution don't change validity, but might allow pruning.
            # So we want to find swaps that increase "slack" (cover counts).
            # Or just random swaps and then prune.
            
            if len(blocks) > 0:
                # Pick random block to swap
                i = random.randint(0, len(blocks) - 1)
                old_block = blocks[i]
                
                # Pick random new block
                new_block = random.randint(0, n_ksub - 1)
                
                if new_block != old_block:
                    # Check if swap keeps solution valid?
                    # We know it's valid now.
                    # After swap, is it still valid?
                    # We need to check if any tsub becomes uncovered.
                    # This is the same calc as SA delta.
                    # If delta_uncovered > 0, it becomes invalid.
                    # We only accept if delta_uncovered == 0 (stays valid).
                    
                    delta = calc_swap_delta(old_block, new_block, cover_count)
                    
                    if delta == 0:
                        # Apply swap
                        blocks[i] = new_block
                        # Update cover_count
                        S_old = k_sub_tsub_indices[old_block]
                        S_new = k_sub_tsub_indices[new_block]
                        for s in S_old:
                            cover_count[s] -= 1
                        for s in S_new:
                            cover_count[s] += 1
                        
                        # This swap might have created redundancy.
                        # We don't know yet. The next iteration of the while loop will try to prune.
                        # So we set improved = True to continue?
                        # No, the swap itself doesn't reduce size.
                        # But it might enable pruning.
                        # So we should continue the search.
                        # However, if we just swap and don't prune, we might cycle.
                        # Let's force a prune attempt after swap.
                        # The loop structure:
                        # While improved:
                        #   Prune (if success, improved=True, restart)
                        #   Swap (if valid, apply, improved=True? No, size didn't change)
                        #   If no prune and no swap improved things, stop.
                        
                        # Let's say a swap is "useful" if it allows a prune.
                        # We can't know that immediately.
                        # So we do:
                        # 1. Prune pass. If prune, restart.
                        # 2. Swap pass. Try K swaps. If any swap leads to a prune, keep it.
                        #    Else, revert.
                        
                        # This is complex.
                        # Simpler:
                        # 1. Prune. If success, restart.
                        # 2. Try 100 random swaps. For each, if valid, apply, then try to prune.
                        #    If prune succeeds, keep the swap and the prune, restart.
                        #    If prune fails, revert the swap.
                        # 3. If no swap+prune succeeds, stop.
                        
                        pass # We'll restructure the loop below

        # Restructured Local Search
        def ls_loop(blocks):
            M = len(blocks)
            cover_count = np.zeros(n_tsub, dtype=np.int32)
            for b_idx in blocks:
                for s_idx in k_sub_tsub_indices[b_idx]:
                    cover_count[s_idx] += 1
                    
            # Verify valid
            if np.any(cover_count == 0):
                return blocks
                
            while time.time() < deadline - 5.0:
                # 1. Prune
                pruned = False
                random.shuffle(blocks)
                for i in range(len(blocks) - 1, -1, -1):
                    b_idx = blocks[i]
                    is_redundant = True
                    for s_idx in k_sub_tsub_indices[b_idx]:
                        if cover_count[s_idx] <= 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        for s_idx in k_sub_tsub_indices[b_idx]:
                            cover_count[s_idx] -= 1
                        blocks = np.delete(blocks, i)
                        pruned = True
                        break # Restart prune/swap cycle
                
                if pruned:
                    continue # Go to top of while loop
                
                # 2. Swap + Prune
                # Try a few swaps
                swap_success = False
                for _ in range(20): # Limit swaps per iteration
                    if time.time() > deadline - 5.0:
                        break
                    
                    i = random.randint(0, len(blocks) - 1)
                    old_block = blocks[i]
                    new_block = random.randint(0, n_ksub - 1)
                    
                    if new_block == old_block:
                        continue
                    
                    delta = calc_swap_delta(old_block, new_block, cover_count)
                    
                    if delta == 0:
                        # Apply swap
                        blocks[i] = new_block
                        S_old = k_sub_tsub_indices[old_block]
                        S_new = k_sub_tsub_indices[new_block]
                        for s in S_old:
                            cover_count[s] -= 1
                        for s in S_new:
                            cover_count[s] += 1
                        
                        # Try to prune
                        pruned = False
                        random.shuffle(blocks)
                        for j in range(len(blocks) - 1, -1, -1):
                            b_idx = blocks[j]
                            is_redundant = True
                            for s_idx in k_sub_tsub_indices[b_idx]:
                                if cover_count[s_idx] <= 1:
                                    is_redundant = False
                                    break
                            if is_redundant:
                                for s_idx in k_sub_tsub_indices[b_idx]:
                                    cover_count[s_idx] -= 1
                                blocks = np.delete(blocks, j)
                                pruned = True
                                break
                        
                        if pruned:
                            swap_success = True
                            break # Break swap loop, go to top of while loop
                        else:
                            # Revert swap
                            blocks[i] = old_block
                            for s in S_new:
                                cover_count[s] -= 1
                            for s in S_old:
                                cover_count[s] += 1
                    
                if not swap_success:
                    break # No improvement found
                    
            return blocks

    # Main GRASP Loop
    best_blocks = None
    best_size = float('inf')
    
    # Generate diverse starts
    num_starts = 20
    for i in range(num_starts):
        if time.time() > deadline - 20.0:
            break
            
        # Vary alpha for diversity
        # Early starts: more greedy (low alpha). Later starts: more random (high alpha).
        alpha = 0.1 + 0.8 * (i / num_starts)
        
        start_blocks = greedy_construct(alpha)
        
        # Local Search
        refined_blocks = ls_loop(start_blocks)
        
        if len(refined_blocks) < best_size:
            best_size = len(refined_blocks)
            best_blocks = refined_blocks.copy()
            
        # If we hit the theoretical lower bound or a very good bound, we might stop early?
        # Lower bound is 131. If we get close, keep going.
        
    # Final Prune on best
    if best_blocks is not None:
        # Do one more pass of pruning just in case
        cover_count = np.zeros(n_tsub, dtype=np.int32)
        for b_idx in best_blocks:
            for s_idx in k_sub_tsub_indices[b_idx]:
                cover_count[s_idx] += 1
                
        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            random.shuffle(best_blocks)
            for i in range(len(best_blocks) - 1, -1, -1):
                b_idx = best_blocks[i]
                is_redundant = True
                for s_idx in k_sub_tsub_indices[b_idx]:
                    if cover_count[s_idx] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for s_idx in k_sub_tsub_indices[b_idx]:
                        cover_count[s_idx] -= 1
                    best_blocks = np.delete(best_blocks, i)
                    changed = True

    # Convert to result
    result = []
    for idx in best_blocks:
        result.append(list(all_k_subsets[idx]))
        
    return result

# EVOLVE-BLOCK-END
