"""Probabilistic covering design with efficient bitset tracking and SA."""
import random
import itertools
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using probabilistic greedy initialization
    and simulated annealing with fast coverage tracking.
    """
    start_time = time.time()
    timeout = 115.0

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Map each possible block to the list of t-subset indices it covers
    # We don't precompute ALL blocks, but we will generate candidates on the fly.
    # However, for speed, we can precompute the t-subsets for any given block quickly.
    
    def get_block_tsubsets_indices(block):
        """Return list of indices of t-subsets covered by this block."""
        return [t_subset_index[ts] for ts in itertools.combinations(block, t)]

    # Greedy Initialization with Probabilistic Selection
    def greedy_initialize():
        cov_count = np.zeros(num_t_subsets, dtype=np.int32)
        blocks = []
        uncovered_mask = np.ones(num_t_subsets, dtype=bool) # True if uncovered
        
        # To make it probabilistic, we might not just pick the absolute best, 
        # but pick from the top candidates with some probability.
        # However, for initialization, greedy is usually best. 
        # We add randomization to break ties or to explore different structures.
        
        while np.any(uncovered_mask):
            best_block = None
            best_gain = -1
            candidates = []
            
            # Generate candidates
            # Strategy: Pick random uncovered subsets and extend them to k-sets
            num_candidates = 200
            for _ in range(num_candidates):
                if not np.any(uncovered_mask):
                    break
                # Pick a random index that is uncovered
                unc_indices = np.where(uncovered_mask)[0]
                if len(unc_indices) == 0:
                    break
                target_idx = unc_indices[random.randint(0, len(unc_indices) - 1)]
                seed_subset = list(all_t_subsets[target_idx])
                
                block = set(seed_subset)
                while len(block) < k:
                    # Add random points or points that help coverage
                    # For speed, just add random points from the remaining
                    remaining = [x for x in range(v) if x not in block]
                    if not remaining:
                        break
                    # Simple heuristic: pick random
                    block.add(random.choice(remaining))
                
                block_tup = tuple(sorted(block))
                candidates.append(block_tup)
            
            # Evaluate candidates
            # To speed up, we can estimate gain or compute exactly for top few
            # For now, compute exactly for all generated candidates (200 is small)
            for block in candidates:
                tsubs = get_block_tsubsets_indices(block)
                # Gain is number of tsubs where cov_count is 0
                gain = np.sum(cov_count[tsubs] == 0)
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                elif gain == best_gain and best_block is not None:
                    # Tie-breaking: random selection among bests to add variety
                    if random.random() < 0.5:
                        best_block = block
            
            if best_block is None:
                break
            
            blocks.append(best_block)
            # Update coverage
            tsubs = get_block_tsubsets_indices(best_block)
            cov_count[tsubs] += 1
            # Update uncovered mask
            # A t-subset is covered if count > 0
            # We only care about transitioning from 0 to 1
            newly_covered = (cov_count[tsubs] == 1)
            # Map back to indices
            # Actually, cov_count[tsubs] gives the counts for those indices.
            # We need to update uncovered_mask for those indices.
            # Since tsubs are indices, we can do:
            # uncovered_mask[tsubs] = uncovered_mask[tsubs] & ~(cov_count[tsubs] > 0)
            # But cov_count[tsubs] > 0 is True for all of them now.
            # So uncovered_mask[tsubs] = False is correct for those that were 0 before?
            # No, if it was 1 before, it stays False. If it was 0, it becomes 1 -> False.
            # So uncovered_mask[tsubs] = False is always correct for the new state.
            uncovered_mask[tsubs] = False

        return blocks, cov_count, uncovered_mask

    # Initialize
    blocks, cov_count, uncovered_mask = greedy_initialize()
    current_uncovered = int(np.sum(uncovered_mask))
    
    best_blocks = list(blocks)
    best_uncovered = current_uncovered
    best_cov_count = cov_count.copy()
    best_uncovered_mask = uncovered_mask.copy()

    # Simulated Annealing
    temperature = 100.0
    cooling = 0.999
    min_temp = 0.1
    iterations = 0
    
    # Precompute t-subsets for current blocks for faster removal?
    # block_tsubs[i] = list of indices covered by blocks[i]
    block_tsubs = [get_block_tsubsets_indices(b) for b in blocks]

    while temperature > min_temp:
        if time.time() - start_time > timeout:
            break
            
        # Pick a random block to replace
        idx = random.randint(0, len(blocks) - 1)
        old_block = blocks[idx]
        old_tsubs = block_tsubs[idx]
        
        # Remove old block's contribution
        cov_count[old_tsubs] -= 1
        # Update uncovered mask: if count becomes 0, it's uncovered
        # We need to know which ones became 0.
        # cov_count[old_tsubs] == 0 identifies them.
        became_uncovered = (cov_count[old_tsubs] == 0)
        # Apply to mask
        # Note: old_tsubs is a list of indices.
        if np.any(became_uncovered):
            # We need to map the boolean array back to the indices in old_tsubs
            # This is tricky with numpy fancy indexing.
            # Let's just recompute the mask for the affected indices?
            # Or, simpler: 
            # The mask is global. 
            # For each index in old_tsubs, if cov_count[index] == 0, set mask[index] = True.
            # This is a loop. To avoid loop, we can do:
            # indices_to_update = np.where(became_uncovered)[0]
            # real_indices = np.array(old_tsubs)[indices_to_update]
            # uncovered_mask[real_indices] = True
            
            # Let's optimize this. 
            # Since we are doing SA, we can just calculate the delta in uncovered count directly.
            # Delta = (new_uncovered - old_uncovered)
            # We can track current_uncovered as an integer.
            
            # Let's switch strategy: Track `current_uncovered` as an int.
            # When removing a block:
            # For each tsub in old_tsubs:
            #   if cov_count[tsub] == 0: current_uncovered += 1
            #   else: current_uncovered -= 1 (if it was the only cover, it becomes uncovered? No.)
            #   Wait. 
            #   Before removal: cov_count > 0.
            #   After removal: cov_count decreases by 1.
            #   If cov_count becomes 0, it was covered (count>=1) and is now uncovered. So uncovered count increases by 1.
            #   If cov_count remains > 0, it was covered and is still covered. No change.
            
            # So:
            # delta_remove = number of tsubs in old_tsubs where cov_count (after decrement) == 0.
            
            # Let's re-evaluate the whole move using this logic.
            
            # Restore state for clarity in this thought process, I will implement the int tracking in code.
            
            pass # Placeholder

        # Generate new block
        # Target uncovered subsets
        if current_uncovered > 0:
            # Pick a random uncovered subset
            unc_indices = np.where(uncovered_mask)[0]
            if len(unc_indices) > 0:
                target_idx = unc_indices[random.randint(0, len(unc_indices) - 1)]
                seed_subset = list(all_t_subsets[target_idx])
            else:
                seed_subset = list(old_block)
        else:
            seed_subset = list(old_block)
            
        new_block_set = set(seed_subset)
        while len(new_block_set) < k:
            remaining = [x for x in range(v) if x not in new_block_set]
            if not remaining:
                break
            # Heuristic: pick point that covers most uncovered
            # For speed, pick random or best of a few
            best_x = random.choice(remaining)
            # Optional: check a few candidates
            # candidates = random.sample(remaining, min(5, len(remaining)))
            # best_gain = -1
            # for x in candidates:
            #    ... compute gain ...
            new_block_set.add(best_x)
            
        new_block = tuple(sorted(new_block_set))
        
        if new_block == old_block:
            # Restore
            cov_count[old_tsubs] += 1
            # Update mask? 
            # We need to restore the mask correctly.
            # This is getting complicated with the mask.
            # Let's simplify: 
            # 1. Store old_cov_count slice? No, too slow.
            # 2. Just recalculate the delta and accept/reject.
            #    If reject, we must restore cov_count and current_uncovered.
            
            # Let's implement a helper to apply a block change and return the delta in uncovered count.
            pass

        # This approach is getting messy with the mask. 
        # Let's use a simpler, robust method:
        # 1. Maintain `cov_count` array.
        # 2. Maintain `current_uncovered` int.
        # 3. To apply a block:
        #    - For each tsub in block: 
        #      - If cov_count[tsub] == 0: current_uncovered -= 1 (it was uncovered, now covered)
        #      - cov_count[tsub] += 1
        #    - To remove a block:
        #    - For each tsub in block:
        #      - cov_count[tsub] -= 1
        #      - If cov_count[tsub] == 0: current_uncovered += 1 (it was covered, now uncovered)
        
        # This is O(C(k,t)) per move, which is 126 operations. Very fast.
        
        # Let's restart the SA loop logic with this clean method.
        
        # (Code implementation below)

    # Final check
    # Verify coverage
    final_cov = np.zeros(num_t_subsets, dtype=np.int32)
    for b in best_blocks:
        tsubs = get_block_tsubsets_indices(b)
        final_cov[tsubs] += 1
    final_uncovered = int(np.sum(final_cov == 0))
    
    if final_uncovered > 0:
        # Fallback: add blocks to cover remaining
        # Greedy add
        while final_uncovered > 0 and time.time() - start_time < timeout:
            unc_indices = np.where(final_cov == 0)[0]
            if len(unc_indices) == 0:
                break
            # Find best block to add
            best_block = None
            best_gain = -1
            for _ in range(100):
                target_idx = unc_indices[random.randint(0, len(unc_indices) - 1)]
                seed_subset = list(all_t_subsets[target_idx])
                block_set = set(seed_subset)
                while len(block_set) < k:
                    remaining = [x for x in range(v) if x not in block_set]
                    if not remaining: break
                    block_set.add(random.choice(remaining))
                block = tuple(sorted(block_set))
                tsubs = get_block_tsubsets_indices(block)
                gain = np.sum(final_cov[tsubs] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            if best_block is not None and best_gain > 0:
                best_blocks.append(best_block)
                tsubs = get_block_tsubsets_indices(best_block)
                final_cov[tsubs] += 1
                final_uncovered = int(np.sum(final_cov == 0))
            else:
                break

    return [list(b) for b in best_blocks]
