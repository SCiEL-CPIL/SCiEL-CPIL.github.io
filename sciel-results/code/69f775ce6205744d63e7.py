import itertools
import time
import random
import sys
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute masks and subset indices for each block
    block_masks = []
    block_subset_indices = []
    
    for block in all_blocks:
        mask = 0
        subs_idx = []
        for sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[sub]
            mask |= (1 << idx)
            subs_idx.append(idx)
        block_masks.append(mask)
        block_subset_indices.append(subs_idx)
        
    def greedy_repair(uncovered_mask, existing_blocks=None):
        if existing_blocks is None:
            existing_blocks = []
        else:
            existing_blocks = list(existing_blocks)
            
        if not uncovered_mask:
            return existing_blocks

        # Use a heap to speed up finding the best block
        # Heap elements: (-gain, block_idx)
        heap = []
        # Initialize heap with all blocks
        for i in range(num_blocks):
            gain = (uncovered_mask & block_masks[i]).bit_count()
            if gain > 0:
                heapq.heappush(heap, (-gain, i))
        
        while uncovered_mask and heap:
            # Pop the best block
            neg_gain, block_idx = heapq.heappop(heap)
            gain = -neg_gain
            
            # Check if this gain is still valid
            # Since we only remove coverage, the gain for this block might have decreased.
            # We need to re-calculate gain to verify.
            actual_gain = (uncovered_mask & block_masks[block_idx]).bit_count()
            
            if actual_gain != gain:
                # Stale entry, push updated entry if still useful
                if actual_gain > 0:
                    heapq.heappush(heap, (-actual_gain, block_idx))
                continue
            
            if actual_gain == 0:
                continue
                
            # Select this block
            existing_blocks.append(block_idx)
            uncovered_mask &= ~block_masks[block_idx]
            
            # Update heap for blocks that might have changed gain
            # This is expensive if we do it for all blocks.
            # Alternative: Lazy evaluation.
            # When we pop, we check if the gain is still correct.
            # If not, we re-push with the correct gain.
            # But how do we know which blocks to re-push?
            # Actually, the standard lazy heap approach for set cover is:
            # 1. Push all blocks.
            # 2. Pop top. Check if gain is still max? No, just check if it's still valid (gain > 0).
            #    Wait, the gain *decreases* as we add blocks.
            #    So if we pop a block with gain G, and its actual gain is G' < G,
            #    it means some of the subsets it covered are now covered by other blocks.
            #    We re-push it with G'.
            #    But we also need to make sure we don't miss a block that became the new best.
            #    However, since gains only decrease, the true best block must be in the heap
            #    with its *current* gain or a *higher* stale gain.
            #    If it has a higher stale gain, it will be popped sooner, checked, and re-pushed
            #    with the lower gain. Eventually, the true best will be on top.
            
            # But we need to ensure we don't loop forever or miss updates.
            # The lazy approach works if we just re-push stale entries.
            
            # Optimization: Instead of re-pushing immediately, we can just let it sit.
            # When popped, if stale, re-push.
            
            # Wait, the code above does exactly that.
            # But there is a subtle issue: if we pop a stale entry, we re-push it.
            # This is correct.
            
            # However, we don't need to re-push if gain is 0.
            
            # One more optimization: If we have a large number of blocks, initializing the heap
            # is O(N log N). This is fine.
            
            # Let's stick to the lazy heap approach.
            
        return existing_blocks

    def reduce_blocks(block_indices):
        if not block_indices:
            return []
            
        cover_count = [0] * num_t_subsets
        
        for idx in block_indices:
            for s_idx in block_subset_indices[idx]:
                cover_count[s_idx] += 1
                
        new_blocks = []
        for idx in block_indices:
            subs = block_subset_indices[idx]
            for s_idx in subs:
                cover_count[s_idx] -= 1
                
            is_redundant = True
            for s_idx in subs:
                if cover_count[s_idx] == 0:
                    is_redundant = False
                    break
                    
            if is_redundant:
                pass
            else:
                for s_idx in subs:
                    cover_count[s_idx] += 1
                new_blocks.append(idx)
                
        return new_blocks

    full_mask = (1 << num_t_subsets) - 1
    initial_blocks = greedy_repair(full_mask)
    current_blocks = reduce_blocks(initial_blocks)
    best_blocks = list(current_blocks)
    best_size = len(best_blocks)
    
    while time.time() < deadline:
        if not current_blocks:
            break
            
        perturbed_blocks = list(current_blocks)
        remove_idx = random.randint(0, len(perturbed_blocks) - 1)
        perturbed_blocks.pop(remove_idx)
        
        covered_mask = 0
        for idx in perturbed_blocks:
            covered_mask |= block_masks[idx]
            
        uncovered_mask = full_mask & ~covered_mask
        
        repaired_blocks = greedy_repair(uncovered_mask, perturbed_blocks)
        new_blocks = reduce_blocks(repaired_blocks)
        
        new_size = len(new_blocks)
        
        if new_size < best_size:
            best_blocks = new_blocks
            best_size = new_size
            current_blocks = new_blocks
        else:
            if new_size == best_size:
                if random.random() < 0.5:
                    current_blocks = new_blocks
            else:
                if random.random() < 0.2:
                    current_blocks = new_blocks
                else:
                    current_blocks = best_blocks

    final_blocks = [list(all_blocks[idx]) for idx in best_blocks]
    return final_blocks
