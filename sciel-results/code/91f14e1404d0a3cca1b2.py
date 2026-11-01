import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Precomputation
    # Generate all t-subsets and map them to indices
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {tset: i for i, tset in enumerate(all_tsets)}
    
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, find the indices of the t-subsets it covers
    block_tsets = []
    for b in all_blocks:
        tsets_in_b = []
        # Iterate over all t-subsets of b
        for ts in itertools.combinations(b, t):
            tsets_in_b.append(tset_to_idx[ts])
        block_tsets.append(np.array(tsets_in_b, dtype=np.int32))
        
    # Precompute tset_to_blocks
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for b_idx, b in enumerate(all_blocks):
        for ts in itertools.combinations(b, t):
            t_idx = tset_to_idx[ts]
            tset_to_blocks[t_idx].append(b_idx)
            
    # Precompute weights for weighted greedy
    # Weight is inverse of frequency, encouraging coverage of rare tsets
    tset_freq = np.zeros(num_tsets, dtype=np.int32)
    for b_idx in range(num_blocks):
        for ts_idx in block_tsets[b_idx]:
            tset_freq[ts_idx] += 1
            
    # Avoid division by zero (though freq should be > 0 for valid covering problems)
    tset_weights = np.ones(num_tsets, dtype=np.float64)
    nonzero_freq = tset_freq > 0
    tset_weights[nonzero_freq] = 1.0 / tset_freq[nonzero_freq]
    
    best_blocks = None
    best_len = float('inf')
    
    def greedy_cover_fast(seed_val, weights=None):
        if weights is None:
            weights = tset_weights
            
        random.seed(seed_val)
        uncovered = np.ones(num_tsets, dtype=bool)
        uncovered_set = set(range(num_tsets))
        blocks = []
        
        while uncovered_set:
            if time.time() > deadline:
                return None
            
            # Pick a random uncovered tset
            target_tset = random.choice(list(uncovered_set))
            candidate_blocks = tset_to_blocks[target_tset]
            
            best_block = -1
            best_gain = -1.0
            
            for b_idx in candidate_blocks:
                gain = 0.0
                for ts_idx in block_tsets[b_idx]:
                    if uncovered[ts_idx]:
                        gain += weights[ts_idx]
                        
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                elif gain == best_gain and random.random() < 0.5:
                    best_block = b_idx
                    
            if best_block == -1:
                break
                
            blocks.append(best_block)
            
            for ts_idx in block_tsets[best_block]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    uncovered_set.remove(ts_idx)
                    
        return blocks

    def local_search_remove(blocks):
        """
        Remove redundant blocks.
        """
        if not blocks:
            return blocks
            
        # Maintain coverage count
        cover_count = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in blocks:
            for ts_idx in block_tsets[b_idx]:
                cover_count[ts_idx] += 1
                
        if np.any(cover_count == 0):
            return None # Invalid
            
        blocks_list = blocks[:]
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
                
            # Try removing each block
            for i in range(len(blocks_list)):
                if time.time() > deadline:
                    break
                    
                b_idx = blocks_list[i]
                
                # Check if safe to remove
                safe = True
                for ts_idx in block_tsets[b_idx]:
                    if cover_count[ts_idx] == 1:
                        safe = False
                        break
                        
                if safe:
                    # Remove it
                    for ts_idx in block_tsets[b_idx]:
                        cover_count[ts_idx] -= 1
                    blocks_list.pop(i)
                    improved = True
                    # Restart loop to check for new redundancies
                    break
                    
        return blocks_list

    def local_search_repair(blocks, num_iterations=50):
        """
        Try to improve by removing a block and repairing.
        This allows escaping local minima of pure deletion.
        """
        if not blocks:
            return blocks
            
        blocks_list = blocks[:]
        current_len = len(blocks_list)
        
        for _ in range(num_iterations):
            if time.time() > deadline:
                break
                
            if not blocks_list:
                break
                
            # Pick a random block to remove
            i = random.randint(0, len(blocks_list) - 1)
            removed_block = blocks_list[i]
            
            # Identify tsets that become uncovered
            # We need to know which tsets are covered ONLY by this block?
            # No, we need to know which tsets are uncovered AFTER removal.
            # To do this efficiently, we should maintain cover counts.
            # Let's maintain cover counts in this function.
            
            # Re-calculate cover counts is O(N * 56). N is ~300. 300*56 = 16800. Cheap.
            cover_count = np.zeros(num_tsets, dtype=np.int32)
            for b_idx in blocks_list:
                for ts_idx in block_tsets[b_idx]:
                    cover_count[ts_idx] += 1
            
            # Temporarily remove
            for ts_idx in block_tsets[removed_block]:
                cover_count[ts_idx] -= 1
                
            # Find uncovered tsets
            uncovered_indices = np.nonzero(cover_count == 0)[0]
            
            if len(uncovered_indices) == 0:
                # It was redundant, just remove it (should have been caught by remove, but maybe due to order?)
                blocks_list.pop(i)
                current_len -= 1
                continue
                
            # Try to repair: Greedily add blocks to cover uncovered_indices
            # We need to add blocks that cover these tsets.
            # We can use a restricted greedy.
            
            new_blocks = []
            # We start with the current set minus the removed block
            current_set = set(blocks_list)
            current_set.remove(removed_block)
            
            # We need to cover the 'uncovered_indices'
            # We can reuse the greedy logic but only for these tsets?
            # Or just run a mini-greedy that picks blocks maximizing coverage of these specific tsets.
            
            # Let's use a simple greedy for the repair phase
            # We want to minimize the number of NEW blocks added.
            # Ideally, we want to add fewer than 1 block (impossible) or replace 1 with 1 (no gain) or 1 with 0 (gain 1).
            # Since we removed 1, if we add 1, size is same. If we add 0, size decreases.
            # If we add 2, size increases (bad).
            
            # So, we only accept if we can cover the gap with 0 or 1 blocks?
            # If 0 blocks: It was redundant. (Handled above)
            # If 1 block: Size stays same. We might accept this if it allows further reductions? 
            # For now, let's only accept if size decreases (0 blocks added) or if size stays same but we make progress?
            # Actually, swapping 1 for 1 doesn't change size. 
            # Let's try to find a single block that covers ALL uncovered_indices.
            
            # Find intersection of tset_to_blocks for all uncovered_indices
            if len(uncovered_indices) <= 10: # Limit search space
                candidate_blocks_set = set(tset_to_blocks[uncovered_indices[0]])
                for ts_idx in uncovered_indices[1:]:
                    candidate_blocks_set.intersection_update(tset_to_blocks[ts_idx])
                
                # Filter candidates: must not be in current_set
                valid_candidates = [b for b in candidate_blocks_set if b not in current_set]
                
                if valid_candidates:
                    # We can add 1 block to cover all gaps.
                    # This results in same size. 
                    # Is it useful? Maybe it changes the structure to allow future removals?
                    # Let's pick one randomly.
                    new_block = random.choice(valid_candidates)
                    
                    # Check if this swap is "better" in some heuristic way?
                    # For now, accept with some probability to allow random walk?
                    # Or just accept if it leads to a smaller solution after a subsequent removal pass?
                    # Let's just perform the swap and then run local_search_remove on the result.
                    
                    blocks_list.pop(i) # Remove the old one
                    blocks_list.append(new_block)
                    
                    # Run removal pass on the new list
                    improved_list = local_search_remove(blocks_list)
                    if improved_list and len(improved_list) < current_len:
                        blocks_list = improved_list
                        current_len = len(blocks_list)
                    # If not improved, we might want to revert? 
                    # But maybe the structure is better? 
                    # To keep it simple, only keep if size strictly decreases.
                    # If size is same, we might loop. 
                    # Let's only keep if size < current_len.
                    else:
                        # Revert
                        blocks_list.pop() # Remove new_block
                        blocks_list.append(removed_block)
                        
            # If we couldn't cover with 1 block, we would need >1, which increases size. 
            # We skip those cases for now to ensure monotonic improvement in size.
            
        return blocks_list

    # Run multiple attempts
    for attempt in range(30):
        if time.time() > deadline:
            break
            
        # 1. Greedy
        blocks = greedy_cover_fast(attempt)
        if blocks is None:
            continue
            
        # 2. Remove redundants
        blocks = local_search_remove(blocks)
        if blocks is None:
            continue
            
        # 3. Repair/Swap local search
        # Use a time limit for this phase
        blocks = local_search_repair(blocks, num_iterations=100)
        
        if blocks is None:
            continue
            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        # Fallback
        blocks = greedy_cover_fast(0)
        if blocks:
            return [list(all_blocks[b]) for b in blocks]
        return []
        
    return [list(all_blocks[b]) for b in best_blocks]
