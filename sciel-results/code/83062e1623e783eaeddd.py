import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    # 1. Precomputation
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {ts: i for i, ts in enumerate(all_t_subs)}
    num_t_subs = len(all_t_subs)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute t-indices for each block
    # C(k, t) is the number of t-subsets in a block
    num_t_in_block = len(list(itertools.combinations(range(k), t)))
    block_t_indices = np.zeros((num_blocks, num_t_in_block), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, ts in enumerate(itertools.combinations(block, t)):
             block_t_indices[i, j] = t_sub_to_idx[ts]
             
    # Helper to get t-indices for a block index
    def get_block_t_indices(block_idx):
        return block_t_indices[block_idx]

    # Helper to map block tuple to index
    block_tuple_to_idx = {b: i for i, b in enumerate(all_blocks)}

    # State:
    # blocks_used: list of block indices
    # t_sub_count: np.array of counts for each t-subset
    # used_mask: np.array bool for blocks

    def greedy_construct():
        uncovered = np.ones(num_t_subs, dtype=np.int8)
        blocks_used = []
        used_mask = np.zeros(num_blocks, dtype=bool)
        
        while uncovered.sum() > 0:
            # Calculate gains
            # gains[i] = sum(uncovered[indices])
            # We only care about unused blocks
            gains = np.zeros(num_blocks, dtype=np.int32)
            # This loop is slow if done in Python. 
            # Vectorized:
            # We can compute gains for all blocks at once?
            # gains = uncovered[block_t_indices].sum(axis=1)
            # This creates a large temp array. 
            # num_blocks * num_t_in_block = 75582 * 56 ~ 4.2M elements. 
            # This is fast in numpy.
            gains = uncovered[block_t_indices].sum(axis=1)
            gains[used_mask] = -1
            
            best_idx = np.argmax(gains)
            if gains[best_idx] <= 0:
                break # Should not happen if covering is possible
            
            blocks_used.append(best_idx)
            used_mask[best_idx] = True
            uncovered[block_t_indices[best_idx]] = 0
            
        # Build counts
        t_sub_count = np.zeros(num_t_subs, dtype=np.int32)
        for b_idx in blocks_used:
            t_sub_count[block_t_indices[b_idx]] += 1
            
        return blocks_used, t_sub_count

    def local_search(blocks_used, t_sub_count):
        # Try to remove blocks
        # We need to check if a block is redundant.
        # A block is redundant if all its t-subsets have count > 1.
        changed = True
        while changed:
            changed = False
            # Iterate through blocks
            # To speed up, we can shuffle the order or just iterate.
            # We need to find ANY redundant block.
            # Checking all is O(N * 56).
            for i in range(len(blocks_used) - 1, -1, -1):
                b_idx = blocks_used[i]
                idxs = block_t_indices[b_idx]
                # If min count > 1, it's safe to remove
                if np.min(t_sub_count[idxs]) > 1:
                    t_sub_count[idxs] -= 1
                    blocks_used.pop(i)
                    changed = True
                    break # Restart to handle cascading effects
        return blocks_used, t_sub_count

    def perturb_and_repair(blocks_used, t_sub_count, num_perturb):
        # Deep copy state
        new_blocks = blocks_used.copy()
        new_counts = t_sub_count.copy()
        
        # Remove random blocks
        # Ensure we don't remove more than available
        num_to_remove = min(num_perturb, len(new_blocks))
        if num_to_remove == 0:
            return new_blocks, new_counts
            
        # Randomly select indices to remove
        # We remove from the list, so we need to be careful with indices.
        # It's easier to select block indices to remove.
        remove_indices = random.sample(range(len(new_blocks)), num_to_remove)
        # Sort descending to remove from end first to keep indices valid for earlier removals?
        # Or just remove one by one.
        for idx in sorted(remove_indices, reverse=True):
            b_idx = new_blocks[idx]
            new_counts[block_t_indices[b_idx]] -= 1
            new_blocks.pop(idx)
            
        # Repair: Add blocks to cover all t-subsets
        # Identify uncovered
        # We need to know which t-subsets are uncovered (count == 0)
        # We don't have a direct "uncovered" mask, but we can derive it.
        # However, for efficiency, let's maintain an uncovered mask or just check counts.
        # In repair, we add blocks greedily based on gain (number of uncovered t-subsets covered).
        
        # We need to track which blocks are currently used to avoid duplicates.
        used_mask = np.zeros(num_blocks, dtype=bool)
        for b_idx in new_blocks:
            used_mask[b_idx] = True
            
        # Loop until all covered
        while np.any(new_counts == 0):
            # Find uncovered t-subsets
            # Gains for each unused block
            # Gain = number of t-subsets in block that are currently uncovered (count==0)
            # We can compute this using new_counts.
            # mask_uncovered = (new_counts == 0)
            # gains = mask_uncovered[block_t_indices].sum(axis=1)
            
            mask_uncovered = (new_counts == 0)
            gains = mask_uncovered[block_t_indices].sum(axis=1)
            gains[used_mask] = -1
            
            best_idx = np.argmax(gains)
            if gains[best_idx] <= 0:
                # Should not happen if covering is possible, but if it does, break to avoid infinite loop
                break
                
            new_blocks.append(best_idx)
            used_mask[best_idx] = True
            new_counts[block_t_indices[best_idx]] += 1
            
        return new_blocks, new_counts

    # Main Logic
    start_time = time.time()
    time_limit = 115.0
    
    # Initial Solution
    blocks_used, t_sub_count = greedy_construct()
    blocks_used, t_sub_count = local_search(blocks_used, t_sub_count)
    
    best_blocks = blocks_used.copy()
    best_count = len(best_blocks)
    
    current_blocks = blocks_used
    current_counts = t_sub_count
    current_count = len(current_blocks)
    
    # Iterated Local Search
    while time.time() - start_time < time_limit:
        # Perturb
        # Number of blocks to remove: random between 5 and 20?
        # Tuning parameter.
        num_perturb = random.randint(5, 20)
        perturbed_blocks, perturbed_counts = perturb_and_repair(current_blocks, current_counts, num_perturb)
        
        # Local Search
        perturbed_blocks, perturbed_counts = local_search(perturbed_blocks, perturbed_counts)
        
        new_count = len(perturbed_blocks)
        
        # Acceptance Criteria
        # If better, always accept.
        # If worse, accept with probability exp(-delta / T)? 
        # For simplicity and speed, we can use a simple hill-climbing with random restarts if stuck, 
        # or standard ILS acceptance.
        # Let's use: Accept if new_count <= current_count (allows neutral moves to explore).
        # Or accept worse moves with low probability to escape local minima.
        
        if new_count <= current_count:
            current_blocks = perturbed_blocks
            current_counts = perturbed_counts
            current_count = new_count
            
            if new_count < best_count:
                best_count = new_count
                best_blocks = current_blocks.copy()
        else:
            # Accept worse move with some probability to escape basin?
            # Delta = new_count - current_count
            # Prob = 0.1 (fixed) or based on delta.
            # Given time constraint, simple acceptance of equal/better is safer and faster.
            # Let's try accepting worse moves with prob 0.05.
            if random.random() < 0.05:
                current_blocks = perturbed_blocks
                current_counts = perturbed_counts
                current_count = new_count

    # Convert indices back to lists of integers
    result_blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    
    return result_blocks
