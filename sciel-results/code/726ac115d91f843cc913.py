import itertools
import random
import time

def construct(v, k, t):
    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Precompute coverage masks for each k-subset
    # Using integers as bitsets is efficient for n_t ~ 4000
    block_masks = [0] * n_k
    for i, k_sub in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask

    # 2. Greedy Initialization
    uncovered_mask = (1 << n_t) - 1
    current_blocks = []
    current_block_indices = set()
    
    while uncovered_mask:
        best_idx = -1
        max_covered = 0
        # Optimize: We can skip blocks already in current_blocks if we assume no duplicates are useful, 
        # but greedy naturally picks distinct ones.
        for i in range(n_k):
            if i in current_block_indices:
                continue
            covered = (uncovered_mask & block_masks[i]).bit_count()
            if covered > max_covered:
                max_covered = covered
                best_idx = i
        
        if max_covered == 0:
            break
            
        current_blocks.append(best_idx)
        current_block_indices.add(best_idx)
        uncovered_mask &= ~block_masks[best_idx]
        
    # 3. Local Search: Try to reduce the number of blocks
    # We use a time limit to ensure we finish within 120s.
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    # Helper to check if a set of block indices covers all t-subsets
    # We can maintain a count of how many times each t-subset is covered.
    
    # Initialize coverage counts
    # coverage_count[j] = number of blocks in current_blocks covering t-subset j
    coverage_counts = [0] * n_t
    for idx in current_blocks:
        mask = block_masks[idx]
        # Iterate set bits in mask to increment counts
        # This is O(n_t) per block, which is fine for ~150 blocks
        m = mask
        while m:
            # Get index of least significant set bit
            lsb = m & -m
            idx_t = lsb.bit_length() - 1
            coverage_counts[idx_t] += 1
            m ^= lsb
            
    def try_remove(block_idx):
        """
        Attempts to remove block_idx from current_blocks.
        Returns True if successful.
        """
        nonlocal current_blocks, current_block_indices, coverage_counts
        
        mask = block_masks[block_idx]
        # Check if any t-subset covered by this block has count == 1
        m = mask
        while m:
            lsb = m & -m
            idx_t = lsb.bit_length() - 1
            if coverage_counts[idx_t] == 1:
                return False
            m ^= lsb
            
        # Safe to remove
        # Update coverage counts
        m = mask
        while m:
            lsb = m & -m
            idx_t = lsb.bit_length() - 1
            coverage_counts[idx_t] -= 1
            m ^= lsb
            
        current_blocks.remove(block_idx)
        current_block_indices.discard(block_idx)
        return True

    def try_swap(remove_idx, add_idx):
        """
        Attempts to swap remove_idx with add_idx.
        Returns True if successful.
        """
        nonlocal current_blocks, current_block_indices, coverage_counts
        
        if add_idx in current_block_indices:
            return False
            
        mask_remove = block_masks[remove_idx]
        mask_add = block_masks[add_idx]
        
        # We need to ensure that any t-subset covered ONLY by remove_idx (count 1)
        # is also covered by add_idx.
        # And that the new set is valid.
        
        # Efficient check:
        # 1. Decrement counts for remove_idx
        # 2. Check if any count becomes 0 for a t-subset NOT covered by add_idx
        # 3. If valid, increment counts for add_idx and update state
        
        # To avoid modifying global state if it fails, we can simulate.
        # But n_t is small (3876). We can just check.
        
        # Identify t-subsets covered by remove_idx but not add_idx
        # These must have count > 1 in the current set (so they remain covered after removal)
        
        # Actually, simpler:
        # A swap is valid if:
        # For all t in remove_idx: count[t] > 1 OR t in add_idx
        
        # Let's check this condition.
        m = mask_remove
        while m:
            lsb = m & -m
            idx_t = lsb.bit_length() - 1
            if coverage_counts[idx_t] == 1:
                # This t-subset is uniquely covered by remove_idx.
                # It MUST be covered by add_idx.
                if not (mask_add & lsb):
                    return False
            m ^= lsb
            
        # If we get here, the swap is valid.
        # Perform the swap:
        
        # Remove remove_idx
        m = mask_remove
        while m:
            lsb = m & -m
            idx_t = lsb.bit_length() - 1
            coverage_counts[idx_t] -= 1
            m ^= lsb
            
        # Add add_idx
        m = mask_add
        while m:
            lsb = m & -m
            idx_t = lsb.bit_length() - 1
            coverage_counts[idx_t] += 1
            m ^= lsb
            
        current_blocks.remove(remove_idx)
        current_block_indices.discard(remove_idx)
        current_blocks.append(add_idx)
        current_block_indices.add(add_idx)
        
        return True

    # Local Search Loop
    # We will try random deletions and swaps.
    # Since deletion reduces size, we prioritize it.
    # Swaps maintain size but might enable future deletions.
    
    # To make it faster, we can just do random restarts of greedy with different random seeds?
    # No, local search on the greedy result is better.
    
    # We'll do a simple hill climbing with random moves.
    # Move 1: Try to delete a random block.
    # Move 2: Try to swap a random block with a random other block.
    
    # To avoid O(N^2) swap checks, we can just try random pairs.
    
    while time.time() - start_time < time_limit:
        progress = False
        
        # 1. Try Deletions
        # Shuffle current blocks to try different ones
        # Instead of shuffling the list, we pick a random index in the list
        if current_blocks:
            rand_idx_in_list = random.randint(0, len(current_blocks) - 1)
            block_idx = current_blocks[rand_idx_in_list]
            if try_remove(block_idx):
                progress = True
                continue # Restart loop as state changed significantly
        
        # 2. Try Swaps
        # Pick a random block to remove
        if current_blocks:
            rand_idx_in_list = random.randint(0, len(current_blocks) - 1)
            remove_idx = current_blocks[rand_idx_in_list]
            
            # Pick a random block to add (from all k_subsets, not just current)
            add_idx = random.randint(0, n_k - 1)
            
            if try_swap(remove_idx, add_idx):
                progress = True
                continue # Restart loop as state changed
        
        # If no progress, we might be stuck in a local minimum.
        # We could implement Simulated Annealing here, but for now,
        # just continuing with random choices might escape slowly.
        # To be more effective, we can occasionally accept a "worse" swap?
        # But our swaps are only accepted if valid. We don't have a "worse" valid swap concept 
        # other than same size. 
        # Actually, swapping one block for another of the same size doesn't change the objective function (minimize blocks).
        # So why swap? To change the structure to allow a deletion later.
        
        # If we are stuck, we can try a "kicker": 
        # Remove a block (making it invalid), then add the best block to fix it.
        # This is more complex. Let's stick to valid swaps.
        
        # If no progress in a while, we might break? 
        # The time limit handles it.
        
    # Convert indices back to lists
    return [list(k_subsets[i]) for i in current_blocks]
