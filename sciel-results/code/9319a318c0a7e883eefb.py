import itertools
import time
import random
import sys

def construct(v, k, t):
    # Parameters
    # v=19, k=7, t=4 expected
    
    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Precompute masks and t-subset indices for each block
    # block_masks[i]: int bitmask
    # block_t_indices[i]: list of int indices of t-subsets contained in block i
    block_masks = [0] * n_k
    block_t_indices = [[] for _ in range(n_k)]
    
    for i, k_sub in enumerate(k_subsets):
        mask = 0
        indices = []
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_t_indices[i] = indices

    # 2. Greedy Initialization
    # We want a valid cover to start with.
    # Greedy: pick block covering most uncovered.
    uncovered_mask = (1 << n_t) - 1
    current_blocks = []
    current_block_set = set()
    coverage_counts = [0] * n_t
    
    # Fast greedy initialization
    while uncovered_mask:
        best_block_idx = -1
        max_covered = 0
        
        for i in range(n_k):
            if i in current_block_set:
                continue
            covered = (uncovered_mask & block_masks[i]).bit_count()
            if covered > max_covered:
                max_covered = covered
                best_block_idx = i
                
        if max_covered == 0:
            break
            
        # Add best block
        current_blocks.append(best_block_idx)
        current_block_set.add(best_block_idx)
        
        # Update coverage counts
        for idx in block_t_indices[best_block_idx]:
            coverage_counts[idx] += 1
            
        uncovered_mask &= ~block_masks[best_block_idx]
        
    # 3. Local Search (Simulated Annealing / ILS)
    # We will try to reduce the number of blocks.
    # State: current_blocks, coverage_counts, uncovered_mask (should be 0)
    
    start_time = time.time()
    time_limit = 110 # Leave 10s margin
    
    # Helper to check if a block can be removed
    def can_remove(block_idx):
        for t_idx in block_t_indices[block_idx]:
            if coverage_counts[t_idx] == 1:
                return False
        return True

    # Helper to remove a block
    def remove_block(block_idx):
        current_blocks.remove(block_idx)
        current_block_set.remove(block_idx)
        for t_idx in block_t_indices[block_idx]:
            coverage_counts[t_idx] -= 1
        # Update uncovered_mask? Not strictly needed if we track counts, but good for consistency.
        # Actually, we only need to know if uncovered_mask is 0.
        # If we remove a block and all counts > 1, uncovered_mask remains 0.

    # Helper to add a block
    def add_block(block_idx):
        current_blocks.append(block_idx)
        current_block_set.add(block_idx)
        for t_idx in block_t_indices[block_idx]:
            coverage_counts[t_idx] += 1
            if coverage_counts[t_idx] == 1:
                # This t-subset was uncovered, now covered
                # We don't explicitly track uncovered_mask in the loop for speed, 
                # but we can check validity by ensuring no count is 0.
                pass

    # We need to track if the cover is valid.
    # Valid if min(coverage_counts) > 0.
    # Since we start valid and only do valid moves (remove if min>1, swap if result valid),
    # we can assume validity if we check carefully.
    
    # To check validity of a swap (remove b1, add b2):
    # 1. Tentatively decrement counts for b1.
    # 2. Check if any count becomes 0. If so, swap invalid.
    # 3. If valid, increment counts for b2.
    
    def try_swap(b1, b2):
        # b1 in current, b2 not in current
        # Check if removing b1 breaks coverage
        # We need to see if any t in b1 has count 1.
        # If yes, removing it makes count 0 -> invalid.
        # Wait, if we swap, we add b2. So the final count for t is (old_count - 1_{t in b1} + 1_{t in b2}).
        # We need final_count >= 1 for all t.
        
        # Efficient check:
        # For t in b1:
        #   if coverage_counts[t] == 1 and t not in b2: return False
        # For t in b2:
        #   if coverage_counts[t] == 0 and t not in b1: return False (but we started valid, so counts >= 1)
        # Actually, since we start valid, counts >= 1.
        # Removing b1 makes counts[t] -= 1 for t in b1.
        # If any becomes 0, it must be covered by b2.
        
        b1_t_set = set(block_t_indices[b1])
        b2_t_set = set(block_t_indices[b2])
        
        # Check removal of b1
        for t_idx in block_t_indices[b1]:
            if coverage_counts[t_idx] == 1:
                if t_idx not in b2_t_set:
                    return False
                # else, b2 covers it, so it's ok after swap
        
        # If we pass this, the swap is valid.
        # Perform swap
        remove_block(b1)
        add_block(b2)
        return True

    # Main Loop
    while time.time() - start_time < time_limit:
        # Phase 1: Try to remove blocks
        removed_any = False
        # Shuffle to avoid bias? Or just iterate.
        # We can iterate through a random permutation of current blocks
        indices_to_try = current_blocks[:]
        random.shuffle(indices_to_try)
        
        for b_idx in indices_to_try:
            if b_idx not in current_block_set:
                continue
            if can_remove(b_idx):
                remove_block(b_idx)
                removed_any = True
        
        if not removed_any:
            # Phase 2: Try swaps to escape local minimum
            # Try a few random swaps
            for _ in range(10): # Limit swaps per stuck phase
                if not current_blocks:
                    break
                b1 = random.choice(current_blocks)
                # Pick random b2 not in current
                b2 = random.randrange(n_k)
                if b2 in current_block_set:
                    continue
                if try_swap(b1, b2):
                    # Swap accepted, break to try removals again
                    break
                    
    # Convert indices back to lists
    return [list(k_subsets[i]) for i in current_blocks]
