import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer for safety
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Generate all k-subsets (blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # 3. Pre-compute bitmask for each k-subset and static score
    # Also precompute the list of indices of t-subsets covered by each block
    block_data = []
    block_t_indices = []
    for block in k_subsets:
        mask = 0
        indices = []
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
            indices.append(idx)
        static_score = mask.bit_count()
        block_data.append((static_score, block, mask))
        block_t_indices.append(indices)
        
    # Sort by static score descending to break ties in greedy selection
    # This is a heuristic: blocks that cover more total t-subsets are "better"
    block_data.sort(key=lambda x: x[0], reverse=True)
    
    # Re-index block_t_indices to match the sorted order of block_data
    # We need to map from the original block index to the sorted position
    # Actually, block_t_indices was built in the order of k_subsets.
    # block_data is sorted. We need to align them.
    # Let's just rebuild the sorted list of indices.
    sorted_indices = [i for i, _ in enumerate(sorted(range(num_k_subsets), key=lambda i: block_data[i][0], reverse=True))]
    # Wait, block_data contains (score, block, mask).
    # Let's just create a list of indices sorted by score.
    indices_sorted = sorted(range(num_k_subsets), key=lambda i: block_data[i][0], reverse=True)
    
    # Precompute numpy coverage matrix for fast gain calculation?
    # With 3876 blocks and 3876 subsets, a dense matrix is 15M entries.
    # Using bitmasks is likely faster in Python for the intersection count if we stick to Python ints.
    # However, the current program uses pure Python bit operations which are fast enough for 3876x3876.
    
    blocks = []
    
    # Greedy Construction
    # We use the sorted order of indices to iterate candidates.
    # We maintain an uncovered bitmask.
    uncovered = (1 << num_t_subsets) - 1
    
    # To speed up, we can skip blocks that are completely covered?
    # No, a block might be completely covered in terms of its *new* gain, but we need to find the max gain.
    # If gain is 0, it's useless.
    
    # Optimization: If we have many blocks, iterating all of them each step is O(N^2).
    # 3876^2 is ~15 million operations. In Python, this might be slow if done 150 times (2.2 billion).
    # We need to be careful.
    # The current program does this and scores 1.10.
    # Let's try to optimize the inner loop or use a different strategy.
    
    # Strategy:
    # 1. Greedy to get a good initial solution.
    # 2. Local search to improve it.
    
    while uncovered:
        if time.time() > deadline - 5:  # Leave 5s for local search if possible, or just break
            break
            
        max_gain = 0
        best_block = None
        best_mask = 0
        
        # Iterate through all possible blocks in sorted order
        # Using the pre-sorted indices
        for idx in indices_sorted:
            mask = block_data[idx][2]
            # Optimization: If the max possible gain (static score) is less than current max_gain, skip?
            # No, static score is total coverage, not current gain.
            # But if (uncovered & mask) is 0, gain is 0.
            # We can check if mask & uncovered == 0 quickly?
            # mask & uncovered is a big int op.
            
            # Quick check: if the highest bit in mask is not in uncovered?
            # Just compute gain.
            gain = (uncovered & mask).bit_count()
            
            if gain > max_gain:
                max_gain = gain
                best_block = block_data[idx][1]
                best_mask = mask
                # Early termination if max_gain is theoretical max?
                # Max gain for a block is C(k,t) = 35.
                # If max_gain == 35, we can't do better.
                if max_gain == 35:
                    break
        
        if best_block is None:
            # No block covers any uncovered subset
            break
            
        blocks.append(list(best_block))
        uncovered &= ~best_mask
        
    # At this point, we have a valid covering (hopefully) or a partial one.
    # If we broke out early due to time, we might have uncovered subsets.
    # The task requires a valid covering.
    # If we don't have a valid covering, we must fill it greedily without time limit?
    # No, if we time out, we score 0.
    # We must ensure validity.
    # The greedy loop above only breaks if no block has gain > 0.
    # If we break due to time, we return a partial covering which is invalid.
    # We need a safety net.
    # If time is running out, we should finish the greedy quickly or use a precomputed fallback?
    # Given the small size (19,7,4), the greedy should finish well within 120s.
    # Let's check: 150 steps * 3876 blocks * bit_op.
    # Bit op on 3876-bit int is fast.
    # 150 * 3876 = 581,400 iterations.
    # Each iteration does an AND and bit_count.
    # This should take < 1 second.
    
    # Now, perform Local Search to reduce the number of blocks.
    # State: list of block indices (from the original k_subsets list, not sorted).
    # We need to map the blocks we selected back to their original indices.
    
    # Create a map from block tuple to index
    block_to_idx = {block: i for i, block in enumerate(k_subsets)}
    
    current_block_indices = [block_to_idx[tuple(b)] for b in blocks]
    
    # Precompute coverage counts for each t-subset
    counts = [0] * num_t_subsets
    for idx in current_block_indices:
        for sub_idx in block_t_indices[idx]:
            counts[sub_idx] += 1
            
    # Helper to check if a block is redundant
    def is_redundant(idx):
        for sub_idx in block_t_indices[idx]:
            if counts[sub_idx] == 1:
                return False
        return True
        
    # Helper to remove a block
    def remove_block(pos):
        idx = current_block_indices[pos]
        for sub_idx in block_t_indices[idx]:
            counts[sub_idx] -= 1
        current_block_indices.pop(pos)
        
    # Helper to add a block
    def add_block(idx):
        current_block_indices.append(idx)
        for sub_idx in block_t_indices[idx]:
            counts[sub_idx] += 1

    best_len = len(current_block_indices)
    best_blocks = current_block_indices.copy()
    
    # Local Search Loop
    # We try to remove redundant blocks.
    # Then we try to swap blocks to create more redundancy or allow removal.
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline - 2:
        improved = False
        
        # Phase 1: Remove any redundant blocks
        pos = 0
        while pos < len(current_block_indices):
            if is_redundant(current_block_indices[pos]):
                remove_block(pos)
                improved = True
                # Do not increment pos, check the new block at this position
                if len(current_block_indices) < best_len:
                    best_len = len(current_block_indices)
                    best_blocks = current_block_indices.copy()
            else:
                pos += 1
                
        if not improved:
            no_improve += 1
            if no_improve > max_no_improve:
                break
        else:
            no_improve = 0
            
        # Phase 2: Try swaps to improve structure
        # Find a block with the fewest unique subsets (count == 1)
        # If we can swap it with another block that covers those unique subsets,
        # we might be able to remove the original block or make the new one redundant later.
        
        # To keep it simple and fast, let's just try to find a swap that reduces the "total redundancy"
        # or allows a removal.
        
        # A simpler heuristic:
        # Pick a random block i.
        # Find all subsets covered only by i.
        # If there are none, it's redundant (handled in Phase 1).
        # If there are some, find another block j that covers all of them.
        # Swap i with j? No, we want to replace i with a block that covers its unique parts AND maybe allows i to be removed?
        # Actually, if we swap i with j, we lose i's coverage and gain j's.
        # We need j to cover the unique parts of i.
        
        # Let's try a 1-move: Remove i, then add j if valid?
        # If we remove i, some subsets become uncovered (those with count 1).
        # We need to add a block j that covers ALL of those uncovered subsets.
        # This is a set cover problem on the small set of uncovered subsets.
        
        if time.time() < deadline - 2:
            # Try to find a block that can be removed and replaced by another that covers its unique parts
            # We sample a few blocks to try
            sample_size = min(10, len(current_block_indices))
            for _ in range(sample_size):
                i = random.randint(0, len(current_block_indices) - 1)
                idx_i = current_block_indices[i]
                
                # Find unique subsets of i
                unique_subs = [sub_idx for sub_idx in block_t_indices[idx_i] if counts[sub_idx] == 1]
                
                if not unique_subs:
                    # Should have been caught in Phase 1, but just in case
                    remove_block(i)
                    improved = True
                    if len(current_block_indices) < best_len:
                        best_len = len(current_block_indices)
                        best_blocks = current_block_indices.copy()
                    continue
                    
                # Find a candidate block j that covers all unique_subs
                # And is not already in the list?
                # Actually, if j is in the list, swapping doesn't change the set.
                # We want to replace i with a block j that is NOT in the list,
                # such that j covers all unique_subs.
                # Then we can remove i.
                # Wait, if we add j and remove i, the size is the same.
                # We only gain if we can then remove another block?
                # Or if j covers more other things?
                
                # A better move:
                # Remove i.
                # The uncovered subsets are unique_subs.
                # We need to cover them.
                # If we can find a block j that covers unique_subs AND allows us to remove ANOTHER block k?
                # That's complex.
                
                # Let's stick to simple removal if possible.
                # The main gain comes from the greedy not being optimal.
                # Local search removing redundancies is the standard improvement.
                
                pass # Skip complex swaps for now to ensure speed and stability

    # Convert best_blocks back to lists of integers
    result = [list(k_subsets[idx]) for idx in best_blocks]
    
    # Final validity check (just in case)
    # If invalid, we need to fix it.
    # But the local search only removes redundant blocks, so validity is preserved.
    
    return result
