"""Covering design C(19,8,5) with vectorized bitset greedy and randomized restarts."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

DEADLINE_SECONDS = 118.0

def construct(v, k, t):
    start_time = time.time()
    
    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute bitset for each block
    # bit i is set if the i-th t-subset is contained in the block
    block_bitsets = [0] * n_blocks
    for bi, block in enumerate(all_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
        block_bitsets[bi] = mask
    
    # Function to perform one greedy run with a specific random seed for tie-breaking
    def run_greedy(seed):
        random.seed(seed)
        uncovered = (1 << n_t) - 1
        blocks_used = []
        
        # Optimization: If we just iterate all blocks, it's O(N_blocks * N_steps).
        # N_blocks ~ 75k, N_steps ~ 300. 22.5M iterations.
        # We can speed up by maintaining a list of blocks and their current gains?
        # But gains change as uncovered changes.
        # Simple iteration is robust and likely fast enough with Python ints.
        
        while uncovered:
            if time.time() - start_time > DEADLINE_SECONDS - 5:
                return None # Timeout
            
            best_gain = -1
            best_blocks = []
            
            # Iterate through all blocks
            # To add randomness in ties, we can shuffle the block order or just pick randomly from best.
            # Shuffling 75k items every step is expensive.
            # Instead, we can just find the max gain, and collect all blocks with that gain.
            # If the list is large, pick one randomly.
            
            # Optimization: We can skip blocks that are subsets of already used blocks?
            # No, they might cover different things if we don't assume nested structure?
            # Actually, if block A is subset of block B, and B is used, A is redundant?
            # Not necessarily, A might cover t-subsets that B doesn't? No, if A subset B, then any t-subset in A is in B.
            # So if B is used, A is redundant.
            # We can maintain a set of used block tuples and skip any block that is a subset of a used block?
            # Checking subset is expensive.
            
            # Let's stick to the simple loop first.
            for bi in range(n_blocks):
                # Optional: Skip if this block is already used? 
                # We don't need to use the same block twice.
                # But we don't have a fast "is used" check for bi.
                # We can maintain a set of used block indices.
                
                mask = block_bitsets[bi]
                # Calculate gain
                common = mask & uncovered
                # Popcount
                # Using bin().count('1') is fast enough?
                # Let's try.
                gain = common.bit_count() # Python 3.10+ has int.bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_blocks = [bi]
                elif gain == best_gain and gain > 0:
                    best_blocks.append(bi)
            
            if not best_blocks:
                break # No block covers any uncovered subset? Should not happen if uncovered is not 0 and we have all blocks.
            
            # Pick a block randomly from the best candidates
            chosen_bi = random.choice(best_blocks)
            blocks_used.append(all_blocks[chosen_bi])
            uncovered &= ~block_bitsets[chosen_bi]
            
        return blocks_used

    # Local improvement: remove redundant blocks
    def improve(blocks):
        if not blocks:
            return blocks
        
        # Convert to set of tuples for fast lookup?
        # We want to remove any block that is redundant.
        # A block B is redundant if every t-subset in B is covered by other blocks.
        
        # To do this efficiently, we can try removing each block and checking if coverage is still complete.
        # But checking coverage is expensive.
        # Alternative: 
        # 1. Build a coverage count for each t-subset.
        # 2. If a t-subset has count 1, the block covering it is essential.
        # 3. If all t-subsets have count >= 2, we can try removing blocks.
        
        # Let's use the bitset approach for verification.
        # We'll try to remove blocks one by one.
        
        # To speed up, we can just iterate and try to remove.
        # If we remove a block, we must verify the rest covers everything.
        # Verification: OR all remaining bitsets and check if it equals full_mask.
        
        full_mask = (1 << n_t) - 1
        current_mask = 0
        for b in blocks:
            # Find index of b in all_blocks?
            # This is O(N) lookup.
            # Better: Precompute a map from block tuple to index.
            pass # This local improvement might be slow if done naively.
            
        # Let's use a simpler heuristic for improvement:
        # Sort blocks by "essentiality" or just try random removals?
        # Given the time limit, the greedy with bitsets is the main driver.
        # Let's implement a fast redundancy check.
        
        # Map block tuple to its bitset index
        block_to_idx = {b: i for i, b in enumerate(all_blocks)}
        
        improved = True
        while improved:
            improved = False
            for i in range(len(blocks)):
                if time.time() - start_time > DEADLINE_SECONDS - 2:
                    return blocks
                
                # Try removing blocks[i]
                # Calculate union of all other blocks
                # This is O(N_blocks) per removal attempt.
                # If we have 300 blocks, and we try removing each, that's 300 * 300 * (cost of OR)
                # Cost of OR is O(1) for ints.
                # So 90,000 OR operations. Very fast.
                
                union_mask = 0
                for j in range(len(blocks)):
                    if j == i:
                        continue
                    b = blocks[j]
                    idx = block_to_idx[b]
                    union_mask |= block_bitsets[idx]
                
                if union_mask == full_mask:
                    # Can remove blocks[i]
                    del blocks[i]
                    improved = True
                    break # Restart loop as indices changed
        
        return blocks

    best_solution = None
    best_size = float('inf')
    
    # Run multiple restarts
    # How many? 
    # One run takes ~1-2 seconds.
    # We have ~110 seconds.
    # So we can do ~50-100 restarts.
    
    seed = 42
    while time.time() - start_time < DEADLINE_SECONDS - 10:
        sol = run_greedy(seed)
        if sol:
            sol = improve(sol)
            if len(sol) < best_size:
                best_size = len(sol)
                best_solution = sol
                # Optional: Print progress
                # print(f"Seed {seed}: {best_size} blocks")
        seed += 1
        
    if best_solution is None:
        # Fallback: Just return a valid covering from the first successful run or empty if timeout
        # But we should have found one.
        return []
        
    return best_solution
# EVOLVE-BLOCK-END
