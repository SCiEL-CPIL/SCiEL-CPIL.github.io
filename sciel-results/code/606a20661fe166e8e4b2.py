import itertools
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Precompute all t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    subset_index = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    # Precompute for each point x and each (t-1)-subset y, the index of the t-subset y + x
    # This helps in fast bitset updates
    # For t=3, y is a pair.
    # Let's just precompute the bitmask for each k-subset block
    # There are C(v, k) possible blocks. For v=24, k=7, C(24,7) = 346104.
    # Storing a 2024-bit int for each is feasible.
    
    # Optimization: We don't need to precompute ALL blocks if we generate them on the fly 
    # or use a smarter structure. But for VNS, we need to evaluate blocks quickly.
    # Let's precompute the bitmask for all possible k-subsets.
    # 346k blocks * 2024 bits ~ 86 MB. Acceptable.
    
    block_masks = {}
    all_k_subsets = list(itertools.combinations(range(v), k))
    for block in all_k_subsets:
        mask = 0
        for t_sub in itertools.combinations(block, t):
            mask |= (1 << subset_index[t_sub])
        block_masks[block] = mask
        
    # Map from block tuple to mask for quick lookup
    # Also, we might want to map mask to block? No, multiple blocks can have same mask? No, distinct blocks have distinct sets of t-subsets?
    # Actually, different k-subsets can cover the same set of t-subsets if v is large relative to k?
    # No, if block A != block B, there is an element in A not in B. 
    # Does A cover a t-subset that B doesn't? Yes, any t-subset containing that element and t-1 others from A\{x} is in A but not B (unless B has those others too, but B doesn't have x).
    # So masks are unique for distinct blocks? 
    # Wait, if t < k, a block covers C(k,t) subsets. 
    # Two different k-subsets A and B. If A != B, let x in A\B. 
    # Pick y in A\{x} of size t-1. {x} U y is a t-subset in A. 
    # Is it in B? No, because x not in B. 
    # So yes, the set of covered t-subsets uniquely identifies the block (for t >= 1, k > t).
    
    # Initialize with a greedy covering
    # uncovered_mask: bits set for uncovered subsets
    # Initially all 1s
    uncovered_mask = (1 << num_t_subsets) - 1
    current_blocks = []
    current_mask = 0
    
    # Greedy Step 1: Fast start
    # Instead of picking the best block globally (expensive), we can use a randomized greedy or standard greedy.
    # Standard greedy: pick block with max gain.
    # To speed up, we can use a heuristic: pick random block? No, greedy is better for initialization.
    # But checking all 346k blocks every step is slow.
    # Let's use a randomized greedy for initialization to be fast, then refine.
    
    def greedy_init():
        unc = (1 << num_t_subsets) - 1
        blocks = []
        while unc:
            # Pick a random uncovered subset to start a block?
            # Or pick the block that covers the most uncovered?
            # For speed, let's pick a random uncovered t-subset and find a random block containing it?
            # No, let's use a slightly smarter greedy:
            # Pick a random uncovered t-subset s.
            # Find all blocks containing s. Pick one that covers most other uncovered subsets.
            
            # Get an index of an uncovered bit
            # Python ints don't have a fast "pick random set bit".
            # We can find the lowest set bit.
            lsb = unc & -unc
            idx = lsb.bit_length() - 1
            s = all_t_subsets[idx]
            
            # Find blocks containing s
            # Precompute: for each t-subset, list of blocks containing it?
            # That's C(v,t) * C(v-t, k-t) lists.
            # For v=24, t=3, k=7: 2024 * C(21,4) = 2024 * 5985 ~ 12M entries. Too much memory/time to precompute fully.
            
            # Alternative: Just iterate all blocks? 346k blocks.
            # Check if block contains s: 
            # We can precompute for each block its t-subsets? No, we have masks.
            # Check if mask has bit idx.
            
            best_block = None
            best_gain = -1
            # To speed this up, we can sample a subset of blocks?
            # Or just iterate all. 346k * (bit check) is fast in Python if optimized.
            # But doing this for ~80 blocks is 80 * 346k = 27M checks. Might be slow in pure Python loop.
            
            # Let's try a different initialization: Randomized Greedy
            # Pick random block, if it covers anything, add it.
            # This is very fast but poor quality.
            
            # Let's stick to a simple heuristic:
            # Pick random uncovered t-subset.
            # Pick random block containing it.
            # How to find a block containing s?
            # We can precompute: for each point x, list of blocks containing x?
            # Or for each t-subset s, list of blocks containing s is hard.
            
            # Let's use a "Randomized Greedy" that picks a random block from the full set, 
            # and keeps it if it adds coverage. This is O(N_blocks * N_iter).
            # 346k * 80 = 27M. With simple bit ops, this is ~1-2 seconds.
            
            shuffled_blocks = list(all_k_subsets)
            random.shuffle(shuffled_blocks)
            for b in shuffled_blocks:
                m = block_masks[b]
                if m & unc:
                    blocks.append(b)
                    unc &= ~m
                    if not unc:
                        break
            # If not covered, continue with standard greedy on remaining?
            # The above might not cover everything if we run out of blocks (we have all blocks, so we will cover everything).
            return blocks, unc

    current_blocks, uncovered_mask = greedy_init()
    
    def remove_unnecessary_blocks(blocks):
        """Remove blocks that are redundant."""
        # Calculate coverage mask for all blocks
        # We want to remove b if mask(b) is subset of OR(mask(others))
        # This is equivalent to: if we remove b, do we still cover all subsets?
        # i.e., is (total_mask ^ mask(b)) == total_mask? No.
        # Is (total_mask & ~mask(b)) == total_mask? No.
        # We need to check if every t-subset covered by b is also covered by some other block.
        
        # Efficient way:
        # For each block b, check if there is any t-subset in b that is NOT covered by any other block.
        # If no such t-subset exists, b is redundant.
        
        # Precompute for each t-subset, the list of blocks covering it?
        # Or, for each block b, iterate its t-subsets.
        # For each t-subset s in b, check if any other block covers s.
        
        # Let's maintain a count of how many blocks cover each t-subset.
        # counts[i] = number of blocks covering t-subset i.
        
        counts = [0] * num_t_subsets
        for b in blocks:
            m = block_masks[b]
            # Iterate set bits in m
            mm = m
            while mm:
                lsb = mm & -mm
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                mm ^= lsb
        
        new_blocks = []
        for b in blocks:
            m = block_masks[b]
            is_redundant = True
            mm = m
            while mm:
                lsb = mm & -mm
                idx = lsb.bit_length() - 1
                if counts[idx] == 1:
                    is_redundant = False
                    break
                mm ^= lsb
            
            if is_redundant:
                # Remove b
                mm = m
                while mm:
                    lsb = mm & -mm
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    mm ^= lsb
            else:
                new_blocks.append(b)
                
        return new_blocks

    # Local Search Loop
    best_blocks = current_blocks
    best_len = len(current_blocks)
    
    # Precompute randomization seed
    random.seed(42)
    
    while time.time() < deadline:
        # 1. Try to remove redundant blocks
        candidate = remove_unnecessary_blocks(current_blocks)
        
        if len(candidate) < best_len:
            best_blocks = candidate
            best_len = len(candidate)
            current_blocks = candidate
        
        # 2. Perturbation: Add a random block
        # Pick a random block from all_k_subsets
        # If it's not already in current_blocks, add it
        b = random.choice(all_k_subsets)
        if b not in current_blocks:
            current_blocks.append(b)
            
        # 3. Repair/Optimize: 
        # The addition might create redundancies. 
        # Run remove_unnecessary_blocks again? 
        # Or, a more aggressive local search:
        # Try to swap a block with another?
        
        # Let's just rely on remove_unnecessary_blocks to clean up.
        # But if we add a block, we might be able to remove two others?
        # remove_unnecessary_blocks only removes 1 if it's redundant.
        # It doesn't do "exchange" moves.
        
        # To do exchanges, we can:
        # 1. Remove a random block b1.
        # 2. Check if covering is broken.
        # 3. If broken, add a random block b2 that covers the new holes.
        # 4. Check if we can remove b3?
        
        # Simpler ILS:
        # 1. Remove a random block from current_blocks.
        # 2. If uncovered becomes non-zero, we must repair.
        #    Find an uncovered t-subset s.
        #    Find a block b covering s. Add b.
        # 3. Run remove_unnecessary_blocks.
        
        if current_blocks:
            # Pick a random block to remove
            idx_rm = random.randint(0, len(current_blocks) - 1)
            b_rm = current_blocks[idx_rm]
            current_blocks.pop(idx_rm)
            
            # Check validity
            # Recalculate uncovered mask?
            # We need to know what was uncovered by removing b_rm.
            # This is complex to track incrementally without a full recompute or careful counting.
            
            # Let's recompute the uncovered mask for the current set of blocks.
            # This is O(N_blocks * 2^t) roughly.
            covered_mask = 0
            for b in current_blocks:
                covered_mask |= block_masks[b]
                
            uncovered = ((1 << num_t_subsets) - 1) & ~covered_mask
            
            if uncovered:
                # Repair: Add blocks until covered
                # Greedy repair
                while uncovered:
                    # Pick random uncovered t-subset
                    lsb = uncovered & -uncovered
                    idx = lsb.bit_length() - 1
                    s = all_t_subsets[idx]
                    
                    # Find a block covering s
                    # We can pick a random block from all_k_subsets that covers s
                    # To find one quickly, we can precompute: for each t-subset, a list of blocks?
                    # Memory: 2024 * 346k is too big.
                    # But we can just iterate all_k_subsets and find the first one that has bit idx set.
                    # This is O(346k).
                    # Doing this for ~10 repairs is ~3M checks. Fast.
                    
                    repair_block = None
                    for b in all_k_subsets:
                        if block_masks[b] & (1 << idx):
                            repair_block = b
                            break
                            
                    if repair_block and repair_block not in current_blocks:
                        current_blocks.append(repair_block)
                        uncovered &= ~block_masks[repair_block]
                    else:
                        # Should not happen if repair_block is valid
                        break
                        
            # After repair, try to remove redundancies
            candidate = remove_unnecessary_blocks(current_blocks)
            
            if len(candidate) < best_len:
                best_blocks = candidate
                best_len = len(candidate)
                current_blocks = candidate
            else:
                # If no improvement, we might revert or keep exploring?
                # ILS usually reverts if no improvement, but here we can keep the "worse" state
                # to explore further? No, keeping the best state is safer.
                # However, if we always revert to best, we might get stuck.
                # Let's keep the current state (which might be worse) for a few iterations?
                # Or just always keep the best?
                # Let's try: if no improvement, revert to best_blocks.
                current_blocks = best_blocks
                # But this means we never explore worse states to find a path to a better one?
                # Actually, the "remove random block" step is the perturbation.
                # If we revert, we are just repeating the same perturbation on the same best state.
                # We should randomize the perturbation.
                
                # To allow exploration, we can keep the current state if it's not *too* bad?
                # Or just accept that we are doing a random walk on the state space of coverings.
                # Let's keep the current state (even if worse) to allow further moves.
                # But we must ensure we don't drift too far.
                # Let's keep the current state.
                pass 

    return [list(b) for b in best_blocks]

# EVOLVE-BLOCK-END
