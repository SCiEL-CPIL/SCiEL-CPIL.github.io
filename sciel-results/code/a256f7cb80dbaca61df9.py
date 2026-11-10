import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0  # Safety margin

    # Precompute t-subsets and their indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {s: i for i, s in enumerate(all_tsubs)}
    
    # Precompute coverage masks for all candidate blocks
    # Each block covers C(k,t) t-subsets
    # We'll represent coverage as a bitset (Python int with n_tsubs bits)
    
    # For v=24, k=7, t=3: C(24,3)=2024 t-subsets, C(7,3)=35 per block
    # C(24,7)=346105 candidate blocks - too many to precompute all
    # Instead, we'll compute masks on-demand but cache them
    
    mask_cache = {}
    
    def get_block_mask(block):
        """Get or compute the coverage bitset for a block."""
        key = tuple(block)
        if key in mask_cache:
            return mask_cache[key]
        
        mask = 0
        for combo in itertools.combinations(block, t):
            idx = tsub_index[combo]
            mask |= (1 << idx)
        
        mask_cache[key] = mask
        return mask

    best_blocks = []
    best_count = float('inf')
    
    # Multiple restarts
    max_restarts = 200  # More restarts possible due to speedup
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(restart)
        
        # Uncovered t-subsets as a bitset
        uncovered = (1 << n_tsubs) - 1  # All bits set initially
        
        blocks = []
        block_masks = []
        
        # Greedy Phase with bitset operations
        while True:
            if time.time() > deadline - 3.0:
                break
            if uncovered == 0:
                break
            
            # Pick a random uncovered t-subset
            # Find one set bit in uncovered
            # Use bit_length and lowest set bit trick
            # Get a random uncovered index
            # For efficiency, pick the lowest set bit (deterministic) or random
            # Let's use a hybrid: pick from a sample of uncovered indices
            
            # Quick way: find some uncovered index
            # uncovered is a Python int, we can use bit operations
            # To get a random uncovered index, we can:
            # 1. Count bits (popcount)
            # 2. Pick random rank
            # 3. Find that bit
            # But this is complex. Simpler: just use the lowest set bit for now
            # Or: convert to numpy for random selection (slower but simpler)
            
            # Actually, let's just pick the lowest set bit for speed
            # This gives a deterministic but valid choice
            # For diversity, we can XOR with a random pattern or use restart index
            
            # Get lowest set bit index
            if uncovered & 1:
                crit_idx = 0
            else:
                # Find lowest set bit
                temp = uncovered
                idx = 0
                while (temp & 1) == 0:
                    temp >>= 1
                    idx += 1
                crit_idx = idx
            
            crit_tsub = all_tsubs[crit_idx]
            
            # Generate all k-subsets containing crit_tsub
            remaining_points = [x for x in range(v) if x not in crit_tsub]
            
            best_block = None
            best_gain = -1
            best_mask = 0
            
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(crit_tsub + combo))
                mask = get_block_mask(block)
                
                # Gain = number of newly covered t-subsets = popcount(mask & uncovered)
                new_cover = mask & uncovered
                gain = bin(new_cover).count('1')
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    best_mask = mask
            
            if best_block is None or best_gain <= 0:
                break
            
            blocks.append(list(best_block))
            block_masks.append(best_mask)
            
            # Update uncovered: clear the bits in best_mask
            uncovered &= ~best_mask
        
        # Local Search: Remove redundant blocks
        # Build coverage count using bit operations
        # cov_count[i] = number of blocks whose mask has bit i set
        
        if blocks:
            cov_count = np.zeros(n_tsubs, dtype=np.int32)
            for bm in block_masks:
                # Extract set bits from bm and increment cov_count
                # This is slow for large bitsets. Alternative:
                # For each block, iterate over its 35 t-subsets
                for b in blocks:
                    pass  # We'll use a different approach
                
            # Simpler: rebuild cov_count by iterating blocks' t-subsets
            cov_count = np.zeros(n_tsubs, dtype=np.int32)
            for b in blocks:
                for combo in itertools.combinations(b, t):
                    idx = tsub_index[combo]
                    cov_count[idx] += 1
            
            final_blocks = list(blocks)
            final_masks = list(block_masks)
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                # Recalculate counts
                cov_count = np.zeros(n_tsubs, dtype=np.int32)
                for b in final_blocks:
                    for combo in itertools.combinations(b, t):
                        idx = tsub_index[combo]
                        cov_count[idx] += 1
                
                # Try to remove each block (from end for efficiency)
                for i in range(len(final_blocks) - 1, -1, -1):
                    if time.time() > deadline - 2.0:
                        break
                    b = final_blocks[i]
                    # Check if all t-subsets in b have count > 1
                    redundant = True
                    for combo in itertools.combinations(b, t):
                        idx = tsub_index[combo]
                        if cov_count[idx] == 1:
                            redundant = False
                            break
                    if redundant:
                        final_blocks.pop(i)
                        final_masks.pop(i)
                        improved = True
                        break  # Restart loop as counts changed
            
            if len(final_blocks) < best_count:
                best_count = len(final_blocks)
                best_blocks = final_blocks

    return best_blocks
