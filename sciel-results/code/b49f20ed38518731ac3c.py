import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute all blocks and their masks
    # A block is a k-subset of v points
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # block_masks[i] is an integer bitmask of t-subsets covered by block i
    block_masks = [0] * num_blocks
    
    # For each t-subset, store list of block indices that cover it
    # t_subset_to_blocks[i] is a list of block indices
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]
    
    # Precomputation loop
    # This might take a second or two, which is acceptable
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[sub]
            mask |= (1 << idx)
            t_subset_to_blocks[idx].append(b_idx)
        block_masks[b_idx] = mask
    
    # Greedy function
    def greedy_once(random_seed=None):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
        
        # Bitset of uncovered t-subsets
        uncovered_mask = (1 << num_t_subsets) - 1
        blocks_used = []
        
        # To pick a random uncovered t-subset efficiently:
        # We can't just pick a random index and check, as it might fail often late in the game.
        # However, with 116k bits, random sampling is decent until coverage is very high.
        # A better way: maintain a list of uncovered indices? 
        # Updating a list of 116k items is O(N).
        # Let's stick to random sampling with a fallback.
        
        while uncovered_mask != 0:
            # Pick a random uncovered t-subset
            s_idx = None
            # Try a few random indices
            for _ in range(20):
                cand_idx = rng.randrange(num_t_subsets)
                if (uncovered_mask >> cand_idx) & 1:
                    s_idx = cand_idx
                    break
            
            if s_idx is None:
                # Fallback: find first set bit
                # This is slow, but should rarely happen if random sampling works
                # We can use a trick: convert to binary string? No, too slow.
                # Just iterate.
                for i in range(num_t_subsets):
                    if (uncovered_mask >> i) & 1:
                        s_idx = i
                        break
                if s_idx is None:
                    break # Should not happen if uncovered_mask != 0
            
            # Get candidate blocks covering this t-subset
            candidates = t_subset_to_blocks[s_idx]
            
            best_block_idx = -1
            best_gain = -1
            
            # Evaluate candidates
            # Optimization: If we have many candidates, we can't just check all if it's too slow.
            # But 364 is small.
            for b_idx in candidates:
                gain = (uncovered_mask & block_masks[b_idx]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                elif gain == best_gain and gain > 0:
                    # Random tie-break
                    if rng.random() < 0.5:
                        best_block_idx = b_idx
            
            if best_block_idx == -1:
                # No block covers this? Impossible if valid covering exists and we haven't finished
                # But if we are stuck, break
                break
            
            blocks_used.append(best_block_idx)
            uncovered_mask &= ~block_masks[best_block_idx]
            
            # Time check
            if time.time() - start_time > time_limit - 5.0:
                return None
        
        return blocks_used

    def local_search(block_indices, random_seed=None):
        """Remove redundant blocks."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        # Convert to set for fast lookup? No, we need to update counts.
        # cover_count[i] = number of blocks in current set that cover t-subset i
        cover_count = [0] * num_t_subsets
        
        # Initialize cover counts
        current_blocks = list(block_indices)
        for b_idx in current_blocks:
            mask = block_masks[b_idx]
            # Iterate set bits in mask
            # Efficient way to iterate set bits in Python int:
            # while mask:
            #   lsb = mask & -mask
            #   idx = lsb.bit_length() - 1
            #   cover_count[idx] += 1
            #   mask ^= lsb
            
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
        
        # Try to remove blocks
        # We will iterate through blocks in random order.
        # If a block is removable (all its t-subsets have count > 1), remove it.
        # Repeat until no more removals possible.
        
        changed = True
        while changed:
            changed = False
            # Shuffle current blocks to avoid bias
            rng.shuffle(current_blocks)
            
            # We need to check removal in a way that doesn't invalidate indices easily.
            # We will build a new list.
            new_blocks = []
            
            for b_idx in current_blocks:
                mask = block_masks[b_idx]
                
                # Check if all t-subsets in this block have count > 1
                removable = True
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] <= 1:
                        removable = False
                        break
                    m ^= lsb
                
                if removable:
                    # Remove this block
                    # Decrement counts
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                    changed = True
                else:
                    new_blocks.append(b_idx)
            
            current_blocks = new_blocks
            
            # Time check
            if time.time() - start_time > time_limit - 5.0:
                return current_blocks
        
        return current_blocks

    best_blocks = None
    best_len = float('inf')
    
    num_restarts = 0
    while time.time() - start_time < time_limit - 10.0:
        seed = num_restarts
        res = greedy_once(seed)
        num_restarts += 1
        
        if res is not None:
            # Apply local search
            refined = local_search(res, seed)
            
            if len(refined) < best_len:
                best_len = len(refined)
                best_blocks = refined
                
        if best_len <= 264:
            break
            
        # If we are running low on time, do one final check
        if time.time() - start_time > time_limit - 15.0:
             # Try to do a few more quick greeds if we have time
             pass

    if best_blocks is None:
        # Fallback
        res = greedy_once(0)
        if res:
            best_blocks = local_search(res, 0)
            
    # Convert indices back to lists of points
    # The interface expects a list of blocks, where each block is a list of k distinct integers
    final_blocks = [list(all_blocks[i]) for i in best_blocks]
    
    return final_blocks
