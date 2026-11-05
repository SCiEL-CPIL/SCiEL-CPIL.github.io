import itertools
import sys

def construct(v, k, t):
    points = range(v)
    
    # 1. Enumerate all t-subsets and map them to indices
    t_subs = list(itertools.combinations(points, t))
    n_t_subs = len(t_subs)
    # Map t-subset (tuple) to index
    sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Enumerate all k-blocks and compute their coverage masks
    # A block covers a t-subset if the t-subset is a subset of the block
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    # Precompute masks for each k-block
    # mask is an integer where bit i is 1 if t_subs[i] is subset of block
    block_masks = [0] * n_k_blocks
    
    # Optimization: For each k-block, iterate through its t-subsets?
    # Or for each t-subset, iterate through k-blocks containing it?
    # Given v=19, k=8, t=5:
    # n_t_subs = 11628
    # n_k_blocks = 75582
    # 
    # Method A: For each block, generate all t-subsets of it.
    # Number of t-subsets in a k-block is C(8,5) = 56.
    # Total operations: 75582 * 56 = 4.2M. This is fast.
    
    for i, block in enumerate(k_blocks):
        block_set = set(block)
        mask = 0
        # Generate all t-subsets of this block
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    # 3. Greedy Set Cover
    uncovered = (1 << n_t_subs) - 1
    selected_block_indices = []
    
    # We need to find the block with max gain.
    # Naive scan: O(n_k_blocks) per step.
    # Steps ~ 335. Total ~ 25M ops.
    # Each op: bitwise AND and popcount.
    # Python int popcount is fast (C implementation).
    
    # To speed up, we can maintain a list of active blocks, but scanning all is likely fine.
    # We can also sort blocks by initial coverage? No, coverage changes.
    
    while uncovered:
        best_idx = -1
        max_gain = 0
        
        # Iterate through all blocks to find the one covering most uncovered subsets
        for i in range(n_k_blocks):
            # If the block covers nothing new, gain is 0
            # We can skip if block_masks[i] & uncovered == 0, but that's the same cost as AND+popcount roughly.
            gain = (block_masks[i] & uncovered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_idx = i
                
        if best_idx == -1:
            # Should not happen if uncovered is not empty and we have valid blocks
            break
            
        selected_block_indices.append(best_idx)
        uncovered &= ~block_masks[best_idx]
        
    # 4. Local Search: Remove redundant blocks
    # A block is redundant if all t-subsets it covers are also covered by other selected blocks.
    # Let S be the set of selected indices.
    # For each b in S, check if mask[b] is subset of (OR of mask[x] for x in S, x != b).
    
    if selected_block_indices:
        # Compute OR of all selected masks
        total_mask = 0
        for idx in selected_block_indices:
            total_mask |= block_masks[idx]
            
        new_selected = []
        for idx in selected_block_indices:
            # Check if this block contributes uniquely
            # If (mask[idx] & ~ (total_mask ^ mask[idx])) == 0, then it's redundant?
            # No. We need to check if mask[idx] is covered by the union of the *other* blocks.
            # Let others_mask = total_mask ^ mask[idx] is NOT correct because bits might be in both.
            # others_mask = total_mask & ~mask[idx] is also not enough.
            # We need: Is there any bit in mask[idx] that is NOT in (OR of others)?
            # Let union_others = OR of mask[j] for j in selected, j != idx.
            # If mask[idx] & ~union_others == 0, then redundant.
            
            # Efficient way:
            # union_others = total_mask 
            # But total_mask includes mask[idx].
            # If a bit is in mask[idx] and in total_mask, it might be ONLY in mask[idx].
            # So we need to calculate union_others properly or use a trick.
            
            # Trick:
            # If mask[idx] is a subset of (total_mask ^ mask[idx])? No.
            # Let's just compute union_others for each candidate.
            # Since N is small (~335), O(N^2) is fine.
            
            # Optimization: 
            # If we iterate and remove, we should be careful.
            # Let's just filter once.
            
            pass # Placeholder for logic below
            
        # Better Local Search Logic:
        # Iterate through selected blocks.
        # For each block, calculate coverage provided by all OTHER selected blocks.
        # If this block adds no new coverage, remove it.
        
        # To do this efficiently:
        # We can maintain a count of how many selected blocks cover each t-subset?
        # Or just re-calculate union of others.
        
        # Let's use the count array approach for robustness and speed.
        # count[i] = number of selected blocks covering t-subset i.
        
        counts = [0] * n_t_subs
        for idx in selected_block_indices:
            m = block_masks[idx]
            # Increment counts for all bits set in m
            # This is O(n_t_subs) per block if done naively.
            # 335 * 11628 = 3.9M ops. Fast enough.
            # But iterating bits is faster if we use bit tricks?
            # Python: while m: lsb = m & -m; idx_bit = lsb.bit_length()-1; counts[idx_bit]+=1; m ^= lsb
            # Number of set bits in a block mask is C(8,5)=56.
            # So 335 * 56 = 18k ops. Very fast.
            
            m = block_masks[idx]
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                counts[bit_idx] += 1
                m ^= lsb
                
        # Now, check for redundancy
        final_indices = []
        for idx in selected_block_indices:
            # Check if this block has any unique coverage
            # i.e., if there is a bit set in block_masks[idx] where counts[bit] == 1
            # If yes, it's essential. If no, it's redundant.
            
            m = block_masks[idx]
            essential = False
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    essential = True
                    break
                m ^= lsb
                
            if essential:
                final_indices.append(idx)
            else:
                # Decrement counts for this block
                m = block_masks[idx]
                while m:
                    lsb = m & -m
                    bit_idx = lsb.bit_length() - 1
                    counts[bit_idx] -= 1
                    m ^= lsb
                    
        selected_block_indices = final_indices
        
    # Convert indices back to blocks
    return [list(k_blocks[i]) for i in selected_block_indices]
