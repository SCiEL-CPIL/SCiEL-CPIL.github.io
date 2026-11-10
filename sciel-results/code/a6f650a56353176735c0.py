import itertools
import time

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using greedy set cover with bitset acceleration.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {subset: i for i, subset in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Generate all k-subsets (blocks) and compute their coverage masks
    # A block covers a t-subset if the t-subset is a subset of the block.
    # We represent the set of covered t-subsets as a Python integer (bitmask).
    all_blocks = []
    block_masks = []
    
    # Optimization: Pre-compute masks for all k-subsets
    # There are C(19, 7) = 50388 blocks.
    # For each block, it contains C(7, 4) = 35 t-subsets.
    for block in itertools.combinations(range(v), k):
        mask = 0
        # Check all t-subsets of this block
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
        all_blocks.append(block)
        block_masks.append(mask)
        
    # 3. Greedy Set Cover
    # Initial state: all t-subsets are uncovered
    # We use a bitmask where 1 means uncovered.
    # Since num_t_subsets can be > 64, we use Python big integers.
    # all_ones_mask = (1 << num_t_subsets) - 1
    # However, creating 1 << 3876 is fine.
    
    # To make "uncovered" easy to track, let's keep a set of uncovered indices? 
    # No, bitmasks are faster for intersection.
    # Let's use a bitmask `uncovered` where bit i is 1 if t_subsets[i] is NOT yet covered.
    uncovered = (1 << num_t_subsets) - 1
    selected_blocks = []
    selected_indices = []
    
    # We will scan all blocks each time to find the one with max gain.
    # To speed up, we can maintain a list of active candidates, but 50k is small.
    
    while uncovered:
        if time.time() > deadline:
            break
        
        best_gain = -1
        best_idx = -1
        
        # Find the block that covers the most currently uncovered t-subsets
        for i, mask in enumerate(block_masks):
            # Count bits in (mask & uncovered)
            # Python 3.8+ has int.bit_count()
            gain = (mask & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx == -1 or best_gain == 0:
            # Should not happen if the problem is solvable, but break to avoid infinite loop
            break
            
        # Select the best block
        selected_blocks.append(all_blocks[best_idx])
        selected_indices.append(best_idx)
        
        # Update uncovered mask
        uncovered &= ~block_masks[best_idx]
        
    # 4. Post-processing: Remove redundant blocks
    # If a block's coverage is a subset of the union of other selected blocks, remove it.
    # This can be done by checking if removing a block leaves all its t-subsets covered by others.
    # This is expensive if done naively O(N^2). 
    # Given the time limit and that greedy is already good, we can do a simple pass.
    
    # Let's try a simple redundancy check:
    # For each selected block, check if all its t-subsets are covered by the other selected blocks.
    # If so, remove it.
    # We need to be careful with the bitmask logic here.
    
    # Re-calculate the union of masks for all selected blocks?
    # Let's just do a few passes of removal if time permits.
    
    if time.time() < deadline - 5.0:
        for _ in range(5): # Try a few passes
            if time.time() > deadline - 2.0:
                break
            changed = False
            for i in range(len(selected_blocks)):
                if time.time() > deadline - 2.0:
                    break
                # Check if block i is redundant
                # Get the mask of block i
                mask_i = block_masks[selected_indices[i]]
                
                # Calculate union of masks of all OTHER blocks
                union_others = 0
                for j in range(len(selected_blocks)):
                    if i != j:
                        union_others |= block_masks[selected_indices[j]]
                        
                # If mask_i is subset of union_others, it's redundant
                if (mask_i & ~union_others) == 0:
                    # Remove block i
                    selected_blocks.pop(i)
                    selected_indices.pop(i)
                    changed = True
                    break # Restart scan as indices changed
            if not changed:
                break
                
    return selected_blocks
