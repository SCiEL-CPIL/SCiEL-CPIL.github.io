import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe within 120s
    
    # Generate all k-subsets (blocks)
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Map each block to an index
    # We will store the blocks themselves in a list for easy access
    block_list = [list(b) for b in all_blocks]
    
    # Precompute the t-subsets for each block
    # Represent t-subsets as sorted tuples or integers.
    # For v=19, t=4, we can use a 19-bit integer mask.
    # sub_mask = 0
    # for p in sub: sub_mask |= (1 << p)
    
    block_subs = []
    sub_to_blocks = {}
    
    # We need to iterate over all blocks and all their t-subsets
    # This is 50388 * 35 operations, which is ~1.7M. This is fast enough.
    
    for i, block in enumerate(block_list):
        subs = []
        for sub in itertools.combinations(block, t):
            mask = 0
            for p in sub:
                mask |= (1 << p)
            subs.append(mask)
            if mask not in sub_to_blocks:
                sub_to_blocks[mask] = []
            sub_to_blocks[mask].append(i)
        block_subs.append(subs)
        
    # Initialize gain for each block
    # gain[i] = number of t-subsets in block i
    # Initially all are uncovered, so gain[i] = C(k, t)
    initial_gain = len(block_subs[0]) if block_subs else 0
    gain = [initial_gain] * num_blocks
    
    # uncovered_mask: set of masks that are currently uncovered
    # We can use a set of masks.
    # Or a dictionary mask -> count? No, we just need to know if it's covered.
    # Since we only add coverage, we can maintain a set of covered masks.
    # But checking `if mask in covered` is O(1).
    # However, we need to update gains.
    # We need to know which subsets are still uncovered to subtract from gains.
    # So we maintain `uncovered_subs` as a set of masks.
    
    all_subs_set = set(sub_to_blocks.keys())
    uncovered_subs = set(all_subs_set)
    
    # To make gain updates fast, we iterate through the subsets of the chosen block.
    # For each subset, if it is uncovered, we remove it from uncovered_subs.
    # Then for every block that contains this subset, we decrement its gain.
    
    blocks_used = []
    
    while uncovered_subs:
        if time.time() > deadline:
            break
        
        # Find block with max gain
        # Using max() on a list of 50k integers is fast (~ms)
        # But we need the index.
        best_idx = -1
        max_gain = -1
        for i in range(num_blocks):
            if gain[i] > max_gain:
                max_gain = gain[i]
                best_idx = i
                
        if best_idx == -1 or max_gain == 0:
            # This should not happen if uncovered_subs is not empty
            # It implies a bug or no block can cover the remaining subsets
            break
            
        blocks_used.append(block_list[best_idx])
        
        # Update gains
        for sub_mask in block_subs[best_idx]:
            if sub_mask in uncovered_subs:
                uncovered_subs.remove(sub_mask)
                # Decrement gain for all blocks covering this sub_mask
                for other_idx in sub_to_blocks[sub_mask]:
                    gain[other_idx] -= 1
                    
    return blocks_used
