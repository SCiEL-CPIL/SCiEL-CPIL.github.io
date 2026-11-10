import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subsets_set = {ts: i for i, ts in enumerate(t_subsets)}
    num_t = len(t_subsets)
    
    # 2. Generate all k-subsets and precompute which t-subsets they cover
    # For v=19, k=8, there are 75582 blocks.
    # Each block covers C(8,5)=56 t-subsets.
    # Total operations ~ 4.2M, which is fast.
    k_subsets = list(itertools.combinations(range(v), k))
    block_t_indices = []
    for block in k_subsets:
        # Get all t-subsets contained in this block
        covered_t = []
        for ts in itertools.combinations(block, t):
            covered_t.append(t_subsets_set[ts])
        block_t_indices.append(covered_t)
    
    # 3. Greedy Construction
    # counts[i] = number of blocks currently covering t-subset i
    counts = [0] * num_t
    blocks = []
    
    # Track uncovered count for quick termination
    uncovered_count = num_t
    
    while uncovered_count > 0:
        if time.time() > deadline:
            break
            
        best_block_idx = -1
        best_gain = -1
        
        # Find the block that covers the most uncovered t-subsets
        for i, t_indices in enumerate(block_t_indices):
            gain = 0
            for idx in t_indices:
                if counts[idx] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                if gain == len(t_indices): # Perfect cover for this block's t-subsets
                    # Can't get better than covering all its t-subsets if they are all uncovered?
                    # Actually, max possible gain is len(t_indices).
                    # But other blocks might have same gain. We just pick the first max.
                    pass
        
        if best_gain == 0:
            # This should not happen if uncovered_count > 0, unless bug
            break
            
        # Add the block
        blocks.append(k_subsets[best_block_idx])
        
        # Update counts
        for idx in block_t_indices[best_block_idx]:
            if counts[idx] == 0:
                uncovered_count -= 1
            counts[idx] += 1
            
    # 4. Local Search: Remove redundant blocks
    # We iterate until no more blocks can be removed
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        # Iterate backwards to handle removals safely
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            block = blocks[i]
            # Check if this block is redundant
            # A block is redundant if for every t-subset it covers, 
            # there is at least one OTHER block covering it.
            # i.e., counts[ts] > 1 for all ts in block_t_indices[i]
            redundant = True
            for idx in block_t_indices[i]:
                if counts[idx] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove block i
                blocks.pop(i)
                # Decrement counts
                for idx in block_t_indices[i]:
                    counts[idx] -= 1
                changed = True
                # We can break the inner loop and restart the outer?
                # Or just continue checking others. Since removals don't make 
                # other blocks *more* redundant (they make counts smaller, 
                # potentially making them *less* redundant? No, 
                # counts decrease, so a block that was redundant (all counts > 1)
                # might become non-redundant if a count drops to 1.
                # So we should restart the scan to be safe, or just continue 
                # and let the next pass catch issues. 
                # Given the small size, restarting is safer and still fast.
                break
        
    # Convert blocks from tuples to lists of ints as required
    # The problem says "list of blocks... each block is a list of k distinct integers"
    final_blocks = [list(b) for b in blocks]
    
    return final_blocks
