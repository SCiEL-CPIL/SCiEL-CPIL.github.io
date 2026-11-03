def _prune_blocks(blocks, v, k, t, t_subsets_map):
    if not blocks:
        return []
        
    # Precompute t-subsets for each block to avoid repeated iteration
    block_subs = [list(itertools.combinations(b, t)) for b in blocks]
    
    # Count coverage
    counts = {}
    for subs in block_subs:
        for s in subs:
            counts[s] = counts.get(s, 0) + 1
            
    changed = True
    while changed:
        changed = False
        new_blocks = []
        new_block_subs = []
        for i, b in enumerate(blocks):
            redundant = True
            for s in block_subs[i]:
                if counts.get(s, 0) <= 1:
                    redundant = False
                    break
            
            if redundant:
                for s in block_subs[i]:
                    counts[s] -= 1
                changed = True
            else:
                new_blocks.append(b)
                new_block_subs.append(block_subs[i])
        
        blocks = new_blocks
        block_subs = new_block_subs
        
    return blocks
