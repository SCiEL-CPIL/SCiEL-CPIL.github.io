import itertools
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a Max-Gain Greedy algorithm
    followed by redundancy removal.
    """
    start_time = time.time()
    deadline = 115.0  # Leave 5s margin for safety

    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Precompute the t-subsets for each k-block
    # Using frozenset for efficient hashing and intersection
    block_tsubs = []
    for b in all_blocks:
        block_tsubs.append(frozenset(itertools.combinations(b, t)))
        
    all_tsubs = set(itertools.combinations(points, t))
    uncovered = set(all_tsubs)
    
    selected_block_indices = []
    
    # Phase 1: Greedy Construction
    # Select blocks that cover the maximum number of currently uncovered t-subsets
    while uncovered:
        if time.time() > deadline:
            break
            
        best_idx = -1
        max_gain = 0
        max_possible_gain = len(block_tsubs[0]) # C(k, t)
        
        for i in range(n_blocks):
            # If we already found a block that covers all its t-subsets, 
            # we can't do better than that.
            if max_gain == max_possible_gain:
                break
                
            gain = len(uncovered.intersection(block_tsubs[i]))
            if gain > max_gain:
                max_gain = gain
                best_idx = i
        
        if best_idx == -1:
            break
            
        selected_block_indices.append(best_idx)
        uncovered.difference_update(block_tsubs[best_idx])
        
    # Phase 2: Redundancy Removal
    # Remove blocks that do not uniquely cover any t-subset
    if selected_block_indices:
        covered_counts = {sub: 0 for sub in all_tsubs}
        for idx in selected_block_indices:
            for sub in block_tsubs[idx]:
                covered_counts[sub] += 1
        
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            i = 0
            while i < len(selected_block_indices):
                idx = selected_block_indices[i]
                redundant = True
                for sub in block_tsubs[idx]:
                    if covered_counts[sub] == 1:
                        redundant = False
                        break
                
                if redundant:
                    for sub in block_tsubs[idx]:
                        covered_counts[sub] -= 1
                    selected_block_indices.pop(i)
                    changed = True
                    # Check the same index again as list shifted
                else:
                    i += 1
        
        # Return the blocks as lists of integers
        blocks = [list(all_blocks[idx]) for idx in selected_block_indices]
        return blocks
    
    # Fallback (should not be reached)
    return [list(all_blocks[0])]
