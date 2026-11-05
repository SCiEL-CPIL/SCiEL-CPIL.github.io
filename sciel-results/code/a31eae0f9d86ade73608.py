import itertools
import random

def construct(v, k, t):
    points = list(range(v))
    all_subsets = list(itertools.combinations(points, t))
    subset_to_idx = {sub: i for i, sub in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    # block_subsets_idx[i] is a list of indices of subsets covered by block i
    block_subsets_idx = []
    for b in all_blocks:
        subs = [subset_to_idx[sub] for sub in itertools.combinations(b, t)]
        block_subsets_idx.append(subs)
        
    def greedy_cover_full(seed):
        random.seed(seed)
        uncovered = set(range(len(all_subsets)))
        blocks_in_use = []
        
        while uncovered:
            best_block_idx = -1
            best_gain = -1
            best_tie = float('inf')
            
            for i in range(len(all_blocks)):
                gain = 0
                # Iterate over subset indices of block i
                subs = block_subsets_idx[i]
                for count, s_idx in enumerate(subs):
                    if s_idx in uncovered:
                        gain += 1
                    # Early termination check
                    if gain + (len(subs) - count - 1) <= best_gain:
                        break
                            
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    best_tie = random.random()
                elif gain == best_gain:
                    r = random.random()
                    if r < best_tie:
                        best_tie = r
                        best_block_idx = i
                        
            if best_block_idx == -1:
                break
                
            blocks_in_use.append(best_block_idx)
            for s_idx in block_subsets_idx[best_block_idx]:
                uncovered.discard(s_idx)
                
        # Post-processing: Remove redundant blocks
        coverage_count = [0] * len(all_subsets)
        for b_idx in blocks_in_use:
            for s_idx in block_subsets_idx[b_idx]:
                coverage_count[s_idx] += 1
                
        to_keep = set(blocks_in_use)
        changed = True
        while changed:
            changed = False
            current_blocks = list(to_keep)
            random.shuffle(current_blocks)
            for b_idx in current_blocks:
                if b_idx not in to_keep:
                    continue
                is_redundant = True
                for s_idx in block_subsets_idx[b_idx]:
                    if coverage_count[s_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    to_keep.remove(b_idx)
                    for s_idx in block_subsets_idx[b_idx]:
                        coverage_count[s_idx] -= 1
                    changed = True
                    
        final_blocks = [list(all_blocks[i]) for i in to_keep]
        return final_blocks

    best_len = float('inf')
    best_blocks = []
    
    # Number of restarts. 4 seems reasonable.
    for seed in range(4):
        blocks = greedy_cover_full(seed)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks

    return best_blocks
