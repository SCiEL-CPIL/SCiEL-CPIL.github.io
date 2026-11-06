import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    points = range(12)
    
    # 1. Generate all 5-subsets and map them to indices
    all_5subs = list(combinations(points, 5))
    n_5subs = len(all_5subs) # 792
    sub_to_idx = {s: i for i, s in enumerate(all_5subs)}
    
    # 2. Generate all 6-subsets (blocks)
    all_6subs = list(combinations(points, 6))
    n_6subs = len(all_6subs) # 924
    
    # 3. For each block, determine which 5-subsets it covers
    block_masks = [0] * n_6subs
    block_5subs_indices = [[] for _ in range(n_6subs)]
    
    for b_idx, b in enumerate(all_6subs):
        mask = 0
        indices = []
        for s in combinations(b, 5):
            idx = sub_to_idx[s]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[b_idx] = mask
        block_5subs_indices[b_idx] = indices
        
    # 4. Invert the incidence: for each 5-subset, which blocks cover it?
    col_to_blocks = [[] for _ in range(n_5subs)]
    for b_idx in range(n_6subs):
        for idx in block_5subs_indices[b_idx]:
            col_to_blocks[idx].append(b_idx)
            
    # 5. Backtracking search
    solution_indices = []
    
    def backtrack(uncovered_mask, current_blocks):
        if time.time() > deadline:
            return False
            
        if uncovered_mask == 0:
            solution_indices.extend(current_blocks)
            return True
            
        # Heuristic: Pick the uncovered 5-subset with the fewest valid blocks
        min_count = n_6subs + 1
        min_col = -1
        
        m = uncovered_mask
        while m:
            lsb = m & (-m)
            col_idx = lsb.bit_length() - 1
            m ^= lsb
            
            count = 0
            for b_idx in col_to_blocks[col_idx]:
                # Check if block b_idx only covers uncovered 5-subsets
                if (block_masks[b_idx] & uncovered_mask) == block_masks[b_idx]:
                    count += 1
                    if count < min_count:
                        min_count = count
                        min_col = col_idx
                        if min_count == 1:
                            break
            if min_count == 1:
                break
                
        if min_col == -1:
            return False
            
        for b_idx in col_to_blocks[min_col]:
            if (block_masks[b_idx] & uncovered_mask) == block_masks[b_idx]:
                new_mask = uncovered_mask & ~block_masks[b_idx]
                current_blocks.append(b_idx)
                if backtrack(new_mask, current_blocks):
                    return True
                current_blocks.pop()
                
        return False

    initial_mask = (1 << n_5subs) - 1
    if backtrack(initial_mask, []):
        return [list(all_6subs[i]) for i in solution_indices]
    else:
        # Fallback
        return []
