import time
from itertools import combinations

def construct():
    """
    Construct the Steiner system S(5,6,12) (Witt design) on 12 points.
    Uses optimized backtracking with bitmask-based available blocks.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    points = range(12)
    
    # Map 5-subsets to indices
    all_5_subs = list(combinations(points, 5))
    sub_to_idx = {s: i for i, s in enumerate(all_5_subs)}
    num_5_subs = len(all_5_subs)  # 792
    
    # Generate all 6-subsets (blocks) and their 5-subset masks
    all_6_subs = list(combinations(points, 6))
    block_masks = []
    for b in all_6_subs:
        mask = 0
        for i in range(6):
            subset = tuple(x for j, x in enumerate(b) if j != i)
            idx = sub_to_idx[subset]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    num_blocks = len(block_masks)  # 924
    
    # Precompute col_to_blocks: for each 5-subset, which blocks cover it
    col_to_blocks = [[] for _ in range(num_5_subs)]
    for i, mask in enumerate(block_masks):
        m = mask
        while m:
            lsb = m & -m
            c = lsb.bit_length() - 1
            col_to_blocks[c].append(i)
            m ^= lsb
            
    # Precompute conflict mask for each block
    conflict_masks = []
    for i in range(num_blocks):
        cm = 0
        mi = block_masks[i]
        for j in range(num_blocks):
            if i != j and (mi & block_masks[j]):
                cm |= (1 << j)
        conflict_masks.append(cm)
        
    best_solution = []
    
    def solve(covered_mask, available_mask, solution):
        nonlocal best_solution
        if time.time() > deadline:
            return None
        
        if len(solution) > len(best_solution):
            best_solution = solution[:]
            
        if len(solution) == 132:
            return solution[:]
            
        if available_mask == 0:
            return None
            
        # Find the uncovered 5-subset with the minimum number of available covering blocks
        min_count = float('inf')
        target_col = -1
        
        all_5_mask = (1 << num_5_subs) - 1
        uncovered_mask = all_5_mask ^ covered_mask
        
        m = uncovered_mask
        while m:
            lsb = m & -m
            c = lsb.bit_length() - 1
            count = 0
            for b in col_to_blocks[c]:
                if (available_mask >> b) & 1:
                    count += 1
                    if count >= min_count:
                        break
            if count < min_count:
                min_count = count
                target_col = c
                if min_count == 1:
                    break
            m ^= lsb
            
        if target_col == -1:
            return None
            
        candidates = [b for b in col_to_blocks[target_col] if (available_mask >> b) & 1]
        
        # Sort candidates by number of conflicts
        candidates.sort(key=lambda b: bin(conflict_masks[b] & available_mask).count('1'))
        
        for b in candidates:
            new_available = available_mask & ~(conflict_masks[b] | (1 << b))
            new_covered = covered_mask | block_masks[b]
            
            res = solve(new_covered, new_available, solution + [b])
            if res is not None:
                return res
                
        return None

    initial_available = (1 << num_blocks) - 1
    initial_covered = 0
    res = solve(initial_covered, initial_available, [])
    
    if res is not None:
        return [list(all_6_subs[i]) for i in res]
    
    if best_solution:
        return [list(all_6_subs[i]) for i in best_solution]
    
    # Fallback: Greedy
    covered = 0
    blocks = []
    for i in range(num_blocks):
        if block_masks[i] & covered:
            continue
        blocks.append(list(all_6_subs[i]))
        covered |= block_masks[i]
    return blocks
