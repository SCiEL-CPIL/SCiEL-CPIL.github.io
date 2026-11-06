import time
from itertools import combinations

def construct():
    """
    Constructs the Steiner system S(5,6,12) by first constructing S(4,5,11) 
    and then extending it.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Step 1: Construct S(4,5,11)
    # Points: 0..10
    n11 = 11
    k11 = 5
    t11 = 4
    
    all_5subs_11 = list(combinations(range(n11), k11))
    num_5subs_11 = len(all_5subs_11) # 462
    
    all_4subs_11 = list(combinations(range(n11), t11))
    num_4subs_11 = len(all_4subs_11) # 330
    
    sub_to_idx_11 = {s: i for i, s in enumerate(all_4subs_11)}
    
    # Precompute masks for blocks in S(4,5,11)
    # Mask bit i is set if 4-subset i is covered
    block_masks_11 = [0] * num_5subs_11
    blocks_covering_4sub_11 = [[] for _ in range(num_4subs_11)]
    
    for b_idx, b in enumerate(all_5subs_11):
        mask = 0
        for sub in combinations(b, t11):
            idx = sub_to_idx_11[sub]
            mask |= (1 << idx)
            blocks_covering_4sub_11[idx].append(b_idx)
        block_masks_11[b_idx] = mask

    FULL_MASK_11 = (1 << num_4subs_11) - 1
    
    solution_11 = []
    found_11 = False
    
    def search_11(covered_mask, candidates):
        nonlocal solution_11, found_11
        if time.time() > deadline:
            return False
            
        if covered_mask == FULL_MASK_11:
            if len(solution_11) == 66: # Expected optimal
                found_11 = True
            return True
            
        # Pruning: Check if enough candidates remain to cover all uncovered 4-subsets
        # Each candidate covers exactly 5 uncovered 4-subsets (since they must be disjoint from covered_mask)
        # If len(candidates) * 5 < number of uncovered 4-subsets, it's impossible.
        uncovered_count = num_4subs_11 - bin(covered_mask).count('1')
        if len(candidates) * 5 < uncovered_count:
            return False

        # MRV Heuristic
        min_count = float('inf')
        target_sub = -1
        
        for s_idx in range(num_4subs_11):
            if (covered_mask >> s_idx) & 1:
                continue
                
            count = 0
            for b_idx in blocks_covering_4sub_11[s_idx]:
                if b_idx in candidates:
                    count += 1
                    if count >= min_count:
                        break
            if count < min_count:
                min_count = count
                target_sub = s_idx
                if count == 0:
                    return False
                if count == 1:
                    break
                    
        if target_sub == -1:
            return False

        target_candidates = []
        for b_idx in blocks_covering_4sub_11[target_sub]:
            if b_idx in candidates:
                target_candidates.append(b_idx)
                
        for b_idx in target_candidates:
            new_mask = covered_mask | block_masks_11[b_idx]
            new_candidates = set()
            for c in candidates:
                if c == b_idx:
                    continue
                if (block_masks_11[c] & block_masks_11[b_idx]) == 0:
                    new_candidates.add(c)
            
            solution_11.append(b_idx)
            if search_11(new_mask, new_candidates):
                return True
            solution_11.pop()
            
        return False

    # Symmetry breaking: Fix first block {0,1,2,3,4}
    # Index of (0,1,2,3,4) in all_5subs_11
    first_block = (0, 1, 2, 3, 4)
    first_block_idx = all_5subs_11.index(first_block)
    
    initial_candidates_11 = set(range(num_5subs_11))
    # Remove first block from candidates as we fix it
    initial_candidates_11.remove(first_block_idx)
    
    # Filter candidates that conflict with first block
    filtered_candidates_11 = set()
    for c in initial_candidates_11:
        if (block_masks_11[c] & block_masks_11[first_block_idx]) == 0:
            filtered_candidates_11.add(c)
            
    solution_11.append(first_block_idx)
    search_11(block_masks_11[first_block_idx], filtered_candidates_11)
    
    if not found_11:
        # Fallback if search fails (should not happen)
        return []

    w11_blocks = [all_5subs_11[i] for i in solution_11]
    
    # Step 2: Extend to S(5,6,12)
    # Points: 0..11
    # Blocks containing 11: {11} U B for B in w11_blocks
    # Blocks not containing 11: {0..10} \ B for B in w11_blocks
    
    w12_blocks = []
    points_11 = set(range(11))
    
    for b in w11_blocks:
        # Block with 11
        b_with_11 = tuple(sorted(list(b) + [11]))
        w12_blocks.append(b_with_11)
        
        # Block without 11 (complement in 0..10)
        b_set = set(b)
        complement = tuple(sorted(points_11 - b_set))
        w12_blocks.append(complement)
        
    # Verify solution
    # Check distinctness
    if len(set(w12_blocks)) != 132:
        return []
        
    # Check no repeated 5-subset
    seen_5subs = set()
    for b in w12_blocks:
        for sub in combinations(b, 5):
            if sub in seen_5subs:
                return []
            seen_5subs.add(sub)
            
    return [list(b) for b in w12_blocks]

if __name__ == "__main__":
    blocks = construct()
    print(f"Number of blocks: {len(blocks)}")
