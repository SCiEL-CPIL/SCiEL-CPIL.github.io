import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    points = range(12)
    all_5_subsets = list(combinations(points, 5))
    index_5 = {s: i for i, s in enumerate(all_5_subsets)}
    num_5 = len(all_5_subsets)

    all_6_subsets = list(combinations(points, 6))
    num_6 = len(all_6_subsets)

    # Precompute: for each 6-subset, which 5-subsets it covers (as indices)
    block_5_indices = []
    for b in all_6_subsets:
        block_5_indices.append([index_5[s5] for s5 in combinations(b, 5)])

    # Precompute bitmask for each 6-subset's covered 5-subsets
    # 792 bits fits in a Python int
    block_masks = []
    for j in range(num_6):
        mask = 0
        for idx in block_5_indices[j]:
            mask |= (1 << idx)
        block_masks.append(mask)

    # Precompute: for each 5-subset, which 6-subsets contain it
    subsets_5_to_6 = [[] for _ in range(num_5)]
    for j in range(num_6):
        for idx in block_5_indices[j]:
            subsets_5_to_6[idx].append(j)

    best_result = []

    # Symmetry Breaking:
    # 1. The first block must contain {0,1,2,3,4}.
    #    The block is {0,1,2,3,4, x} for x in 5..11.
    
    for x in range(5, 12):
        if time.time() > deadline - 5:
            break
            
        # B1 = {0, 1, 2, 3, 4, x}
        b1 = tuple(sorted([0, 1, 2, 3, 4, x]))
        b1_idx = all_6_subsets.index(b1)
        
        blocks = [b1]
        covered_mask = block_masks[b1_idx]
        num_covered = 6  # Each block covers exactly 6 5-subsets
        
        def backtrack(covered_mask, num_covered, blocks_list):
            if len(blocks_list) == 132:
                return True
            
            if time.time() > deadline - 5:
                return False
            
            # Find an uncovered 5-subset with MRV heuristic
            min_cands = float('inf')
            target_5_idx = -1
            valid_cands = []
            
            # Scan all 5-subsets to find uncovered ones
            for i in range(num_5):
                if (covered_mask >> i) & 1:
                    continue  # Already covered
                
                cands = subsets_5_to_6[i]
                # Count valid candidates (blocks that don't conflict)
                valid = []
                for j in cands:
                    if not (block_masks[j] & covered_mask):
                        valid.append(j)
                
                if len(valid) < min_cands:
                    min_cands = len(valid)
                    target_5_idx = i
                    valid_cands = valid
                    if min_cands == 0:
                        return False
            
            if target_5_idx == -1:
                return False # All covered? Shouldn't happen before 132 blocks.
                
            # Try candidates
            for j in valid_cands:
                b = all_6_subsets[j]
                blocks_list.append(b)
                new_mask = covered_mask | block_masks[j]
                
                if backtrack(new_mask, num_covered + 6, blocks_list):
                    return True
                
                blocks_list.pop()
            
            return False

        if backtrack(covered_mask, num_covered, blocks):
            return [list(b) for b in blocks]

    # Fallback: Random restarts if symmetry breaking fails
    import random
    for restart in range(10):
        if time.time() > deadline - 5:
            break
        rng = random.Random(restart)
        # Random start
        first_block_idx = rng.randint(0, num_6 - 1)
        first_block = all_6_subsets[first_block_idx]
        blocks = [first_block]
        covered_mask = block_masks[first_block_idx]
        
        def backtrack2(covered_mask, blocks_list):
            if len(blocks_list) == 132:
                return True
            if time.time() > deadline - 5:
                return False
            
            min_cands = float('inf')
            target_5_idx = -1
            valid_cands = []
            
            for i in range(num_5):
                if (covered_mask >> i) & 1:
                    continue
                cands = subsets_5_to_6[i]
                valid = [j for j in cands if not (block_masks[j] & covered_mask)]
                if len(valid) < min_cands:
                    min_cands = len(valid)
                    target_5_idx = i
                    valid_cands = valid
                    if min_cands == 0:
                        return False
            
            if target_5_idx == -1:
                return False
                
            rng.shuffle(valid_cands)
            for j in valid_cands:
                b = all_6_subsets[j]
                blocks_list.append(b)
                new_mask = covered_mask | block_masks[j]
                
                if backtrack2(new_mask, blocks_list):
                    return True
                blocks_list.pop()
            return False

        if backtrack2(covered_mask, blocks):
            return [list(b) for b in blocks]

    return []
