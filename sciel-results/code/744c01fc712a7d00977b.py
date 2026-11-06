import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    points = range(12)
    all_5_subsets = list(combinations(points, 5))
    index_5 = {s: i for i, s in enumerate(all_5_subsets)}
    
    all_6_subsets = list(combinations(points, 6))
    num_6 = len(all_6_subsets)
    
    # Precompute block masks
    # 792 bits -> 13 64-bit ints (or just Python ints)
    # Let's use a list of 13 ints, or just one big int?
    # Python big ints are efficient. Let's use a single Python int for the mask.
    # 792 bits is small for Python int.
    
    block_masks = [0] * num_6
    block_5_indices = [None] * num_6 # Still useful for tracking newly covered?
    
    for j, b in enumerate(all_6_subsets):
        mask = 0
        inds = []
        for s in combinations(b, 5):
            idx = index_5[s]
            mask |= (1 << idx)
            inds.append(idx)
        block_masks[j] = mask
        block_5_indices[j] = inds

    # Precompute for each 5-subset, the list of 6-subset indices
    subsets_5_to_6 = [[] for _ in range(792)]
    for i in range(792):
        # We need to find which blocks contain this 5-subset
        # A 5-subset is contained in a 6-subset if the 6-subset is a superset.
        # There are 12-5=7 points not in the 5-subset. So 7 such 6-subsets.
        s5 = all_5_subsets[i]
        s5_set = set(s5)
        for j, b in enumerate(all_6_subsets):
            if s5_set.issubset(set(b)):
                subsets_5_to_6[i].append(j)

    # Symmetry breaking
    first_block = (0, 1, 2, 3, 4, 5)
    first_block_idx = all_6_subsets.index(first_block)
    
    blocks = [first_block]
    covered_mask = block_masks[first_block_idx]
    uncovered = [i for i in range(792) if not (covered_mask >> i) & 1]
    
    def backtrack():
        if len(blocks) == 132:
            return True
        
        if time.time() > deadline:
            return False
            
        if not uncovered:
            return False # Should not happen before 132 blocks if valid, but safety
        
        # MRV Heuristic
        min_candidates = float('inf')
        target_idx = -1
        valid_cands = []
        
        for i in uncovered:
            cands = subsets_5_to_6[i]
            valid = []
            for j in cands:
                if (covered_mask & block_masks[j]) == 0:
                    valid.append(j)
            
            if len(valid) < min_candidates:
                min_candidates = len(valid)
                target_idx = i
                valid_cands = valid
                if min_candidates == 0:
                    break

        if target_idx == -1 or min_candidates == 0:
            return False

        for j in valid_cands:
            blocks.append(all_6_subsets[j])
            
            # Update covered mask and uncovered list
            new_mask = block_masks[j]
            new_covered_mask = covered_mask | new_mask
            newly_covered = block_5_indices[j]
            
            # Filter uncovered
            # Optimization: only remove those that are in newly_covered
            # Since we know exactly which ones are added, we can rebuild or filter
            # Rebuilding 792 items is fast enough?
            # Or maintain a set?
            # Let's just filter.
            new_uncovered = [k for k in uncovered if not (new_mask >> k) & 1]
            
            if backtrack(new_covered_mask, new_uncovered):
                return True
                
            blocks.pop()
            
        return False

    # Wrapper to pass state
    def bt_wrapper(covered_mask, uncovered):
        if len(blocks) == 132:
            return True
        
        if time.time() > deadline:
            return False
            
        if not uncovered:
            return False
        
        min_candidates = float('inf')
        target_idx = -1
        valid_cands = []
        
        for i in uncovered:
            cands = subsets_5_to_6[i]
            valid = []
            for j in cands:
                if (covered_mask & block_masks[j]) == 0:
                    valid.append(j)
            
            if len(valid) < min_candidates:
                min_candidates = len(valid)
                target_idx = i
                valid_cands = valid
                if min_candidates == 0:
                    break

        if target_idx == -1 or min_candidates == 0:
            return False

        for j in valid_cands:
            blocks.append(all_6_subsets[j])
            
            new_mask = block_masks[j]
            new_covered_mask = covered_mask | new_mask
            new_uncovered = [k for k in uncovered if not (new_mask >> k) & 1]
            
            if bt_wrapper(new_covered_mask, new_uncovered):
                return True
                
            blocks.pop()
            
        return False

    if bt_wrapper(covered_mask, uncovered):
        return [list(b) for b in blocks]
    else:
        return [list(b) for b in blocks]
