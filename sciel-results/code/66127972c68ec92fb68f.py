import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    points = range(12)
    # Map 5-subsets to unique indices (0 to 791)
    # Total 5-subsets is C(12,5) = 792
    all_5_subsets = list(combinations(points, 5))
    index_5 = {s: i for i, s in enumerate(all_5_subsets)}
    
    # Precompute for each 6-subset, the list of indices of its 5-subsets
    all_6_subsets = list(combinations(points, 6))
    num_6 = len(all_6_subsets)
    block_5_indices = [None] * num_6
    block_sets = [None] * num_6
    for j, b in enumerate(all_6_subsets):
        inds = [index_5[s] for s in combinations(b, 5)]
        block_5_indices[j] = inds
        block_sets[j] = set(b)

    # Precompute for each 5-subset, the list of 6-subset indices that contain it
    # There are exactly 7 such 6-subsets for each 5-subset
    subsets_5_to_6 = [[] for _ in range(792)]
    for i, s5 in enumerate(all_5_subsets):
        s5_set = set(s5)
        for j, b in enumerate(all_6_subsets):
            if s5_set.issubset(block_sets[j]):
                subsets_5_to_6[i].append(j)

    # Symmetry breaking: First block is (0, 1, 2, 3, 4, 5)
    first_block = (0, 1, 2, 3, 4, 5)
    # Find the index of the first block in all_6_subsets
    first_block_idx = all_6_subsets.index(first_block)
    
    blocks = [first_block]
    blocks_idx = [first_block_idx]
    
    # Use a boolean array for covered 5-subsets (faster than set)
    covered = [False] * 792
    for c in block_5_indices[first_block_idx]:
        covered[c] = True
    
    def backtrack():
        if len(blocks) == 132:
            return True
        
        if time.time() > deadline:
            return False
            
        # MRV Heuristic: Find the uncovered 5-subset with the fewest valid candidates
        min_candidates = float('inf')
        target_idx = -1
        valid_cands = []
        
        for i in range(792):
            if covered[i]:
                continue
            cands = subsets_5_to_6[i]
            # Filter valid candidates
            valid = []
            for j in cands:
                b_5s = block_5_indices[j]
                ok = True
                for c in b_5s:
                    if covered[c]:
                        ok = False
                        break
                if ok:
                    valid.append(j)
            
            if len(valid) < min_candidates:
                min_candidates = len(valid)
                target_idx = i
                valid_cands = valid
                if min_candidates == 0:
                    break

        if target_idx == -1:
            return False
            
        if min_candidates == 0:
            return False

        for j in valid_cands:
            b = all_6_subsets[j]
            blocks.append(b)
            blocks_idx.append(j)
            b_5s = block_5_indices[j]
            
            # Track newly covered items for undo
            newly_covered = []
            for c in b_5s:
                if not covered[c]:
                    newly_covered.append(c)
                covered[c] = True
            
            if backtrack():
                return True
                
            # Undo
            blocks.pop()
            blocks_idx.pop()
            for c in newly_covered:
                covered[c] = False
            
        return False
        if len(blocks) == 132:
            return True
        
        if time.time() > deadline:
            return False
            
        if not uncovered:
            return False
        
        # MRV Heuristic: Find the uncovered 5-subset with the fewest valid candidates
        min_candidates = float('inf')
        target_idx = -1
        valid_cands = []
        
        for i in uncovered:
            cands = subsets_5_to_6[i]
            # Filter valid candidates
            valid = []
            for j in cands:
                b_5s = block_5_indices[j]
                ok = True
                for c in b_5s:
                    if covered[c]:
                        ok = False
                        break
                if ok:
                    valid.append(j)
            
            if len(valid) < min_candidates:
                min_candidates = len(valid)
                target_idx = i
                valid_cands = valid
                if min_candidates == 0:
                    break

        if target_idx == -1:
            return False
            
        if min_candidates == 0:
            return False

        for j in valid_cands:
            b = all_6_subsets[j]
            blocks.append(b)
            blocks_idx.append(j)
            b_5s = block_5_indices[j]
            for c in b_5s:
                covered[c] = True
            
            if backtrack():
                return True
                
            # Undo
            blocks.pop()
            blocks_idx.pop()
            for c in b_5s:
                covered[c] = False
            
        return False
        
    if backtrack():
        return [list(b) for b in blocks]
    else:
        return [list(b) for b in blocks]
