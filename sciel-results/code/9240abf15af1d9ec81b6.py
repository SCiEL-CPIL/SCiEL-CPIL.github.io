import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    points = list(range(1, 12))
    
    # All 4-subsets of points (1..11)
    all_4_subsets = list(combinations(points, 4))
    subset_to_idx = {s: i for i, s in enumerate(all_4_subsets)}
    n_4 = len(all_4_subsets)  # 330
    
    # All 5-subsets of points (candidates for S(4,5,11) blocks)
    candidates = list(combinations(points, 5))
    n_cands = len(candidates)  # 462
    
    # Precompute: for each candidate (5-subset), which 4-subset indices it covers
    cand_4_indices = []
    # For each 4-subset index, which candidate indices contain it
    subset_to_cands = [[] for _ in range(n_4)]
    
    for i, c in enumerate(candidates):
        indices = []
        for s in combinations(c, 4):
            idx = subset_to_idx[s]
            indices.append(idx)
            subset_to_cands[idx].append(i)
        cand_4_indices.append(indices)
    
    # Symmetry breaking: Fix the first block as (1, 2, 3, 4, 5)
    first_block = (1, 2, 3, 4, 5)
    first_idx = candidates.index(first_block)
    
    # Initial state
    covered_4 = set(cand_4_indices[first_idx])
    blocks_4511 = [first_block]
    
    all_indices = set(range(n_4))
    uncovered = all_indices - covered_4
    
    def backtrack(covered, current_blocks, uncovered):
        if time.time() > deadline:
            return None
        if len(current_blocks) == 66:
            return current_blocks
        if not uncovered:
            return None
        
        # Heuristic: Pick the uncovered 4-subset with the fewest valid (compatible) candidates
        min_count = float('inf')
        target_idx = -1
        valid_cands_for_target = []
        
        for u_idx in uncovered:
            cands = subset_to_cands[u_idx]
            valid_cands = []
            for c_idx in cands:
                # Check if candidate c_idx is compatible with current covered set
                # i.e., none of its 4-subsets are already covered
                compatible = True
                for s_idx in cand_4_indices[c_idx]:
                    if s_idx in covered:
                        compatible = False
                        break
                if compatible:
                    valid_cands.append(c_idx)
            
            if len(valid_cands) < min_count:
                min_count = len(valid_cands)
                target_idx = u_idx
                valid_cands_for_target = valid_cands
                if min_count == 0:
                    break
            
            if min_count == 1:
                break  # Can't do better than 1
        
        if min_count == 0:
            return None
        
        # Try each valid candidate for the most constrained uncovered subset
        for c_idx in valid_cands_for_target:
            c = candidates[c_idx]
            new_covered = set(covered)
            for s_idx in cand_4_indices[c_idx]:
                new_covered.add(s_idx)
            new_uncovered = uncovered - new_covered
            
            result = backtrack(new_covered, current_blocks + [c], new_uncovered)
            if result is not None:
                return result
        
        return None
    
    sol_4511 = backtrack(covered_4, blocks_4511, uncovered)
    
    if sol_4511 is None:
        # Fallback: greedy approach
        covered_4 = set()
        blocks_4511 = []
        for b in candidates:
            subsets = list(combinations(b, 4))
            if not any(subset_to_idx[s] in covered_4 for s in subsets):
                blocks_4511.append(b)
                for s in subsets:
                    covered_4.add(subset_to_idx[s])
                if len(blocks_4511) == 66:
                    break
        sol_4511 = blocks_4511
    
    # Step 2: Construct S(5,6,12) from S(4,5,11)
    final_blocks = []
    
    # Blocks containing 0
    for b in sol_4511:
        final_blocks.append([0] + list(b))
        
    # Blocks not containing 0 (complements in 1..11)
    for b in sol_4511:
        b_set = set(b)
        comp = [p for p in points if p not in b_set]
        final_blocks.append(comp)
        
    return final_blocks
