import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 12.0  # 12s is plenty for S(4,5,11)
    
    points = list(range(1, 12))
    all_4_subsets = set(combinations(points, 4))
    
    # Precompute 4-subsets for each 5-subset
    candidates = list(combinations(points, 5))
    cand_4s = [set(combinations(c, 4)) for c in candidates]
    
    # Map 4-subset to indices of candidates containing it
    subset_to_cands = {s: [] for s in all_4_subsets}
    for i, c4s in enumerate(cand_4s):
        for s in c4s:
            subset_to_cands[s].append(i)
            
    # Symmetry breaking: Fix the first block
    # The block (1,2,3,4,5) is a valid choice? 
    # In S(4,5,11), any 5-set can be part of a solution?
    # Actually, just pick the first one.
    first_idx = 0
    first_block = candidates[first_idx]
    covered = set(cand_4s[first_idx])
    current_blocks = [first_block]
    uncovered = all_4_subsets - covered
    
    def backtrack(covered, current_blocks, uncovered):
        if time.time() > deadline:
            return None
        if len(current_blocks) == 66:
            return current_blocks
        if not uncovered:
            return current_blocks if len(current_blocks) == 66 else None
            
        # Heuristic: Most Constrained Variable (uncovered 4-subset with fewest candidates)
        min_count = float('inf')
        target_subset = None
        valid_cands = []
        
        for s in uncovered:
            cands = subset_to_cands[s]
            # Filter compatible candidates
            compat = [i for i in cands if not (cand_4s[i] & covered)]
            if not compat:
                return None # Dead end
            if len(compat) < min_count:
                min_count = len(compat)
                target_subset = s
                valid_cands = compat
                if min_count == 1:
                    break
        
        if target_subset is None:
            return None
            
        # Sort candidates by how many new subsets they cover (prefer more coverage)
        valid_cands.sort(key=lambda i: len(cand_4s[i] & uncovered), reverse=True)
        
        for idx in valid_cands:
            c = candidates[idx]
            new_covered = set(covered)
            for s in cand_4s[idx]:
                new_covered.add(s)
            new_uncovered = uncovered - new_covered
            
            # Forward checking: ensure no uncovered subset becomes unsatisfiable
            dead_end = False
            for s in new_uncovered:
                cands = subset_to_cands[s]
                compat = [i for i in cands if not (cand_4s[i] & new_covered)]
                if not compat:
                    dead_end = True
                    break
            if dead_end:
                continue
            
            result = backtrack(new_covered, current_blocks + [c], new_uncovered)
            if result is not None:
                return result
                
        return None

    sol_4511 = backtrack(covered, current_blocks, uncovered)
    
    if sol_4511 is None:
        # Fallback: Greedy with MRV
        covered = set()
        blocks = []
        remaining = list(range(len(candidates)))
        while len(blocks) < 66 and remaining:
            # Find the candidate that covers the most uncovered 4-subsets
            best_idx = None
            best_new_cover = -1
            for idx in remaining:
                if cand_4s[idx] & covered:
                    continue
                new_cover = len(cand_4s[idx] - covered)
                if new_cover > best_new_cover:
                    best_new_cover = new_cover
                    best_idx = idx
            
            if best_idx is None:
                break
            
            blocks.append(candidates[best_idx])
            covered.update(cand_4s[best_idx])
            remaining.remove(best_idx)
        sol_4511 = blocks

    final_blocks = []
    for b in sol_4511:
        final_blocks.append([0] + list(b))
        b_set = set(b)
        comp = [p for p in points if p not in b_set]
        final_blocks.append(comp)
        
    return final_blocks
