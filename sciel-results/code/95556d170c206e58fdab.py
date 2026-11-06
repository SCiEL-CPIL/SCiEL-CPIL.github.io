import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    points = range(12)
    all_5_subsets = list(combinations(points, 5))
    index_5 = {s: i for i, s in enumerate(all_5_subsets)}
    
    all_6_subsets = list(combinations(points, 6))
    num_6 = len(all_6_subsets)
    block_5_indices = [None] * num_6
    for j, b in enumerate(all_6_subsets):
        inds = [index_5[s] for s in combinations(b, 5)]
        block_5_indices[j] = inds

    subsets_5_to_6 = [[] for _ in range(792)]
    for i, s5 in enumerate(all_5_subsets):
        s5_set = set(s5)
        for j, b in enumerate(all_6_subsets):
            if s5_set.issubset(set(b)):
                subsets_5_to_6[i].append(j)

    first_block = (0, 1, 2, 3, 4, 5)
    first_block_idx = all_6_subsets.index(first_block)
    
    blocks = [first_block]
    blocks_idx = [first_block_idx]
    
    # Use a bytearray for covered 5-subsets for faster access
    covered = bytearray(792)
    for idx in block_5_indices[first_block_idx]:
        covered[idx] = 1
        
    def backtrack():
        if len(blocks) == 132:
            return True
        
        if time.time() > deadline:
            return False
            
        # MRV Heuristic
        min_candidates = float('inf')
        target_idx = -1
        valid_cands = []
        
        # Shuffle the order of checking 5-subsets to randomize tie-breaking?
        # Or just shuffle valid_cands when selecting?
        # Shuffling 792 items every node is expensive.
        # Better: If multiple 5-subsets have the same min_candidates, pick one randomly.
        
        # We need to find the min.
        # To handle ties randomly, we can collect all indices with min_candidates?
        # That requires two passes or storing a list.
        # Given 792 is small, we can do:
        
        candidates_for_min = []
        
        for i in range(792):
            if covered[i] == 0:
                cands = subsets_5_to_6[i]
                valid = []
                for j in cands:
                    b_5s = block_5_indices[j]
                    # Check if any 5-subset of block j is covered
                    conflict = False
                    for c in b_5s:
                        if covered[c]:
                            conflict = True
                            break
                    if not conflict:
                        valid.append(j)
                
                if len(valid) < min_candidates:
                    min_candidates = len(valid)
                    candidates_for_min = [i]
                    valid_cands = valid
                    if min_candidates == 0:
                        break
                elif len(valid) == min_candidates:
                    candidates_for_min.append(i)
                    # We don't need to update valid_cands yet, we'll pick from candidates_for_min
            
        if target_idx == -1 and not candidates_for_min:
            return False
            
        if min_candidates == 0:
            return False

        # Pick a random 5-subset from the tied minimums
        if candidates_for_min:
            target_idx = random.choice(candidates_for_min)
            # We need the valid_cands for this specific target_idx
            # We stored valid_cands for the FIRST one found with min.
            # If we picked a different one, we need to recompute valid_cands for it.
            # To avoid recomputation, we could store a dict {idx: valid_cands}.
            # Or just recompute for the chosen target_idx. It's cheap (7 checks).
            cands = subsets_5_to_6[target_idx]
            valid_cands = []
            for j in cands:
                b_5s = block_5_indices[j]
                conflict = False
                for c in b_5s:
                    if covered[c]:
                        conflict = True
                        break
                if not conflict:
                    valid_cands.append(j)
            
            # Shuffle valid_cands to randomize block selection order
            random.shuffle(valid_cands)

        for j in valid_cands:
            b = all_6_subsets[j]
            blocks.append(b)
            blocks_idx.append(j)
            b_5s = block_5_indices[j]
            for c in b_5s:
                covered[c] = 1
            
            if backtrack():
                return True
                
            # Undo
            blocks.pop()
            blocks_idx.pop()
            for c in b_5s:
                covered[c] = 0
            
        return False
        
    if backtrack():
        return [list(b) for b in blocks]
    else:
        return [list(b) for b in blocks]
