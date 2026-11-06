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
    block_5_indices = {}
    for b in all_6_subsets:
        inds = [index_5[s] for s in combinations(b, 5)]
        block_5_indices[b] = inds
    
    # Precompute for each 5-subset index, which 6-subsets contain it
    # This is much faster than checking all 924 6-subsets each time
    contains_5 = [[] for _ in range(792)]
    for b_idx, b in enumerate(all_6_subsets):
        for s in combinations(b, 5):
            contains_5[index_5[s]].append(b_idx)
    
    # Symmetry breaking: First block is (0, 1, 2, 3, 4, 5)
    first_block = (0, 1, 2, 3, 4, 5)
    blocks = [first_block]
    
    # Use a list of booleans for covered 5-subsets (faster than set for this size)
    covered = [False] * 792
    for idx in block_5_indices[first_block]:
        covered[idx] = True
    
    # Count uncovered for MRV heuristic
    uncovered_count = 792 - len(block_5_indices[first_block])
    
    def backtrack():
        nonlocal uncovered_count
        
        if len(blocks) == 132:
            return True
        
        if time.time() > deadline:
            return False
            
        # Find the first uncovered 5-subset using MRV-like approach
        # We want to pick an uncovered 5-subset that has fewest valid candidate 6-subsets
        # But for speed, we first try the simplest: first uncovered
        target_idx = -1
        for i in range(792):
            if not covered[i]:
                target_idx = i
                break
                
        if target_idx == -1:
            return False
            
        # Get candidates: 6-subsets that contain target_5 and don't conflict
        target_candidates = contains_5[target_idx]
        candidates = []
        
        for b_idx in target_candidates:
            b_5s = block_5_indices[all_6_subsets[b_idx]]
            # Check if any 5-subset in this block is already covered
            conflict = False
            for idx in b_5s:
                if covered[idx]:
                    conflict = True
                    break
            if not conflict:
                candidates.append(b_idx)
        
        # Sort candidates by how many new 5-subsets they cover (prefer those covering more)
        # This helps prune the search tree faster
        def sort_key(b_idx):
            b_5s = block_5_indices[all_6_subsets[b_idx]]
            new_count = 0
            for idx in b_5s:
                if not covered[idx]:
                    new_count += 1
            return -new_count  # Descending order
        
        candidates.sort(key=sort_key)
        
        for b_idx in candidates:
            b = all_6_subsets[b_idx]
            blocks.append(b)
            b_5s = block_5_indices[b]
            for idx in b_5s:
                covered[idx] = True
            
            if backtrack():
                return True
                
            # Undo
            blocks.pop()
            for idx in b_5s:
                covered[idx] = False
            
        return False
        
    if backtrack():
        return [list(b) for b in blocks]
    else:
        return [list(b) for b in blocks]
