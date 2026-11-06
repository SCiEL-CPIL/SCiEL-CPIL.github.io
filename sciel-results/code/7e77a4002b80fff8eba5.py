import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    points = range(12)
    
    # Precompute all 5-subsets as bitmasks
    all_5_subsets = []
    for s in combinations(points, 5):
        mask = 0
        for p in s:
            mask |= (1 << p)
        all_5_subsets.append(mask)
    
    # Generate all 924 6-subsets
    all_6_subsets = []
    for b in combinations(points, 6):
        mask = 0
        for p in b:
            mask |= (1 << p)
        all_6_subsets.append(mask)
    
    # Precompute: for each 6-subset, its 5-subsets
    block_5s = {}
    for b_mask in all_6_subsets:
        pts = [i for i in range(12) if (b_mask >> i) & 1]
        masks = []
        for i in range(6):
            sub_mask = b_mask & ~(1 << pts[i])
            masks.append(sub_mask)
        block_5s[b_mask] = masks
    
    # Precompute: for each 5-subset, the list of 6-subsets that contain it
    blocks_by_5 = {mask: [] for mask in all_5_subsets}
    for b_mask in all_6_subsets:
        for s in block_5s[b_mask]:
            blocks_by_5[s].append(b_mask)
    
    # Backtracking with MRV
    solution = []
    covered = set()
    uncovered = set(all_5_subsets)
    
    def backtrack(depth):
        if time.time() > deadline:
            return False
            
        if depth == 132:
            return len(uncovered) == 0

        # MRV: find uncovered 5-subset with fewest available blocks
        best_5_mask = None
        best_candidates = None
        min_candidates = float('inf')
        
        for mask in uncovered:
            # Find available blocks: none of its 5-subsets already covered
            candidates = []
            for b_mask in blocks_by_5[mask]:
                if not any(s in covered for s in block_5s[b_mask]):
                    candidates.append(b_mask)
            
            if len(candidates) < min_candidates:
                min_candidates = len(candidates)
                best_5_mask = mask
                best_candidates = candidates
                
                if min_candidates == 0:
                    return False
                if min_candidates == 1:
                    break
        
        if best_candidates is None or len(best_candidates) == 0:
            return False
            
        for b_mask in best_candidates:
            solution.append(b_mask)
            new_covers = block_5s[b_mask]
            for s in new_covers:
                covered.add(s)
                uncovered.discard(s)
                
            if backtrack(depth + 1):
                return True
                
            solution.pop()
            for s in new_covers:
                covered.discard(s)
                uncovered.add(s)
                
        return False

    if backtrack(0):
        result = []
        for mask in solution:
            result.append([i for i in range(12) if (mask >> i) & 1])
        return result
    else:
        return solution
