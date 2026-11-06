import sys
from itertools import combinations

sys.setrecursionlimit(3000)

def construct():
    # Step 1: Construct S(4,5,11) on points 0..10
    n11 = 11
    points11 = list(range(n11))
    
    # All 4-subsets
    fours = list(combinations(points11, 4))
    four_to_idx = {f: i for i, f in enumerate(fours)}
    num_fours = len(fours) # 330
    
    # All 5-subsets
    fives11 = list(combinations(points11, 5))
    five_masks11 = []
    for s in fives11:
        mask = 0
        for f in combinations(s, 4):
            idx = four_to_idx[f]
            mask |= (1 << idx)
        five_masks11.append(mask)
        
    num_fives11 = len(fives11) # 462
    
    # covers4[idx_4] = list of indices of 5-sets containing this 4-set
    covers4 = [[] for _ in range(num_fours)]
    for i, mask in enumerate(five_masks11):
        s = fives11[i]
        for f in combinations(s, 4):
            idx_f = four_to_idx[f]
            covers4[idx_f].append(i)
            
    solution_11 = []
    
    def backtrack_11(covered_mask, available_blocks, depth):
        if depth == 66:
            if covered_mask == (1 << num_fours) - 1:
                return True
            return False
            
        # Find first uncovered 4-subset
        target_idx = -1
        for i in range(num_fours):
            if not (covered_mask & (1 << i)):
                target_idx = i
                break
                
        if target_idx == -1:
            return False
            
        candidates = []
        for block_idx in covers4[target_idx]:
            if block_idx in available_blocks:
                mask = five_masks11[block_idx]
                if (mask & covered_mask) == 0:
                    candidates.append(block_idx)
                    
        if not candidates:
            return False
            
        for block_idx in candidates:
            mask = five_masks11[block_idx]
            new_covered = covered_mask | mask
            
            new_available = []
            for b_idx in available_blocks:
                if b_idx == block_idx:
                    continue
                if (five_masks11[b_idx] & mask) != 0:
                    continue
                new_available.append(b_idx)
                
            solution_11.append(fives11[block_idx])
            
            if backtrack_11(new_covered, new_available, depth + 1):
                return True
                
            solution_11.pop()
            
        return False

    # Symmetry breaking: The first block must contain 0.
    # We can filter initial candidates to only those containing 0.
    # Actually, to be safe and fast, we just run it. 11 points is small.
    # But to avoid isomorphic duplicates in search tree, we can fix the first block.
    # The lexicographically first block in any S(4,5,11) is {0,1,2,3,4}?
    # Not necessarily. But we can force the first chosen block to be the smallest index
    # that contains 0.
    
    # Let's just run the standard backtrack. It should be fast enough.
    initial_available = list(range(num_fives11))
    
    # Optimization: Force the first block to be the one with smallest index containing 0?
    # No, just run.
    
    if not backtrack_11(0, initial_available, 0):
        # Fallback: Return empty or raise
        return []

    # Step 2: Extend S(4,5,11) to S(5,6,12)
    # Points are 0..11.
    # Blocks are:
    # 1. {11} U B for B in solution_11
    # 2. {0..10} \ B for B in solution_11
    
    final_blocks = []
    for B in solution_11:
        # B is a tuple of 5 elements from 0..10
        block1 = tuple(sorted([11] + list(B)))
        block2 = tuple(sorted(set(range(11)) - set(B)))
        final_blocks.append(block1)
        final_blocks.append(block2)
        
    return final_blocks
