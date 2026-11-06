import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    points = range(12)
    
    # 1. Generate all 5-subsets and map them to indices
    all_5_subsets = list(combinations(points, 5))
    index_5 = {s: i for i, s in enumerate(all_5_subsets)}
    num_5 = len(all_5_subsets) # 792
    
    # 2. Generate all 6-subsets (blocks)
    all_6_subsets = list(combinations(points, 6))
    num_6 = len(all_6_subsets) # 924
    
    # 3. Pre-compute bitmask for each block
    block_masks = [0] * num_6
    for j, b in enumerate(all_6_subsets):
        mask = 0
        for s5 in combinations(b, 5):
            idx = index_5[s5]
            mask |= (1 << idx)
        block_masks[j] = mask
        
    # 4. Pre-compute which blocks cover each 5-subset
    blocks_by_5 = [[] for _ in range(num_5)]
    for j, b in enumerate(all_6_subsets):
        for s5 in combinations(b, 5):
            idx = index_5[s5]
            blocks_by_5[idx].append(j)
            
    covered_mask = 0
    blocks = []
    
    def backtrack():
        if len(blocks) == 132:
            return True
            
        if time.time() > deadline:
            return False
            
        # Find an uncovered 5-subset
        # Try random indices first for diversity
        target_idx = -1
        # 12 points, 792 5-subsets. 
        # As we fill up, uncovered ones become rare.
        # But at start, most are uncovered.
        for _ in range(5):
            idx = random.randint(0, num_5 - 1)
            if not (covered_mask & (1 << idx)):
                target_idx = idx
                break
                
        if target_idx == -1:
            # Fallback: linear scan
            for i in range(num_5):
                if not (covered_mask & (1 << i)):
                    target_idx = i
                    break
                    
        if target_idx == -1:
            return len(blocks) == 132

        # Get candidate blocks
        cands = blocks_by_5[target_idx]
        valid_cands = []
        for j in cands:
            if (block_masks[j] & covered_mask) == 0:
                valid_cands.append(j)
                
        if not valid_cands:
            return False
            
        random.shuffle(valid_cands)
        
        for j in valid_cands:
            blocks.append(j)
            covered_mask_local = covered_mask | block_masks[j]
            
            # Recurse with new mask
            # Note: Python passes integers by value, so we don't need to undo the mask
            # if we pass it as an argument, but here we use closure variable.
            # So we must undo.
            old_mask = covered_mask
            covered_mask = covered_mask_local
            
            if backtrack():
                return True
                
            # Undo
            blocks.pop()
            covered_mask = old_mask
            
        return False
        
    # Run the solver
    # To ensure robustness, we can try a few times if it fails
    # But it should be very fast.
    
    success = False
    for _ in range(20):
        covered_mask = 0
        blocks = []
        if backtrack():
            success = True
            break
            
    if success:
        return [list(all_6_subsets[j]) for j in blocks]
    else:
        return [list(all_6_subsets[j]) for j in blocks]
