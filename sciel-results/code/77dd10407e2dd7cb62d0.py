import time
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    points = range(12)
    # Map 5-subsets to unique indices (0 to 791)
    all_5_subsets = list(combinations(points, 5))
    index_5 = {s: i for i, s in enumerate(all_5_subsets)}
    
    # Precompute for each 6-subset, the list of indices of its 5-subsets
    all_6_subsets = list(combinations(points, 6))
    block_5_indices = {}
    for b in all_6_subsets:
        inds = [index_5[s] for s in combinations(b, 5)]
        block_5_indices[b] = inds
    
    # Precompute: for each 5-subset index, which 6-subsets contain it
    # This avoids scanning all 924 6-subsets at each step
    contains_5 = [[] for _ in range(792)]
    for b_idx, b in enumerate(all_6_subsets):
        for idx in block_5_indices[b]:
            contains_5[idx].append(b_idx)
    
    # Convert block_5_indices to bitmask representation
    # Each 5-subset index maps to a bit position in a Python int
    block_masks = {}
    for b in all_6_subsets:
        mask = 0
        for idx in block_5_indices[b]:
            mask |= (1 << idx)
        block_masks[b] = mask
    
    def backtrack(blocks, covered_mask, depth):
        if depth == 132:
            return True
        
        if time.time() > deadline:
            return False
            
        # Find the uncovered 5-subset with fewest available candidates (MRV)
        # First, find all uncovered 5-subsets
        best_target = -1
        best_count = 999
        best_candidates = None
        
        # We need to find uncovered 5-subsets. Since we have a bitmask,
        # we can iterate through bits. But 792 bits is a lot to scan.
        # Alternative: maintain a count of how many blocks could still cover each 5-subset
        # For now, let's use a simpler approach: find first uncovered, then check candidates
        
        # Actually, let's try a different approach: 
        # Find uncovered 5-subsets by checking which bits are 0 in covered_mask
        # But iterating 792 positions each time is O(792) which is fine
        
        target_idx = -1
        for i in range(792):
            if not (covered_mask >> i) & 1:
                target_idx = i
                break
                
        if target_idx == -1:
            return False
            
        # Get candidates for this 5-subset
        candidates = contains_5[target_idx]
        
        # Filter to valid candidates (no overlap with covered)
        valid_candidates = []
        for b_idx in candidates:
            b = all_6_subsets[b_idx]
            if block_masks[b] & covered_mask == 0:
                valid_candidates.append(b_idx)
                
        if not valid_candidates:
            return False
        
        # Shuffle for randomness to help escape local optima
        random.shuffle(valid_candidates)
        
        for b_idx in valid_candidates:
            b = all_6_subsets[b_idx]
            blocks.append(b)
            new_mask = covered_mask | block_masks[b]
            
            if backtrack(blocks, new_mask, depth + 1):
                return True
                
            blocks.pop()
            
        return False
    
    best_blocks = []
    
    # Try multiple restarts with different random seeds
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        
        random.seed(attempt * 1000 + 42)
        
        # Symmetry breaking: First block is (0, 1, 2, 3, 4, 5)
        first_block = (0, 1, 2, 3, 4, 5)
        blocks = [first_block]
        covered_mask = block_masks[first_block]
        
        if backtrack(blocks, covered_mask, 1):
            if len(blocks) > len(best_blocks):
                best_blocks = list(blocks)
            if len(blocks) == 132:
                break
    
    if not best_blocks:
        # Fallback: just return the first block
        best_blocks = [(0, 1, 2, 3, 4, 5)]
    
    return [list(b) for b in best_blocks]
