import time
import sys
from itertools import combinations

def construct():
    """
    Constructs the Steiner System S(5, 6, 12) by solving the exact cover problem
    where the items are the 5-subsets of {0..11} and the sets are the 6-subsets.
    """
    start_time = time.time()
    timeout = 100.0 # Safety margin below 150s
    
    points = range(12)
    all_5subsets = list(combinations(points, 5))
    num_5subsets = len(all_5subsets)
    
    # Map each 5-subset to an index for fast lookup
    subset_to_idx = {s: i for i, s in enumerate(all_5subsets)}
    
    # Generate all 6-subsets (blocks)
    all_6subsets = list(combinations(points, 6))
    num_blocks = len(all_6subsets)
    
    # For each block (6-subset), find the indices of the 5-subsets it contains
    # block_to_5subsets[block] = list of indices in all_5subsets
    block_to_5subsets = []
    for b in all_6subsets:
        subs = [subset_to_idx[s] for s in combinations(b, 5)]
        block_to_5subsets.append(subs)
        
    # For each 5-subset, find the list of blocks that contain it
    # This is the "candidates" for that 5-subset
    # We maintain a set of uncovered 5-subsets.
    
    # Optimization: Pre-compute candidate lists for each 5-subset
    # candidates[subset_idx] = list of block indices that cover this subset
    candidates = [[] for _ in range(num_5subsets)]
    for b_idx, subs in enumerate(block_to_5subsets):
        for s_idx in subs:
            candidates[s_idx].append(b_idx)

    # Pre-compute conflict masks.
    # conflict_mask[b_idx] is a bitmask where bit j is 1 if block j conflicts with b_idx 
    # (i.e., they share a 5-subset) OR if j == b_idx.
    # Using bitmasks allows O(1) update of valid blocks.
    conflict_mask = [0] * num_blocks
    
    for b_idx, subs in enumerate(block_to_5subsets):
        mask = 1 << b_idx # Include self
        for s_idx in subs:
            for other_b in candidates[s_idx]:
                mask |= (1 << other_b)
        conflict_mask[b_idx] = mask

    # State for backtracking:
    # uncovered: set of 5-subset indices still to be covered
    # valid_mask: integer bitmask of valid blocks
    # chosen_blocks: list of block indices chosen so far
    
    best_solution = []
    
    def backtrack(uncovered, valid_mask, current_blocks):
        nonlocal best_solution
        if time.time() - start_time > timeout:
            return False
            
        if not uncovered:
            best_solution = current_blocks[:]
            return True
            
        # Find the uncovered subset with the minimum number of valid candidate blocks
        min_subset = -1
        min_count = float('inf')
        
        # Iterate through uncovered subsets
        # Optimization: We can just iterate. 792 is small.
        for s_idx in uncovered:
            # Count how many valid blocks cover s_idx
            count = 0
            for b_idx in candidates[s_idx]:
                if (valid_mask >> b_idx) & 1:
                    count += 1
            if count < min_count:
                min_count = count
                min_subset = s_idx
            if min_count == 0:
                return False # Dead end
            
        # Try each valid block covering min_subset
        for b_idx in candidates[min_subset]:
            if not ((valid_mask >> b_idx) & 1):
                continue
                
            # Choose b_idx
            # New uncovered: uncovered minus the 5-subsets of b_idx
            # New valid_mask: valid_mask AND NOT conflict_mask[b_idx]
            
            new_uncovered = set(uncovered)
            for s_idx in block_to_5subsets[b_idx]:
                new_uncovered.discard(s_idx)
                
            new_valid_mask = valid_mask & ~conflict_mask[b_idx]
            
            if backtrack(new_uncovered, new_valid_mask, current_blocks + [b_idx]):
                return True
                
        return False

    initial_uncovered = set(range(num_5subsets))
    initial_valid_mask = (1 << num_blocks) - 1
    
    if backtrack(initial_uncovered, initial_valid_mask, []):
        # Convert block indices back to lists of points
        result = [list(all_6subsets[i]) for i in best_solution]
        return result
    else:
        # Fallback to greedy if backtracking fails or times out (unlikely)
        covered = set()
        blocks = []
        for b in all_6subsets:
            subsets = tuple(combinations(b, 5))
            if not any(s in covered for s in subsets):
                blocks.append(list(b))
                covered.update(subsets)
        return blocks
