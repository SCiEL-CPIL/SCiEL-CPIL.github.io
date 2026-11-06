import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Points are 0..11
    points = range(12)
    
    # Generate all possible 6-subsets
    all_blocks = list(combinations(points, 6))
    
    # Pre-compute 5-subsets for each block
    # A block is valid if none of its 5-subsets are in 'covered'
    # We need to maximize the count. Max is 132.
    
    # To speed up search, we can use a greedy initialization or just backtrack.
    # Given the small size (924 candidates), backtracking with pruning should be very fast.
    
    covered = set()
    current_blocks = []
    candidates = list(range(len(all_blocks)))
    
    # Optimization: Sort candidates to find a solution quickly?
    # Lexicographic order is fine.
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        
        if len(current_blocks) == 132:
            return True
            
        # Pruning: If we can't reach 132, return False
        # Max possible additional blocks: len(candidates[idx:])
        if len(current_blocks) + len(candidates[idx:]) < 132:
            return False

        for i in range(idx, len(candidates)):
            b_idx = candidates[i]
            b = all_blocks[b_idx]
            
            # Check if this block conflicts with covered 5-subsets
            conflict = False
            for sub in combinations(b, 5):
                if sub in covered:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            # Add block
            current_blocks.append(b)
            for sub in combinations(b, 5):
                covered.add(sub)
                
            # Recurse
            if backtrack(i + 1):
                return True
                
            # Undo
            current_blocks.pop()
            for sub in combinations(b, 5):
                covered.remove(sub)
                
        return False

    # Start backtracking
    # To help the solver, we can try to force the first block to be (0,1,2,3,4,5) due to symmetry
    # This reduces the search space significantly.
    
    # Fix first block to (0,1,2,3,4,5)
    first_block = (0, 1, 2, 3, 4, 5)
    current_blocks.append(first_block)
    for sub in combinations(first_block, 5):
        covered.add(sub)
        
    # Filter candidates to exclude those conflicting with the first block
    # And exclude the first block itself
    filtered_candidates = []
    for i, b in enumerate(all_blocks):
        if b == first_block:
            continue
        conflict = False
        for sub in combinations(b, 5):
            if sub in covered:
                conflict = True
                break
        if not conflict:
            filtered_candidates.append(i)
            
    # Now backtrack on the filtered candidates
    # We need to find 131 more blocks.
    
    def backtrack_filtered(idx, target):
        if time.time() > deadline:
            return False
        
        if len(current_blocks) == target:
            return True
            
        # Pruning
        if len(current_blocks) + len(filtered_candidates[idx:]) < target:
            return False

        for i in range(idx, len(filtered_candidates)):
            b_idx = filtered_candidates[i]
            b = all_blocks[b_idx]
            
            conflict = False
            for sub in combinations(b, 5):
                if sub in covered:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            # Add block
            current_blocks.append(b)
            for sub in combinations(b, 5):
                covered.add(sub)
                
            if backtrack_filtered(i + 1, target):
                return True
                
            # Undo
            current_blocks.pop()
            for sub in combinations(b, 5):
                covered.remove(sub)
                
        return False

    success = backtrack_filtered(0, 132)
    
    if success:
        return [list(b) for b in current_blocks]
    else:
        # Fallback: return whatever we have (should be 132 if logic is correct)
        # If it fails, it might be due to time, but 132 is the unique optimum.
        # If we didn't find 132, we return the best we have.
        return [list(b) for b in current_blocks]

# EVOLVE-BLOCK-END
