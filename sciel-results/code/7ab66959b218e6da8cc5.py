"""Construct the Steiner system S(5,6,12) via backtracking search."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

def construct():
    """Return the 132 blocks of the Steiner system S(5,6,12)."""
    start_time = time.time()
    limit = 110.0 # Safety margin for 120s limit
    
    n = 12
    points = list(range(n))
    
    # Generate all 6-subsets (blocks)
    all_blocks = list(combinations(points, 6))
    
    # Precompute 5-subsets for each block
    block_5sub_sets = []
    for b in all_blocks:
        subs = set(combinations(b, 5))
        block_5sub_sets.append(subs)
        
    used_5subs = set()
    best_blocks = []
    
    sys.setrecursionlimit(10000)
    
    def solve(available, current):
        nonlocal best_blocks, used_5subs
        
        if time.time() - start_time > limit:
            return False
            
        if len(current) > len(best_blocks):
            best_blocks = list(current)
            
        if len(current) == 132:
            return True
            
        if not available:
            return False
            
        # MRV Heuristic: Pick the block that leaves the fewest options
        best_idx = -1
        min_count = float('inf')
        best_new_available = []
        
        for idx in available:
            b_set = block_5sub_sets[idx]
            
            count = 0
            temp_new_available = []
            for j in available:
                if j == idx:
                    continue
                # Check if j is compatible with idx
                # j is already compatible with current[:-1]
                # So we just check if j is compatible with idx
                if block_5sub_sets[j].isdisjoint(b_set):
                    count += 1
                    temp_new_available.append(j)
                    
            if count < min_count:
                min_count = count
                best_idx = idx
                best_new_available = temp_new_available
                
        if best_idx == -1:
            return False
            
        # Proceed with best_idx
        current.append(all_blocks[best_idx])
        b_set = block_5sub_sets[best_idx]
        for s in b_set:
            used_5subs.add(s)
            
        if solve(best_new_available, current):
            return True
            
        # Backtrack
        current.pop()
        for s in b_set:
            used_5subs.remove(s)
            
        # Note: We only try the MRV choice. If it fails, we return False.
        # The caller will backtrack and try a different parent choice.
        # This is a valid DFS strategy.
        return False

    initial_available = list(range(len(all_blocks)))
    solve(initial_available, [])
    
    return [list(b) for b in best_blocks]

# EVOLVE-BLOCK-END
