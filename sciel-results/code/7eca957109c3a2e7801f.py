import time
from itertools import combinations

def construct():
    """
    Constructs the Steiner system S(5,6,12).
    Returns a list of 132 distinct 6-subsets of 0..11.
    """
    start_time = time.time()
    deadline = start_time + 100
    
    points = range(12)
    all_blocks = list(combinations(points, 6))
    
    # Map block to its 5-subsets
    block_5s = {}
    for b in all_blocks:
        block_5s[b] = tuple(combinations(b, 5))
        
    # Total 5-subsets
    total_5_subsets = 792
    blocks_found = []
    
    def backtrack(index, current_blocks, current_covered):
        if time.time() > deadline:
            return False
            
        if len(current_blocks) == 132:
            blocks_found.extend(current_blocks)
            return True
            
        remaining_needed = 132 - len(current_blocks)
        if len(all_blocks) - index < remaining_needed:
            return False
            
        for i in range(index, len(all_blocks)):
            b = all_blocks[i]
            b_5s = block_5s[b]
            
            conflict = False
            for s in b_5s:
                if s in current_covered:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            new_covered = current_covered.union(b_5s)
            if backtrack(i + 1, current_blocks + [b], new_covered):
                return True
                
        return False

    # Force first block to (0, 1, 2, 3, 4, 5) to leverage symmetry
    first_block = (0, 1, 2, 3, 4, 5)
    first_block_5s = block_5s[first_block]
    first_block_index = all_blocks.index(first_block)
    
    if backtrack(first_block_index + 1, [first_block], set(first_block_5s)):
        return [list(b) for b in blocks_found]
        
    # Fallback full search
    if backtrack(0, [], set()):
        return [list(b) for b in blocks_found]
        
    return blocks_found
