import sys
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 110.0 # 10 seconds margin
    
    # Bitmask for 12 points
    # Block is a 12-bit integer with 6 bits set.
    
    # Precompute 5-subsets for each possible 6-subset?
    # There are C(12,6) = 924 possible blocks.
    # For each block, the 6 5-subsets can be precomputed.
    
    from itertools import combinations
    
    # Map from bitmask to list of 5-subset bitmasks
    block_5subs = {}
    for b in combinations(range(12), 6):
        mask = 0
        for i in b:
            mask |= (1 << i)
        subs = []
        for s in combinations(b, 5):
            smask = 0
            for i in s:
                smask |= (1 << i)
            subs.append(smask)
        block_5subs[mask] = subs

    # List of all possible blocks sorted lexicographically (by mask)
    # We need to iterate through them in a specific order to find the unique solution.
    # The solution is unique, so any valid packing of 132 is it.
    # However, we must find a packing of size 132.
    
    # We use backtracking.
    # State: 
    #   current_blocks: list of masks
    #   covered_5subs: set of 5-subset masks
    #   next_block_index: index in the sorted list of all blocks
    
    all_blocks = sorted(block_5subs.keys())
    n_blocks = len(all_blocks)
    
    # Symmetry breaking:
    # The first block can be assumed to be the one with the smallest mask: {0,1,2,3,4,5}
    # Mask: 0b001111111111 = 0x03FF = 1023
    first_block_mask = (1 << 6) - 1
    if first_block_mask not in block_5subs:
        return []
        
    initial_covered = set(block_5subs[first_block_mask])
    initial_blocks = [first_block_mask]
    
    # We need to find 131 more blocks.
    # We iterate through the remaining blocks.
    
    # To speed up, we can filter blocks that are incompatible with the first block.
    # A block is incompatible if it shares a 5-subset with the first block.
    # Since the first block is {0,1,2,3,4,5}, any block containing 5 of these is incompatible.
    # This means any block that is a subset of {0,1,2,3,4,5} plus one other point is incompatible.
    # Actually, if a block shares 5 points with the first block, it shares a 5-subset.
    # So we exclude blocks that contain 5 or 6 points from {0,1,2,3,4,5}.
    # Since block size is 6, it can contain 5 or 6 points from the first block.
    # If it contains 6, it IS the first block.
    # If it contains 5, it is incompatible.
    
    compatible_blocks = []
    for mask in all_blocks:
        if mask == first_block_mask:
            continue
        # Count intersection with first_block_mask
        inter = mask & first_block_mask
        # Count bits in inter
        # If inter has 5 or 6 bits, it's incompatible.
        # inter has 6 bits only if mask == first_block_mask (already skipped).
        # So check if inter has 5 bits.
        if bin(inter).count('1') == 5:
            continue
        compatible_blocks.append(mask)
        
    # We need to pick 131 blocks from compatible_blocks.
    # This is a large search space.
    # However, the structure is rigid.
    
    # We use a recursive DFS.
    # To avoid recursion depth issues and overhead, we might use an iterative approach or just standard recursion with sys.setrecursionlimit.
    sys.setrecursionlimit(2000)
    
    best_solution = []
    
    def backtrack(index, current_blocks, covered):
        nonlocal best_solution
        if time.time() > deadline:
            return
        
        if len(current_blocks) == 132:
            best_solution = current_blocks[:]
            return
        
        # Heuristic: Try blocks in lexicographic order
        # We can prune if the remaining number of blocks is not enough to reach 132.
        # But we don't know the exact count easily.
        
        for i in range(index, len(compatible_blocks)):
            mask = compatible_blocks[i]
            subs = block_5subs[mask]
            
            # Check conflict
            conflict = False
            for s in subs:
                if s in covered:
                    conflict = True
                    break
            
            if not conflict:
                # Add block
                new_covered = set(covered)
                new_covered.update(subs)
                new_blocks = current_blocks + [mask]
                
                backtrack(i + 1, new_blocks, new_covered)
                
                if best_solution:
                    return
                # Undo (handled by creating new sets/lists)
                
    backtrack(0, initial_blocks, initial_covered)
    
    if best_solution:
        # Convert masks back to lists of points
        result = []
        for mask in best_solution:
            pts = []
            for i in range(12):
                if mask & (1 << i):
                    pts.append(i)
            result.append(pts)
        return result
    else:
        # Fallback to greedy if backtracking fails (should not happen for S(5,6,12))
        covered, blocks = set(), []
        for b in combinations(range(12), 6):
            subsets = list(combinations(b, 5))
            if not any(s in covered for s in subsets):
                blocks.append(list(b))
                covered.update(subsets)
        return blocks

# EVOLVE-BLOCK-START
def construct():
    # Implementation as above
    import time
    import sys
    from itertools import combinations

    start_time = time.time()
    deadline = 110.0

    block_5subs = {}
    for b in combinations(range(12), 6):
        mask = 0
        for i in b:
            mask |= (1 << i)
        subs = []
        for s in combinations(b, 5):
            smask = 0
            for i in s:
                smask |= (1 << i)
            subs.append(smask)
        block_5subs[mask] = subs

    all_blocks = sorted(block_5subs.keys())
    
    first_block_mask = (1 << 6) - 1
    if first_block_mask not in block_5subs:
        return []
        
    initial_covered = set(block_5subs[first_block_mask])
    initial_blocks = [first_block_mask]
    
    compatible_blocks = []
    for mask in all_blocks:
        if mask == first_block_mask:
            continue
        inter = mask & first_block_mask
        if bin(inter).count('1') == 5:
            continue
        compatible_blocks.append(mask)
        
    sys.setrecursionlimit(2000)
    best_solution = []
    
    def backtrack(index, current_blocks, covered):
        nonlocal best_solution
        if time.time() > deadline:
            return
        
        if len(current_blocks) == 132:
            best_solution = current_blocks[:]
            return
        
        for i in range(index, len(compatible_blocks)):
            mask = compatible_blocks[i]
            subs = block_5subs[mask]
            
            conflict = False
            for s in subs:
                if s in covered:
                    conflict = True
                    break
            
            if not conflict:
                new_covered = set(covered)
                new_covered.update(subs)
                new_blocks = current_blocks + [mask]
                
                backtrack(i + 1, new_blocks, new_covered)
                
                if best_solution:
                    return
                
    backtrack(0, initial_blocks, initial_covered)
    
    if best_solution:
        result = []
        for mask in best_solution:
            pts = []
            for i in range(12):
                if mask & (1 << i):
                    pts.append(i)
            result.append(pts)
        return result
    else:
        covered, blocks = set(), []
        for b in combinations(range(12), 6):
            subsets = list(combinations(b, 5))
            if not any(s in covered for s in subsets):
                blocks.append(list(b))
                covered.update(subsets)
        return blocks
# EVOLVE-BLOCK-END
