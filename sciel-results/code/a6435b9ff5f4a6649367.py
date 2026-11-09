import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 110.0 # 10 seconds margin before 120s limit

    # 1. Precomputation
    # Points are 0..11
    # A block is a 12-bit integer.
    # A 5-subset is a 12-bit integer with 5 bits set.
    
    # Generate all 6-subsets (blocks)
    all_blocks = []
    block_to_subs = {}
    
    for b in combinations(range(12), 6):
        mask = 0
        for i in b:
            mask |= (1 << i)
        all_blocks.append(mask)
        
        # Precompute the 5-subsets of this block
        subs = []
        for s in combinations(b, 5):
            smask = 0
            for i in s:
                smask |= (1 << i)
            subs.append(smask)
        block_to_subs[mask] = subs

    # Sort blocks for deterministic and potentially faster search (lexicographic)
    all_blocks.sort()
    
    # Symmetry breaking: 
    # The first block is fixed to {0, 1, 2, 3, 4, 5}
    first_block_mask = (1 << 6) - 1 # 0b001111111111
    if first_block_mask not in block_to_subs:
        # Fallback if something went wrong, though this shouldn't happen
        return []
        
    initial_covered = set(block_to_subs[first_block_mask])
    initial_blocks = [first_block_mask]
    
    # Filter candidate blocks
    # A block is compatible with the first block if it shares at most 4 points.
    # If it shares 5 or 6 points, it contains a 5-subset of the first block.
    # Since we already covered all 5-subsets of the first block, any such block is invalid.
    candidates = []
    for mask in all_blocks:
        if mask == first_block_mask:
            continue
        inter = mask & first_block_mask
        if bin(inter).count('1') <= 4:
            candidates.append(mask)
            
    # We need to find 131 more blocks from 'candidates'
    # Total blocks = 132
    
    sys.setrecursionlimit(10000)
    
    def backtrack(idx, covered, current_blocks):
        if time.time() > deadline:
            return None
            
        if len(current_blocks) == 132:
            return current_blocks[:]
            
        # Pruning: Not enough blocks left to reach 132
        if len(current_blocks) + (len(candidates) - idx) < 132:
            return None
            
        for i in range(idx, len(candidates)):
            mask = candidates[i]
            subs = block_to_subs[mask]
            
            # Check for conflict
            conflict = False
            for s in subs:
                if s in covered:
                    conflict = True
                    break
            
            if not conflict:
                # Include this block
                new_covered = set(covered)
                new_covered.update(subs)
                new_blocks = current_blocks + [mask]
                
                res = backtrack(i + 1, new_covered, new_blocks)
                if res is not None:
                    return res
            
            # Note: In a Steiner system, if we skip a block, we might still find a solution 
            # with other blocks. However, for S(5,6,12), the structure is very rigid.
            # A simple "try include, then try exclude" is safe.
            # But we can optimize: if we include a block, we don't need to try excluding it 
            # in the same branch if we are just looking for *a* solution, but we DO need to 
            # backtrack if including it leads to a dead end.
            
        return None

    # Start search
    solution_masks = backtrack(0, initial_covered, initial_blocks)
    
    if solution_masks:
        # Convert masks back to list of points
        result = []
        for mask in solution_masks:
            pts = []
            for i in range(12):
                if mask & (1 << i):
                    pts.append(i)
            result.append(pts)
        return result
    else:
        # Fallback: Greedy (should not be reached if backtracking is correct)
        covered, blocks = set(), []
        for b in combinations(range(12), 6):
            mask = 0
            for i in b:
                mask |= (1 << i)
            if mask in block_to_subs:
                subs = block_to_subs[mask]
                if not any(s in covered for s in subs):
                    blocks.append(list(b))
                    covered.update(subs)
        return blocks

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Found {len(res)} blocks")
    # Verify
    if len(res) == 132:
        all_5subs = set()
        for b in res:
            for s in combinations(b, 5):
                s_mask = 0
                for i in s:
                    s_mask |= (1 << i)
                if s_mask in all_5subs:
                    print("ERROR: Duplicate 5-subset")
                    break
                all_5subs.add(s_mask)
        if len(all_5subs) == 792:
            print("Valid S(5,6,12) constructed")
        else:
            print(f"Invalid: {len(all_5subs)} unique 5-subsets")
    else:
        print("Invalid: Wrong number of blocks")
