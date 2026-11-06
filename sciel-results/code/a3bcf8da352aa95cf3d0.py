import sys
from itertools import combinations

# Increase recursion limit just in case, though 132 is low
sys.setrecursionlimit(3000)

def construct():
    n = 12
    points = range(n)
    
    # 1. Generate all 5-subsets and map to indices
    fives = list(combinations(points, 5))
    five_to_idx = {f: i for i, f in enumerate(fives)}
    num_fives = len(fives) # 792
    
    # 2. Generate all 6-subsets and their bitmasks
    sixes = list(combinations(points, 6))
    six_masks = []
    six_tuples = []
    
    for s in sixes:
        mask = 0
        # A 6-set contains 6 5-subsets
        for i in range(6):
            # Remove the i-th element to get a 5-subset
            # s is a tuple. 
            # We can just generate all 5-subsets of s
            pass
        # Better: iterate 5-combinations of the 6-set
        for f in combinations(s, 5):
            idx = five_to_idx[f]
            mask |= (1 << idx)
        six_masks.append(mask)
        six_tuples.append(s)
        
    num_sixes = len(sixes) # 924
    
    # 3. Pre-compute which 6-sets contain which 5-set
    # covers[idx_5] = list of indices of 6-sets that contain this 5-set
    covers = [[] for _ in range(num_fives)]
    for i, mask in enumerate(six_masks):
        # Iterate through the 6 5-subsets in this 6-set
        # We can extract them by checking bits? Or just re-generate.
        # Re-generating is fine, it's pre-computation.
        s = six_tuples[i]
        for f in combinations(s, 5):
            idx_f = five_to_idx[f]
            covers[idx_f].append(i)
            
    # 4. Backtracking search
    # State: covered_mask, available_blocks (list of indices)
    # We want to find 132 blocks.
    
    solution = []
    
    def backtrack(covered_mask, available_blocks, depth):
        if depth == 132:
            # Check if all covered
            if covered_mask == (1 << num_fives) - 1:
                return True
            return False
            
        # Find an uncovered 5-subset
        # We pick the first uncovered one for simplicity (symmetry)
        target_idx = -1
        for i in range(num_fives):
            if not (covered_mask & (1 << i)):
                target_idx = i
                break
                
        if target_idx == -1:
            # All covered, but depth < 132? Impossible if valid.
            return False
            
        # Get candidates: blocks in available_blocks that contain target_idx
        # and are disjoint from covered_mask
        candidates = []
        for block_idx in covers[target_idx]:
            if block_idx in available_blocks:
                mask = six_masks[block_idx]
                if (mask & covered_mask) == 0:
                    candidates.append(block_idx)
                    
        if not candidates:
            return False
            
        # Try each candidate
        for block_idx in candidates:
            mask = six_masks[block_idx]
            new_covered = covered_mask | mask
            # Remove this block from available
            # Also, we should remove any other blocks that conflict with this one
            # to prune the search space.
            # A block conflicts if it shares any 5-subset with the chosen one.
            # Since the chosen block covers 6 specific 5-subsets, any other block
            # containing any of those 6 is now invalid.
            
            # Optimization:
            # Instead of passing a list of available blocks, we can maintain a set.
            # But sets are expensive to copy.
            # Alternative: pass a boolean array or just filter in the loop.
            # Given the small depth and branching, filtering `available_blocks` list is O(N) which is fine.
            
            new_available = []
            for b_idx in available_blocks:
                if b_idx == block_idx:
                    continue
                # Check if b_idx conflicts with block_idx
                if (six_masks[b_idx] & mask) != 0:
                    continue
                new_available.append(b_idx)
                
            solution.append(six_tuples[block_idx])
            
            if backtrack(new_covered, new_available, depth + 1):
                return True
                
            solution.pop()
            
        return False

    # Initial call
    initial_available = list(range(num_sixes))
    if backtrack(0, initial_available, 0):
        return solution
    else:
        # Fallback: return empty or raise
        return []
