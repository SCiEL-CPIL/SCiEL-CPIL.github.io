import time
import sys
from itertools import combinations

# Precomputation
# Points 0..11
# All 6-subsets
all_blocks = []
block_to_5sets = []
# Map 5-subset to ID
# 5-subsets of 0..11
five_sets = list(combinations(range(12), 5))
five_set_to_id = {s: i for i, s in enumerate(five_sets)}
n_5sets = len(five_sets)

# For each block, store its 5-set IDs
for b in combinations(range(12), 6):
    ids = [five_set_to_id[s] for s in combinations(b, 5)]
    all_blocks.append(list(b))
    block_to_5sets.append(ids)

# Invert: for each 5-set ID, which blocks contain it?
blocks_containing_5set = [[] for _ in range(n_5sets)]
for i, ids in enumerate(block_to_5sets):
    for id in ids:
        blocks_containing_5set[id].append(i)

# Precompute for each block: bitmask of its 5-sets
# 792 5-sets -> use Python int as bitmask
block_mask = []
for ids in block_to_5sets:
    m = 0
    for id in ids:
        m |= (1 << id)
    block_mask.append(m)

# Precompute: for each 5-set, the list of block indices containing it (already have)
# Also precompute: for each block, the list of other 5-set IDs (excluding a given one)
# Actually, we'll compute validity on the fly but use bitmasks for speed.

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    covered_mask = 0
    blocks_found = []
    
    def backtrack():
        nonlocal covered_mask
        
        if time.time() > deadline:
            return False
        
        # Find uncovered 5-sets and check for forced moves
        # We need to find the uncovered 5-set with fewest valid candidates
        # A block is valid for 5-set s if all its 5-sets are uncovered
        
        min_count = float('inf')
        best_candidates = []
        
        # Iterate through all 5-sets to find uncovered ones
        for s_id in range(n_5sets):
            if covered_mask & (1 << s_id):
                continue  # Already covered
            
            # Count valid blocks containing this 5-set
            candidates = []
            for b_idx in blocks_containing_5set[s_id]:
                # Check if all 5-sets of this block are uncovered
                # block_mask[b_idx] has bits for all 5-sets in the block
                # If any of those bits are in covered_mask, the block is invalid
                if covered_mask & block_mask[b_idx] == 0:
                    candidates.append(b_idx)
            
            if len(candidates) < min_count:
                min_count = len(candidates)
                best_candidates = candidates
                
                if min_count == 0:
                    return False  # Dead end
                
                if min_count == 1:
                    # Forced move - we must take this block
                    # But we still need to verify no other 5-set is stuck
                    # Actually, if we take this block, it might cover other 5-sets
                    # and make them have 0 candidates. So we should continue scanning.
                    pass
        
        if min_count == 0:
            return False
        
        # Try candidates
        for b_idx in best_candidates:
            blocks_found.append(all_blocks[b_idx])
            covered_mask |= block_mask[b_idx]
            
            if backtrack():
                return True
            
            blocks_found.pop()
            covered_mask &= ~block_mask[b_idx]
        
        return False
    
    if backtrack():
        return blocks_found
    
    return blocks_found
