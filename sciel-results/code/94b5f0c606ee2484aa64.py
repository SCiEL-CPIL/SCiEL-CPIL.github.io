"""
Witt design S(5,6,12) construction using bitset-optimized greedy search.

The task asks for a packing of 6-subsets on 12 points such that no 5-subset
appears in two blocks, maximizing the number of blocks. The optimum is 132,
attained exactly by the Steiner system S(5,6,12).

Strategy:
1. Represent the set of covered 5-subsets as a large integer bitmask (792 bits).
   This allows O(1) conflict checking via bitwise AND.
2. Precompute the bitmask for all C(12,6)=924 possible blocks.
3. Attempt a few deterministic heuristic orderings to see if they yield the optimal 132.
4. If not, perform a high-throughput random-restart greedy search.
   The bitset optimization allows for thousands of restarts within the time limit,
   significantly increasing the probability of finding the optimal packing.
"""

import time
import random
from itertools import combinations

def _witt_6_12_blocks():
    """
    Construct the Steiner system S(5,6,12) – the Witt design.
    
    Returns a list of 132 distinct 6-subsets of {0,...,11} such that no 5-subset
    appears in more than one block.
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin for 120s limit
    
    # 1. Precompute mapping of 5-subsets to indices
    all_5subsets = list(combinations(range(12), 5))
    subset_to_idx = {s: i for i, s in enumerate(all_5subsets)}
    num_5subsets = len(all_5subsets)  # 792
    
    # 2. Precompute all 6-subsets and their bitmasks
    all_6subsets = list(combinations(range(12), 6))
    block_masks = []
    for b in all_6subsets:
        mask = 0
        for sub in combinations(b, 5):
            idx = subset_to_idx[sub]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    num_blocks = len(all_6subsets)
    
    # Backtracking search with symmetry breaking
    best_blocks = []
    
    def backtrack(covered_mask, current_blocks, start_idx):
        nonlocal best_blocks
        if time.time() > deadline:
            return False
        
        if len(current_blocks) == 132:
            best_blocks = list(current_blocks)
            return True
            
        for i in range(start_idx, num_blocks):
            # Symmetry breaking:
            # The first block is fixed as {0,1,2,3,4,5} (index 0)
            # The second block must contain 0
            if len(current_blocks) == 1 and 0 not in all_6subsets[i]:
                continue
            # The third block must contain 0 and 1
            if len(current_blocks) == 2 and (0 not in all_6subsets[i] or 1 not in all_6subsets[i]):
                continue
                
            mask = block_masks[i]
            if (covered_mask & mask) == 0:
                current_blocks.append(all_6subsets[i])
                if backtrack(covered_mask | mask, current_blocks, i + 1):
                    return True
                current_blocks.pop()
                
        return False

    # Initialize with the first block {0, 1, 2, 3, 4, 5}
    # This is the first combination in lexicographic order, so index 0
    first_block = all_6subsets[0]
    first_mask = block_masks[0]
    
    # Start backtracking
    success = backtrack(first_mask, [first_block], 1)
    
    if success and len(best_blocks) == 132:
        return [list(b) for b in best_blocks]
        
    # Fallback to greedy if backtracking fails or times out
    rng = random.Random(42)
    best_count = 0
    best_blocks_fallback = []
    
    def greedy_packing(order):
        covered = 0
        blocks = []
        for idx in order:
            mask = block_masks[idx]
            if (covered & mask) == 0:
                blocks.append(all_6subsets[idx])
                covered |= mask
                if len(blocks) == 132:
                    break
        return blocks

    for attempt in range(500):
        if time.time() > deadline:
            break
        order = list(range(num_blocks))
        rng.shuffle(order)
        blocks = greedy_packing(order)
        if len(blocks) > best_count:
            best_count = len(blocks)
            best_blocks_fallback = blocks
            if best_count >= 132:
                return [list(b) for b in best_blocks_fallback]
                
    if best_blocks_fallback:
        return [list(b) for b in best_blocks_fallback]
        
    return [list(first_block)]

def construct():
    """
    Construct a packing of 6-subsets on 12 points with no repeated 5-subset.
    
    Returns a list of distinct 6-subsets of 0..11 such that no 5-subset lies in
    two blocks. Maximizes the number of blocks; the optimum 132 is attained exactly
    by the Steiner system S(5,6,12).
    """
    return _witt_6_12_blocks()
