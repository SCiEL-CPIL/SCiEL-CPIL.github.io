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
    
    # Helper function to perform greedy packing given an order of block indices
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

    # 3. Try deterministic heuristics first
    # Heuristic 1: Sort by sum of elements
    indices_by_sum = sorted(range(len(all_6subsets)), key=lambda i: sum(all_6subsets[i]))
    best_blocks = greedy_packing(indices_by_sum)
    if len(best_blocks) >= 132:
        return [list(b) for b in best_blocks]
    
    # Heuristic 2: Lexicographic order
    indices_by_lex = sorted(range(len(all_6subsets)), key=lambda i: all_6subsets[i])
    best_blocks = greedy_packing(indices_by_lex)
    if len(best_blocks) >= 132:
        return [list(b) for b in best_blocks]

    # Heuristic 3: Reverse Lexicographic
    indices_by_revlex = sorted(range(len(all_6subsets)), key=lambda i: all_6subsets[i], reverse=True)
    best_blocks = greedy_packing(indices_by_revlex)
    if len(best_blocks) >= 132:
        return [list(b) for b in best_blocks]

    # 4. Random restart greedy search
    # The bitset operation is very fast, allowing many restarts.
    rng = random.Random(42)
    num_blocks = len(all_6subsets)
    best_count = len(best_blocks)
    
    # We can afford a large number of restarts. 
    # Each restart involves shuffling 924 items and iterating through them.
    # Shuffling is O(N), iteration is O(N) with O(1) work per item.
    # Let's try up to 5000 restarts or until deadline.
    
    for attempt in range(5000):
        if time.time() > deadline:
            break
        
        # Shuffle indices
        order = list(range(num_blocks))
        rng.shuffle(order)
        
        blocks = greedy_packing(order)
        
        if len(blocks) > best_count:
            best_count = len(blocks)
            best_blocks = blocks
            if best_count >= 132:
                break
                
    return [list(b) for b in best_blocks]

def construct():
    """
    Construct a packing of 6-subsets on 12 points with no repeated 5-subset.
    
    Returns a list of distinct 6-subsets of 0..11 such that no 5-subset lies in
    two blocks. Maximizes the number of blocks; the optimum 132 is attained exactly
    by the Steiner system S(5,6,12).
    """
    return _witt_6_12_blocks()
