import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin
    
    n = 12
    k = 6
    t = 5
    
    # Pre-compute all blocks (6-subsets) and their 5-subsets
    all_blocks = list(itertools.combinations(range(n), k))
    block_to_5sets = {}
    for b in all_blocks:
        # Store as a tuple of frozensets or just tuples for hashing?
        # Tuples of sorted elements are hashable and efficient.
        # combinations returns tuples in sorted order already.
        s5 = tuple(itertools.combinations(b, t))
        block_to_5sets[b] = s5
    
    best_blocks = []
    
    # Run multiple randomized greedy attempts
    # The number of attempts is not fixed, but time-limited.
    # Each attempt iterates through ~924 blocks.
    # We can probably do thousands of attempts in 140s.
    
    while time.time() < deadline:
        random.shuffle(all_blocks)
        covered = set()
        current_blocks = []
        
        for b in all_blocks:
            # Check if any 5-subset is already covered
            # Since block_to_5sets[b] is a tuple of tuples, we can check intersection with covered set
            # However, `any(s in covered for s in ...)` is O(6) per block, which is fast.
            # A faster way might be to check if the set of 5-subsets is a subset of the complement of covered.
            # But given small size, simple loop is fine.
            
            # Optimization: Check if any of the 6 five-subsets is in covered
            conflict = False
            for s in block_to_5sets[b]:
                if s in covered:
                    conflict = True
                    break
            
            if not conflict:
                current_blocks.append(b)
                # Add all 5-subsets of this block to covered
                covered.update(block_to_5sets[b])
                
        # Compare with best
        if len(current_blocks) > len(best_blocks):
            best_blocks = current_blocks
            
            # If we found the theoretical maximum, we can stop early
            # Max blocks = C(12,5) / C(6,5) = 792 / 6 = 132
            if len(best_blocks) == 132:
                break

    # Return as list of lists to match interface expectation if strict, 
    # though list of tuples is usually fine for "subsets". 
    # The prompt says "list of distinct 6-subsets". Tuples are immutable sequences.
    # Let's return list of lists to be safe if the evaluator expects lists.
    return [list(b) for b in best_blocks]
