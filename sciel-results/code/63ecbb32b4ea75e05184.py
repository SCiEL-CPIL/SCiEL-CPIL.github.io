import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds to be safe within 120s limit
    
    points = range(12)
    all_blocks = list(combinations(points, 6))
    # Precompute the 5-subsets for each block to speed up checks
    # Map each 5-subset to a unique ID or just use the tuple itself in a set
    # Using tuples of 5 integers is efficient enough for 792 subsets
    
    best_blocks = []
    best_count = 0
    
    # Number of restarts. We can do many because each run is fast.
    # Each run iterates through 924 blocks.
    # Checking 6 subsets per block against a set is O(1) average.
    # So one run is ~ 924 * 6 * O(1) operations. Very fast.
    # We can do thousands of restarts.
    
    # Strategy: Randomized Greedy
    # 1. Shuffle blocks
    # 2. Greedily add if valid
    # 3. Track best
    
    # To improve efficiency, we can also try different random seeds or shuffling strategies.
    
    # Initial deterministic run to get a baseline
    # But randomized is better. Let's just do randomized.
    
    # We'll run until deadline or a high number of restarts
    # Let's say 5000 restarts, but break if time runs out.
    
    for i in range(5000):
        if time.time() > deadline:
            break
            
        random.shuffle(all_blocks)
        covered = set()
        current_blocks = []
        
        for block in all_blocks:
            # Check if any 5-subset of this block is in covered
            # A block has 6 five-subsets
            valid = True
            subsets = []
            for j in range(6):
                # Create 5-subset by removing element at index j
                sub = tuple(block[k] for k in range(6) if k != j)
                subsets.append(sub)
                if sub in covered:
                    valid = False
                    break
            
            if valid:
                current_blocks.append(block)
                for sub in subsets:
                    covered.add(sub)
        
        if len(current_blocks) > best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks
            
            # If we found the optimum, we can stop early?
            # The optimum is 132.
            if best_count == 132:
                # We can stop, but maybe run a few more to ensure consistency?
                # 132 is the max possible. So we can break.
                break
        
        # Optional: If we are getting stuck at a lower number, we might need a different heuristic.
        # But randomized greedy is known to work well here.
        
    # Return as list of lists to match interface if required, or list of tuples?
    # "list of distinct 6-subsets" usually implies list of lists or tuples.
    # The seed program returns `list(b)` where b is a tuple. So list of lists.
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
