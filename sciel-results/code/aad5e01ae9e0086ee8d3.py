import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Correct base blocks for S(4, 5, 11)
    # These generate the unique Steiner System S(4, 5, 11) when developed mod 11
    base_blocks = [
        (0, 1, 3, 7, 9),
        (0, 1, 4, 6, 10),
        (0, 2, 3, 5, 8),
        (0, 2, 4, 7, 10),
        (0, 2, 5, 6, 9),
        (0, 3, 4, 5, 10)
    ]
    
    # Generate all 66 blocks of S(4, 5, 11)
    s4511_blocks = []
    for base in base_blocks:
        for i in range(11):
            block = tuple(sorted((x + i) % 11 for x in base))
            s4511_blocks.append(block)
            
    # Remove duplicates (should be 66 unique blocks)
    s4511_blocks = list(set(s4511_blocks))
    
    if len(s4511_blocks) == 66:
        # Construct S(5, 6, 12) blocks
        # Points are 0..11, where 11 is the point at infinity.
        # Type 1: {11} U B for B in S(4, 5, 11)
        # Type 2: {0..10} \ B for B in S(4, 5, 11)
        
        blocks = []
        points_0_10 = set(range(11))
        
        for B in s4511_blocks:
            # Type 1
            b1 = tuple(sorted(list(B) + [11]))
            blocks.append(b1)
            
            # Type 2
            b2_set = points_0_10 - set(B)
            b2 = tuple(sorted(b2_set))
            blocks.append(b2)
        
        # Verify the constructed design
        if len(blocks) == 132 and len(set(blocks)) == 132:
            covered_5s = set()
            valid = True
            for b in blocks:
                for s in combinations(b, 5):
                    if s in covered_5s:
                        valid = False
                        break
                    covered_5s.add(s)
                if not valid:
                    break
            
            if valid:
                return [list(b) for b in blocks]

    # Fallback: Randomized Multi-Restart Greedy
    # This is used if the direct construction fails or is invalid.
    return greedy_construct_randomized()

def greedy_construct_randomized():
    """
    Fallback greedy construction with random restarts.
    """
    points = list(range(12))
    all_6_subsets = list(combinations(points, 6))
    
    best_blocks = []
    best_count = 0
    
    # Number of restarts based on time
    # Each restart is fast (iterating 924 items)
    start_time = time.time()
    deadline = start_time + 110.0
    
    num_restarts = 50 # Cap restarts to ensure we finish
    
    for _ in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        random.shuffle(all_6_subsets)
        covered = set()
        blocks = []
        
        for b in all_6_subsets:
            block_5s = list(combinations(b, 5))
            if not any(s in covered for s in block_5s):
                blocks.append(b)
                covered.update(block_5s)
                
        if len(blocks) > best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        # If we hit the optimum, we can stop early
        if best_count == 132:
            break
            
    return [list(b) for b in best_blocks]
