# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    total_t = len(all_t_subsets)
    
    # For C(24,7,3): total_t = 2024
    
    # Track coverage count for each t-subset
    # Index t-subsets for fast lookup
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # coverage[i] = number of blocks that cover all_t_subsets[i]
    coverage = [0] * total_t
    
    # uncovered[i] = True if all_t_subsets[i] is not yet covered
    uncovered = [True] * total_t
    uncovered_count = total_t
    
    # Precompute for t=3: for each pair (a,b) with a<b, and each x not in {a,b},
    # the index of the triple sorted((a,b,x))
    triple_index = {}
    for a in range(v):
        for b in range(a+1, v):
            pair = (a, b)
            triple_index[pair] = {}
            for x in range(v):
                if x != a and x != b:
                    triple = tuple(sorted((a, b, x)))
                    triple_index[pair][x] = t_subset_index[triple]
    
    def get_gain(block, x):
        """Compute how many uncovered t-subsets are covered by adding x to block."""
        gain = 0
        for i in range(len(block)):
            for j in range(i+1, len(block)):
                a, b = block[i], block[j]
                if a > b:
                    a, b = b, a
                pair = (a, b)
                idx = triple_index[pair][x]
                if uncovered[idx]:
                    gain += 1
        return gain
    
    def cover_block(block):
        """Update coverage and uncovered arrays for the given block."""
        nonlocal uncovered_count
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            if uncovered[idx]:
                uncovered[idx] = False
                uncovered_count -= 1
            coverage[idx] += 1
    
    # Greedy construction with multiple strategies
    best_blocks = None
    best_count = float('inf')
    
    # Try multiple random seeds for diversification
    for seed in range(20):
        if time.time() > deadline - 5:
            break
            
        random.seed(seed)
        
        # Reset state for this run
        coverage = [0] * total_t
        uncovered = [True] * total_t
        uncovered_count = total_t
        blocks = []
        
        # Use a priority queue approach: always start with the "hardest" uncovered t-subset
        # But for simplicity, we'll use a randomized greedy with lookahead
        
        while uncovered_count > 0:
            if time.time() > deadline - 2:
                break
            
            # Find the "hardest" uncovered t-subset
            # Use the one with minimum coverage (all are 0 initially)
            # To add diversity, we can weight by some heuristic
            candidates = [i for i in range(total_t) if uncovered[i]]
            if not candidates:
                break
            
            # Pick a random candidate to add diversity
            start_idx = random.choice(candidates)
            start_ts = all_t_subsets[start_idx]
            
            # Start block with the points of start_ts
            block = list(start_ts)
            block_set = set(block)
            
            # Greedily add points to maximize gain, with some randomization
            while len(block) < k:
                # Find best point to add
                best_x = None
                best_gain = -1
                second_best_x = None
                second_best_gain = -1
                
                for x in range(v):
                    if x not in block_set:
                        g = get_gain(block, x)
                        if g > best_gain:
                            second_best_x = best_x
                            second_best_gain = best_gain
                            best_x = x
                            best_gain = g
                        elif g > second_best_gain:
                            second_best_x = x
                            second_best_gain = g
                
                if best_x is None:
                    break
                
                # Add some randomization: 80% chance to pick best, 20% to pick second best or random
                if best_gain <= 0:
                    available = [x for x in range(v) if x not in block_set]
                    if available:
                        best_x = random.choice(available)
                elif random.random() < 0.2 and second_best_x is not None and second_best_gain >= 0:
                    best_x = second_best_x
                elif random.random() < 0.1:
                    available = [x for x in range(v) if x not in block_set]
                    if available:
                        best_x = random.choice(available)
                
                block.append(best_x)
                block_set.add(best_x)
            
            if len(block) == k:
                block.sort()
                b_tuple = tuple(block)
                # Check if this block is already in our list
                if b_tuple not in [tuple(b) for b in blocks]:
                    blocks.append(block)
                    cover_block(block)
        
        # Check if this is a valid complete covering
        if uncovered_count == 0 and len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
    
    # If no complete covering found, return the best partial one
    if best_blocks is not None:
        return best_blocks
    
    # Fallback: return whatever we have (may be incomplete)
    return blocks if 'blocks' in dir() else []
# EVOLVE-BLOCK-END
