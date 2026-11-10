"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # All possible t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # For each possible block (k-subset), precompute which t-subsets it covers
    # This is expensive but necessary for efficient greedy set cover
    # However, C(19,8) = 75582 blocks, each covering C(8,5)=56 subsets
    # Total coverage entries: 75582 * 56 = 4,232,592 - manageable
    
    # We'll use a lazy approach: instead of precomputing all blocks, 
    # we'll use the greedy point-addition approach but with randomization
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple randomized greedy attempts
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Randomize the order of t-subsets
        uncovered_indices = list(range(n_t_subsets))
        random.shuffle(uncovered_indices)
        
        uncovered_set = set(uncovered_indices)
        blocks = []
        
        while uncovered_set and time.time() < deadline:
            # Pick a random uncovered t-subset to start the block
            start_idx = random.choice(list(uncovered_set))
            start_subset = all_t_subsets[start_idx]
            
            block = list(start_subset)
            
            # Greedily add points to maximize coverage of uncovered t-subsets
            while len(block) < k:
                best_gain = -1
                best_point = None
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    # Calculate how many uncovered t-subsets would be covered
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in subset_to_idx:
                            idx = subset_to_idx[candidate]
                            if idx in uncovered_set:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point is not None:
                    block.append(best_point)
                else:
                    # No point improves coverage, pick a random one
                    remaining = [x for x in range(v) if x not in block]
                    if remaining:
                        block.append(random.choice(remaining))
                    else:
                        break
            
            block.sort()
            blocks.append(tuple(block))
            
            # Remove all t-subsets covered by this block
            for ts in itertools.combinations(block, t):
                idx = subset_to_idx.get(ts)
                if idx is not None:
                    uncovered_set.discard(idx)
        
        # Check if we have a complete cover
        if not uncovered_set:
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = [list(b) for b in blocks]
        
        # Early termination if we found a very good solution
        if best_count <= 280:
            break
    
    if best_blocks is None:
        # Fallback: use the original deterministic greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered and time.time() < deadline:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        best_blocks = blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
