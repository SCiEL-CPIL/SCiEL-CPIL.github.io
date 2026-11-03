"""New program using Pure Greedy Set Cover with Incremental Gain Updates."""
import itertools
import random
import numpy as np
from collections import defaultdict
from math import comb

def construct(v, k, t):
    # Generate all k-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map t-subsets to indices for coverage tracking
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # For each t-subset, list of blocks that contain it
    # This allows efficient gain updates
    tsub_to_blocks = defaultdict(list)
    # Also, for each block, list of t-subsets it contains (for marking covered)
    block_to_tsubs = []
    
    for b_idx, block in enumerate(all_blocks):
        tsubs_in_block = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            tsub_to_blocks[idx].append(b_idx)
            tsubs_in_block.append(idx)
        block_to_tsubs.append(tsubs_in_block)
        
    # Initial gain for each block is C(k, t)
    init_gain = comb(k, t)
    gain = np.full(n_blocks, init_gain, dtype=int)
    
    covered = np.zeros(n_tsubs, dtype=bool)
    
    def run_greedy(seed):
        rng = random.Random(seed)
        curr_gain = gain.copy()
        curr_covered = covered.copy()
        selected = []
        
        while True:
            # Find max gain
            max_gain = np.max(curr_gain)
            if max_gain == 0:
                break
            
            # Find all blocks with max gain
            candidates = np.where(curr_gain == max_gain)[0]
            if len(candidates) == 0:
                break
            
            # Pick one randomly to diversify solutions
            b_idx = candidates[rng.randint(0, len(candidates) - 1)]
            selected.append(b_idx)
            
            # Update coverage and gains
            for ts_idx in block_to_tsubs[b_idx]:
                if not curr_covered[ts_idx]:
                    curr_covered[ts_idx] = True
                    # Decrement gain for all blocks containing this t-subset
                    for other_b_idx in tsub_to_blocks[ts_idx]:
                        curr_gain[other_b_idx] -= 1
                        
        return selected
    
    # Run multiple restarts to find a good solution
    best_solution = None
    best_len = float('inf')
    
    # How many restarts? 20 should be fast enough (< 10s)
    n_restarts = 20
    
    for i in range(n_restarts):
        selected_indices = run_greedy(i * 1000 + 7)
        if len(selected_indices) < best_len:
            best_len = len(selected_indices)
            best_solution = selected_indices
            
    # Convert indices back to blocks
    if best_solution is None:
        return []
        
    return [list(all_blocks[i]) for i in best_solution]
