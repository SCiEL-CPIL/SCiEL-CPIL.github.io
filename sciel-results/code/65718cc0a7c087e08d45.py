import itertools
import random
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    
    # Map t-subset to index
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Precompute all possible k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, precompute which t-subsets it covers
    block_tsub_indices = []
    for block in all_blocks:
        tsubs = [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_tsub_indices.append(tsubs)
    
    # Precompute for each (t-1)-subset, the mapping from point x to t-subset index
    # for t-subsets that contain this (t-1)-subset
    t_minus_1_ext = {}
    for tsub in all_t_subsets:
        idx = t_sub_to_idx[tsub]
        for i in range(t):
            rest = tsub[:i] + tsub[i+1:]
            x = tsub[i]
            key = frozenset(rest)
            if key not in t_minus_1_ext:
                t_minus_1_ext[key] = {}
            t_minus_1_ext[key][x] = idx
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple runs with different random seeds
    for run_id in range(200):  # Max 200 runs, but limited by time
        if time.time() > deadline - 5.0:
            break
            
        random.seed(run_id * 12345 + 67890)
        
        uncovered = np.ones(num_t, dtype=bool)
        blocks = []
        
        while uncovered.any():
            # Pick a random uncovered t-subset to seed the block
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            seed_idx = random.choice(uncovered_indices)
            seed_subset = all_t_subsets[seed_idx]
            block = list(seed_subset)
            
            while len(block) < k:
                # Find x that covers most uncovered t-subsets
                best_x = -1
                best_gain = -1
                tied_candidates = []
                
                block_set = set(block)
                
                # Get all (t-1)-subsets of current block
                if len(block) >= t - 1:
                    t_minus_1_subsets_of_block = list(itertools.combinations(block, t - 1))
                else:
                    t_minus_1_subsets_of_block = []
                
                # For each candidate x, compute gain using precomputed lookup
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    for rest in t_minus_1_subsets_of_block:
                        key = frozenset(rest)
                        if key in t_minus_1_ext and x in t_minus_1_ext[key]:
                            pidx = t_minus_1_ext[key][x]
                            if uncovered[pidx]:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        tied_candidates = [x]
                    elif gain == best_gain:
                        tied_candidates.append(x)
                        
                if best_x == -1:
                    break
                    
                # Random tie-breaking
                if len(tied_candidates) > 1:
                    best_x = random.choice(tied_candidates)
                    
                block.append(best_x)
                
            block.sort()
            blocks.append(block)
            
            # Remove covered t-subsets
            for sub in itertools.combinations(block, t):
                if sub in t_sub_to_idx:
                    uncovered[t_sub_to_idx[sub]] = False
            
        # End while uncovered
        
        # Local Optimization: Remove redundant blocks
        if len(blocks) < best_count:
            # Try to remove redundant blocks iteratively
            changed = True
            while changed:
                changed = False
                if time.time() > deadline - 2.0:
                    break
                    
                # Build coverage count
                coverage_count = np.zeros(num_t, dtype=np.int32)
                for b in blocks:
                    for sub in itertools.combinations(b, t):
                        if sub in t_sub_to_idx:
                            coverage_count[t_sub_to_idx[sub]] += 1
                
                # Check each block for redundancy
                redundant_indices = []
                for i, b in enumerate(blocks):
                    is_essential = False
                    for sub in itertools.combinations(b, t):
                        if sub in t_sub_to_idx:
                            if coverage_count[t_sub_to_idx[sub]] == 1:
                                is_essential = True
                                break
                    if not is_essential:
                        redundant_indices.append(i)
                        
                if redundant_indices:
                    # Remove one redundant block at a time
                    idx = random.choice(redundant_indices)
                    blocks.pop(idx)
                    changed = True
                    
        # End redundancy check
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
    # If no blocks found, fallback to simple greedy
    if best_blocks is None:
        uncovered = np.ones(num_t, dtype=bool)
        blocks = []
        while uncovered.any():
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            seed_idx = uncovered_indices[0]
            seed_subset = all_t_subsets[seed_idx]
            block = list(seed_subset)
            
            while len(block) < k:
                best_x = -1
                best_gain = -1
                block_set = set(block)
                if len(block) >= t - 1:
                    t_minus_1_subsets = list(itertools.combinations(block, t - 1))
                else:
                    t_minus_1_subsets = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in t_minus_1_subsets:
                        key = frozenset(rest)
                        if key in t_minus_1_ext and x in t_minus_1_ext[key]:
                            pidx = t_minus_1_ext[key][x]
                            if uncovered[pidx]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            blocks.append(block)
            
            for sub in itertools.combinations(block, t):
                if sub in t_sub_to_idx:
                    uncovered[t_sub_to_idx[sub]] = False
                    
        best_blocks = blocks
        best_count = len(blocks)
        
    return best_blocks
