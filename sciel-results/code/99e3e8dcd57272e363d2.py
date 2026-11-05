import numpy as np
import itertools
import time
from math import comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Total number of t-subsets
    total_tsubsets = comb(v, t)
    
    # Represent each t-subset as a sorted tuple -> index mapping
    # For efficiency, we'll use a bitset approach with numpy
    # Each t-subset is identified by its sorted elements
    
    # Generate all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Track which t-subsets are covered: 1 = covered, 0 = not
    covered = np.zeros(total_tsubsets, dtype=np.int8)
    
    # For each point, precompute which t-subsets contain it
    # point_tsubsets[p] = list of indices of t-subsets containing point p
    point_tsubsets = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubsets):
        for p in ts:
            point_tsubsets[p].append(i)
    
    # Convert to numpy arrays for speed
    point_tsubset_arrays = [np.array(pts, dtype=np.int32) for pts in point_tsubsets]
    
    blocks = []
    
    # Greedy covering with multiple strategies
    # Strategy: pick uncovered t-subset with highest "urgency", then build block greedily
    
    while True:
        if time.time() > deadline - 5:
            break
        
        # Find uncovered t-subsets
        uncovered_indices = np.where(covered == 0)[0]
        if len(uncovered_indices) == 0:
            break
        
        # Pick the first uncovered t-subset (or use a heuristic)
        # Use the one with highest "cost" - but for simplicity, just pick first
        target_ts = all_tsubsets[uncovered_indices[0]]
        
        # Start block with the target t-subset's elements
        block_set = set(target_ts)
        block_list = list(target_ts)
        
        # Greedily add points that cover the most uncovered t-subsets
        while len(block_list) < k:
            if time.time() > deadline - 3:
                break
            
            best_point = -1
            best_gain = -1
            
            for p in range(v):
                if p in block_set:
                    continue
                
                # Calculate gain: how many uncovered t-subsets would be covered
                # A t-subset is covered by adding p if it contains p and (t-1) elements from block_list
                gain = 0
                if t - 1 <= len(block_list):
                    # For each (t-1)-subset of block_list, check if adding p forms an uncovered t-subset
                    for rest in itertools.combinations(block_list, t - 1):
                        candidate = tuple(sorted(rest + (p,)))
                        if candidate in tsubset_to_idx:
                            idx = tsubset_to_idx[candidate]
                            if covered[idx] == 0:
                                gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
            
            if best_point == -1 or best_gain == 0:
                # Pick a random point
                available = [p for p in range(v) if p not in block_set]
                if available:
                    best_point = available[np.random.randint(len(available))]
                else:
                    break
            
            block_set.add(best_point)
            block_list.append(best_point)
        
        # Sort block for consistency
        block_list.sort()
        
        # Check if this block is already in our list
        block_tuple = tuple(block_list)
        if block_tuple not in [tuple(b) for b in blocks]:
            blocks.append(block_list)
            
            # Update covered array
            for ts in itertools.combinations(block_list, t):
                idx = tsubset_to_idx[ts]
                covered[idx] = 1
        
        # Check if we've covered everything
        if np.all(covered == 1):
            break
    
    # Post-processing: try to remove redundant blocks
    if time.time() < deadline - 2 and len(blocks) > 0:
        # For each block, check if it's redundant
        i = 0
        while i < len(blocks):
            if time.time() > deadline - 2:
                break
            
            # Temporarily remove block i
            temp_blocks = blocks[:i] + blocks[i+1:]
            
            # Check if all t-subsets are still covered
            still_covered = np.zeros(total_tsubsets, dtype=np.int8)
            for b in temp_blocks:
                for ts in itertools.combinations(b, t):
                    still_covered[tsubset_to_idx[ts]] = 1
            
            if np.all(still_covered == 1):
                # Block i is redundant, remove it
                blocks.pop(i)
            else:
                i += 1
    
    return blocks
