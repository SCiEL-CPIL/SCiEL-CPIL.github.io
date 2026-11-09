import numpy as np
from itertools import combinations
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Precompute all t-subsets
    all_t_subs = list(combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Initialize coverage count for each t-subset
    coverage = np.zeros(n_t, dtype=np.int32)
    
    blocks = []
    
    # Precompute for each point, which t-subsets it participates in
    # point_t_subs[x] = list of indices of t-subsets containing x
    point_t_subs = [[] for _ in range(v)]
    for i, sub in enumerate(all_t_subs):
        for x in sub:
            point_t_subs[x].append(i)
    
    # Greedy construction
    while True:
        if time.time() > deadline:
            break
        
        # Find the most uncovered t-subset (one with coverage 0, prefer lowest index for determinism)
        uncovered_mask = coverage == 0
        if not uncovered_mask.any():
            break
        
        # Pick the first uncovered t-subset
        target_idx = np.argmax(uncovered_mask)
        target_sub = all_t_subs[target_idx]
        
        # Build a block starting with the target subset
        block_set = set(target_sub)
        
        # Greedily add points to maximize coverage of currently uncovered t-subsets
        while len(block_set) < k:
            if time.time() > deadline:
                break
            
            best_point = None
            best_gain = -1
            
            candidates = [x for x in range(v) if x not in block_set]
            
            for x in candidates:
                # Compute how many currently uncovered t-subsets would be covered by adding x
                # A t-subset is covered by block ∪ {x} if it's a subset of block ∪ {x}
                # and contains x (since it wasn't covered by block alone)
                # Equivalently: t-subsets that contain x and are subsets of block ∪ {x}
                # These are t-subsets of form (t-1 subset of block) ∪ {x}
                
                gain = 0
                # For each (t-1)-subset of block, check if combined with x forms an uncovered t-subset
                for rest in combinations(block_set, t - 1):
                    candidate_sub = tuple(sorted(rest + (x,)))
                    idx = t_sub_idx.get(candidate_sub)
                    if idx is not None and coverage[idx] == 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None:
                break
            block_set.add(best_point)
        
        if len(block_set) < k:
            # If we couldn't fill the block (shouldn't happen normally), skip
            break
        
        block = sorted(block_set)
        blocks.append(block)
        
        # Update coverage
        for sub in combinations(block, t):
            idx = t_sub_idx[sub]
            coverage[idx] += 1
    
    return blocks
