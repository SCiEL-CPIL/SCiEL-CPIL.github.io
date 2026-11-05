"""Optimized covering design C(24,7,3) solver using vectorized greedy + local search."""
import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 115s safety margin within 120s limit
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute coverage matrix: for each point, which t-subsets it can cover
    # coverage[x][ts_idx] = 1 if point x can be part of a block covering ts_idx
    # More precisely: for each point x, the t-subsets that contain x
    point_covers = np.zeros((v, n_tsubsets), dtype=np.int8)
    for x in range(v):
        for ts_idx, ts in enumerate(all_tsubsets):
            if x in ts:
                point_covers[x][ts_idx] = 1
    
    # Greedy set cover with rarity heuristic
    uncovered = np.ones(n_tsubsets, dtype=np.int8)
    blocks = []
    
    while np.any(uncovered):
        if time.time() > deadline - 5:
            break
        
        # Find the rarest uncovered t-subset (fewest points can cover it)
        # For each uncovered t-subset, count how many points can cover it
        # A point x can cover ts_idx if x is in the t-subset
        # So the number of points that can cover ts_idx is exactly t (the size of the subset)
        # Instead, we use a different rarity: how many blocks could cover it given current coverage
        
        # Better approach: find uncovered t-subset with minimum "coverage potential"
        # For each uncovered ts, count how many points in ts are not yet in any block
        # Actually, let's use a simpler but effective heuristic:
        # Pick the uncovered t-subset that has the fewest "available" points to extend a block
        
        # Simplest effective approach: pick random uncovered t-subset as anchor
        uncovered_indices = np.where(uncovered == 1)[0]
        if len(uncovered_indices) == 0:
            break
        first_uncovered = random.choice(uncovered_indices.tolist())
        
        anchor = all_tsubsets[first_uncovered]
        block = set(anchor)
        
        # Greedily add points that cover the most uncovered t-subsets
        while len(block) < k:
            best_point = None
            best_gain = -1
            
            # Vectorized gain computation
            block_set = set(block)
            gains = np.zeros(v, dtype=np.int32)
            
            for x in range(v):
                if x in block_set:
                    continue
                
                # Count uncovered t-subsets that x can complete given current block
                # x can cover ts_idx if:
                # 1. uncovered[ts_idx] == 1
                # 2. x is in ts_idx (point_covers[x][ts_idx] == 1)
                # 3. All other (t-1) elements of ts_idx are in block
                # Condition 3: for ts = (a, b, c), if x=a, then b,c must be in block
                
                # Compute mask of t-subsets where all elements except x are in block
                # This is expensive, so let's do it differently
                pass
            
            # Non-vectorized but correct approach
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                for ts_idx in np.where(point_covers[x] == 1)[0]:
                    if uncovered[ts_idx] == 0:
                        continue
                    ts = all_tsubsets[ts_idx]
                    # Check if all elements of ts except x are in block
                    rest = [p for p in ts if p != x]
                    if all(p in block_set for p in rest):
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None or best_gain <= 0:
                candidates = [x for x in range(v) if x not in block_set]
                if candidates:
                    best_point = random.choice(candidates)
                else:
                    break
            
            block.add(best_point)
        
        block = sorted(block)
        blocks.append(block)
        
        # Mark covered t-subsets
        for ts in itertools.combinations(block, t):
            idx = tsubset_to_idx[tuple(ts)]
            uncovered[idx] = 0
    
    # Local search: remove redundant blocks
    if time.time() > deadline - 10:
        return blocks
    
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            
            removable = True
            block_i = blocks[i]
            for ts in itertools.combinations(block_i, t):
                idx = tsubset_to_idx[tuple(ts)]
                if uncovered[idx] == 0:
                    found_in_other = False
                    for j in range(len(blocks)):
                        if j == i:
                            continue
                        if all(p in blocks[j] for p in ts):
                            found_in_other = True
                            break
                    if not found_in_other:
                        removable = False
                        break
            
            if removable:
                blocks.pop(i)
                changed = True
    
    return blocks
