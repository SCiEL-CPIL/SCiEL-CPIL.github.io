import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 115.0
    
    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # block_tsubs[i] = list of t-subset indices covered by block i
    block_tsubs = [None] * n_blocks
    # tsub_blocks[j] = list of block indices covering t-subset j
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    # tsub_masks[j] = bitmask of blocks covering t-subset j
    tsub_masks = [0] * n_tsubsets
    
    for bi, block in enumerate(all_blocks):
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs[bi] = tsubs
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)
            tsub_masks[ts_idx] |= (1 << bi)
            
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = [True] * n_tsubsets
        # gain[i] = number of uncovered t-subsets in block i
        gain = [0] * n_blocks
        for bi in range(n_blocks):
            gain[bi] = len(block_tsubs[bi])
            
        selected = []
        
        while True:
            if time.time() > deadline - 5:
                break
                
            # Find block with max gain
            max_gain = -1
            best_bi = -1
            candidates = []
            for bi in range(n_blocks):
                if gain[bi] > max_gain:
                    max_gain = gain[bi]
                    candidates = [bi]
                elif gain[bi] == max_gain and gain[bi] > 0:
                    candidates.append(bi)
                    
            if max_gain <= 0:
                break
                
            # Pick random from candidates
            bi = candidates[rng.randint(0, len(candidates) - 1)]
            selected.append(bi)
            
            # Update gains
            for ts_idx in block_tsubs[bi]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    # Decrement gain for all blocks covering this ts_idx
                    for bj in tsub_blocks[ts_idx]:
                        gain[bj] -= 1
                        
            gain[bi] = -1 # Prevent re-selection
            
        return selected

    def local_search(blocks, seed):
        rng = random.Random(seed)
        block_set = set(blocks)
        current_mask = 0
        for bi in block_set:
            current_mask |= (1 << bi)
            
        cover_count = [0] * n_tsubsets
        
        # Initialize cover_count
        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1
                
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Phase 1: Remove redundant blocks
            to_remove = []
            for bi in block_set:
                redundant = True
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(bi)
                    
            if to_remove:
                for bi in to_remove:
                    block_set.remove(bi)
                    current_mask &= ~(1 << bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                improved = True
                continue # Restart loop as structure changed significantly
            
            # Phase 2: Try swaps (1-for-1) using bitmasks
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            for bi in current_blocks:
                if time.time() > deadline - 2:
                    break
                if bi not in block_set:
                    continue
                    
                # Find unique subsets for bi
                unique_tsubs = []
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)
                        
                if not unique_tsubs:
                    # Redundant, should have been caught, but handle just in case
                    block_set.remove(bi)
                    current_mask &= ~(1 << bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break # Restart
                    
                # Find candidate blocks that cover all unique_tsubs
                # Intersection of masks
                cand_mask = (1 << n_blocks) - 1 # All 1s
                for ts_idx in unique_tsubs:
                    cand_mask &= tsub_masks[ts_idx]
                
                # Remove currently selected blocks
                cand_mask &= ~current_mask
                
                if cand_mask == 0:
                    continue
                    
                # Iterate through set bits in cand_mask
                # Get list of candidate indices
                # We can just pick a random one or iterate
                # To avoid bias, we can collect them or just pick one
                # Picking one is faster. Let's pick a random one from the candidates.
                # How to pick random bit?
                # Convert to list of indices? That's slow if many candidates.
                # But candidates are usually few.
                # Let's just iterate and pick the first one or random.
                # For now, let's collect them if count is small, else pick first.
                
                # Efficient way to get indices from bitmask:
                # while cand_mask:
                #   lsb = cand_mask & -cand_mask
                #   idx = lsb.bit_length() - 1
                #   ...
                #   cand_mask ^= lsb
                
                # Let's just pick a random candidate.
                # We can store candidates in a list if it's small.
                cands = []
                temp_mask = cand_mask
                while temp_mask:
                    lsb = temp_mask & -temp_mask
                    idx = lsb.bit_length() - 1
                    cands.append(idx)
                    temp_mask ^= lsb
                    
                if not cands:
                    continue
                    
                # Try a random candidate
                bj = cands[rng.randint(0, len(cands) - 1)]
                
                # Perform swap
                block_set.remove(bi)
                current_mask &= ~(1 << bi)
                for ts_idx in block_tsubs[bi]:
                    cover_count[ts_idx] -= 1
                    
                block_set.add(bj)
                current_mask |= (1 << bj)
                for ts_idx in block_tsubs[bj]:
                    cover_count[ts_idx] += 1
                    
                improved = True
                break # Restart while loop
                    
        return list(block_set)

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy starts
    num_trials = 100
    for i in range(num_trials):
        if time.time() > deadline - 20:
            break
            
        blocks = greedy_cover(i)
        if not blocks:
            continue
            
        # Local search
        blocks = local_search(blocks, i)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    # If we have time, try to improve the best solution further
    if best_blocks is not None and time.time() < deadline - 10:
        # Run local search again on the best
        best_blocks = local_search(best_blocks, 99999)
        
    # Fallback if something went wrong
    if best_blocks is None:
        best_blocks = greedy_cover(0)
        
    return [all_blocks[i] for i in best_blocks]
