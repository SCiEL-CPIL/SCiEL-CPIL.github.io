import itertools
import time
import random
import numpy as np
from collections import defaultdict
import heapq

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
    
    for bi, block in enumerate(all_blocks):
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs[bi] = tsubs
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)

    # Convert block_tsubs to numpy arrays for faster slicing? 
    # Actually, lists are fine for iteration if we don't do vectorized math on them directly in the inner loop.
    # But for cover_count updates, we can use numpy if we represent incidence as sparse or dense.
    # Dense incidence matrix: n_blocks x n_tsubsets (50388 x 3876) ~ 195M entries. 
    # If bool, that's ~200MB. Might be okay, but sparse is better.
    # Let's stick to lists but optimize the loops.

    def greedy_cover(seed, noise_factor=0.0):
        rng = random.Random(seed)
        uncovered = [True] * n_tsubsets
        # gain[i] = number of uncovered t-subsets in block i
        gain = [0] * n_blocks
        for bi in range(n_blocks):
            gain[bi] = len(block_tsubs[bi])
            
        selected = []
        
        # Use a heap for max-gain selection? 
        # Heap supports efficient max extraction. 
        # But we need to update gains frequently. 
        # Lazy deletion in heap is standard.
        
        # Actually, for n_blocks=50k, a linear scan is ~50k ops. 
        # Number of blocks selected is ~150. 
        # 150 * 50k = 7.5M ops. This is fast in Python if the inner loop is tight.
        # Heap might be slower due to overhead of pushing 50k items initially and updating.
        # Let's stick to linear scan but optimize it.
        
        while True:
            if time.time() > deadline - 5:
                break
                
            max_gain = -1
            best_bi = -1
            candidates = []
            
            # Random tie-breaking
            for bi in range(n_blocks):
                g = gain[bi]
                if noise_factor > 0:
                    g += rng.uniform(0, noise_factor)
                if g > max_gain:
                    max_gain = g
                    candidates = [bi]
                elif g == max_gain and g > 0:
                    candidates.append(bi)
                    
            if max_gain <= 0:
                break
                
            bi = candidates[rng.randint(0, len(candidates) - 1)]
            selected.append(bi)
            
            # Update gains
            for ts_idx in block_tsubs[bi]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    for bj in tsub_blocks[ts_idx]:
                        gain[bj] -= 1
                        
            gain[bi] = -1 # Prevent re-selection
            
        return selected

    def local_search(blocks, seed):
        rng = random.Random(seed)
        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        
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
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                improved = True
                continue
            
            # Phase 2: Try swaps
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            for bi in current_blocks:
                if time.time() > deadline - 2:
                    break
                if bi not in block_set:
                    continue
                    
                unique_tsubs = []
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)
                        
                if not unique_tsubs:
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break
                    
                # Find candidates covering all unique_tsubs
                min_ts = min(unique_tsubs, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_blocks[min_ts])
                
                for ts_idx in unique_tsubs[1:]:
                    if not candidates:
                        break
                    candidates &= set(tsub_blocks[ts_idx])
                
                candidates.discard(bi)
                
                if not candidates:
                    continue
                
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                swapped = False
                for bj in cand_list[:10]:
                    if bj in block_set:
                        continue
                        
                    # Swap is valid if bj covers all unique_tsubs
                    # We already checked intersection.
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                        
                    block_set.add(bj)
                    for ts_idx in block_tsubs[bj]:
                        cover_count[ts_idx] += 1
                        
                    improved = True
                    swapped = True
                    break
                    
                if swapped:
                    break
                    
        return list(block_set)

    def two_block_swap_search(blocks, seed):
        rng = random.Random(seed)
        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        
        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1
                
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            n_checked = 0
            for i, bi in enumerate(current_blocks):
                if time.time() > deadline - 2:
                    break
                if bi not in block_set:
                    continue
                    
                if len(block_set) < 2:
                    break
                others = [b for b in block_set if b != bi]
                if not others:
                    continue
                bj = rng.choice(others)
                
                n_checked += 1
                if n_checked > 2000:
                    break
                
                req_bi = [ts for ts in block_tsubs[bi] if cover_count[ts] == 1]
                req_bj = [ts for ts in block_tsubs[bj] if cover_count[ts] == 1]
                
                set_bi = set(block_tsubs[bi])
                set_bj = set(block_tsubs[bj])
                
                required = set(req_bi)
                required.update(req_bj)
                
                inter = set_bi & set_bj
                for ts in inter:
                    if cover_count[ts] == 2:
                        required.add(ts)
                
                if not required:
                    continue
                
                min_ts = min(required, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_blocks[min_ts])
                
                for ts in required:
                    if ts == min_ts:
                        continue
                    if not candidates:
                        break
                    candidates &= set(tsub_blocks[ts])
                    
                candidates.discard(bi)
                candidates.discard(bj)
                
                if not candidates:
                    continue
                
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                swapped = False
                for bk in cand_list[:5]:
                    if bk in block_set:
                        continue
                        
                    block_set.remove(bi)
                    for ts in block_tsubs[bi]:
                        cover_count[ts] -= 1
                    block_set.remove(bj)
                    for ts in block_tsubs[bj]:
                        cover_count[ts] -= 1
                    block_set.add(bk)
                    for ts in block_tsubs[bk]:
                        cover_count[ts] += 1
                        
                    improved = True
                    swapped = True
                    break
                    
                if swapped:
                    break
                    
        return list(block_set)

    best_blocks = None
    best_len = float('inf')
    
    # Dynamic number of trials
    # Estimate time per trial
    start_time = time.time()
    
    # Run a few initial trials to estimate speed
    # Or just run as many as fit in 100s
    
    trial_count = 0
    while time.time() < deadline - 40:
        trial_count += 1
        seed = trial_count
        # Add noise to greedy for diversity
        noise = 0.1 * (trial_count % 5)
        blocks = greedy_cover(seed, noise_factor=noise)
        if not blocks:
            continue
            
        blocks = local_search(blocks, seed)
        
        if time.time() < deadline - 20:
            blocks = two_block_swap_search(blocks, seed + 1000)
            
        if time.time() < deadline - 10:
            # Quick SA or just another local search pass
            blocks = local_search(blocks, seed + 2000)
            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        # If we find a very good solution, we can reduce noise or focus on polishing
        if best_len <= 152:
             # Try to polish further
             pass

    if best_blocks is not None and time.time() < deadline - 5:
        best_blocks = local_search(best_blocks, 99999)
        if time.time() < deadline - 2:
            best_blocks = two_block_swap_search(best_blocks, 88888)

    if best_blocks is None:
        best_blocks = greedy_cover(0)
        
    return [all_blocks[i] for i in best_blocks]
