import itertools
import time
import random
import numpy as np
from collections import defaultdict, deque

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # 40s safety margin for 150s limit

    # Precompute all t-subsets and blocks as bitmasks
    # v=19, so bitmasks fit in Python int
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsubset_mask_to_idx = {}
    tsubset_masks = []
    for i, ts in enumerate(all_tsubsets):
        mask = 0
        for p in ts:
            mask |= (1 << p)
        tsubset_masks.append(mask)
        tsubset_mask_to_idx[mask] = i

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute which t-subset indices each block covers
    block_covered_ts_indices = []
    for blk in all_blocks:
        covered_indices = []
        # Iterate over all t-subsets of the block
        for ts in itertools.combinations(blk, t):
            ts_mask = 0
            for p in ts:
                ts_mask |= (1 << p)
            idx = tsubset_mask_to_idx[ts_mask]
            covered_indices.append(idx)
        block_covered_ts_indices.append(covered_indices)

    # Precompute ts_to_blocks
    ts_to_blocks = [[] for _ in range(n_tsubsets)]
    for b_idx, ts_list in enumerate(block_covered_ts_indices):
        for ts_idx in ts_list:
            ts_to_blocks[ts_idx].append(b_idx)

    def run_greedy_reduce(seed_val):
        random.seed(seed_val)
        
        # Greedy Phase
        # counts[i] = number of selected blocks covering t-subset i
        counts = np.zeros(n_tsubsets, dtype=np.int8)
        selected_block_indices = []
        remaining = set(range(n_tsubsets))
        
        # Maintain a priority queue of (gain, b_idx) for blocks
        # We need to efficiently update gains as remaining changes.
        # Since gains only decrease, we can use a lazy deletion heap.
        # However, gains are not monotonic per block in a simple way if we don't track them carefully.
        # Let's stick to the pivot heuristic but optimize the candidate selection.
        
        # Alternative: Use a "best gain" array and update it.
        # gain[b_idx] = number of remaining ts covered by b_idx.
        # Initially, gain[b_idx] = len(block_covered_ts_indices[b_idx]).
        # When a ts is covered, we decrement gain for all blocks covering it.
        # We can maintain a max-heap of (-gain, b_idx).
        
        # But we need to handle "stale" entries in the heap.
        
        # Let's try the Pivot strategy with a small optimization:
        # Instead of scanning all candidates for the pivot, we can pre-compute
        # or just scan. C(19,5)=11628 ts. Each ts is in C(14,3)=364 blocks.
        # So candidates list is ~364 items.
        # Scanning 364 items is fast.
        
        while remaining:
            if time.time() > deadline:
                break
            
            # Pick a pivot t-subset.
            # Strategy: Pick the one with the fewest candidates? (Most constrained)
            # Or random?
            # Let's try random for diversity.
            pivot_ts = random.choice(tuple(remaining))
            
            best_block_idx = -1
            best_gain = -1
            
            candidates = ts_to_blocks[pivot_ts]
            
            for b_idx in candidates:
                # Calculate gain
                gain = 0
                for ts_idx in block_covered_ts_indices[b_idx]:
                    if ts_idx in remaining:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                break
                
            selected_block_indices.append(best_block_idx)
            
            # Update counts and remaining
            for ts_idx in block_covered_ts_indices[best_block_idx]:
                counts[ts_idx] += 1
                if counts[ts_idx] == 1:
                    remaining.discard(ts_idx)
            
        # Reduction Phase
        # A block is redundant if all ts it covers have count > 1
        changed = True
        while changed:
            changed = False
            for i in range(len(selected_block_indices) - 1, -1, -1):
                b_idx = selected_block_indices[i]
                is_redundant = True
                for ts_idx in block_covered_ts_indices[b_idx]:
                    if counts[ts_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove block
                    selected_block_indices.pop(i)
                    for ts_idx in block_covered_ts_indices[b_idx]:
                        counts[ts_idx] -= 1
                    changed = True
                    if time.time() > deadline:
                        return selected_block_indices
        
        return selected_block_indices

    best_solution = None
    best_len = float('inf')
    
    # Multiple restarts
    # We have ~110s budget.
    # Each greedy run takes ~?
    # Let's estimate. 300 blocks * 364 candidates * 56 ts = 6M ops.
    # In Python, maybe 0.5s per run?
    # So we can do ~100 restarts.
    
    num_restarts = 50 # Conservative estimate to avoid timeout
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Use different seeds for diversity
        sol = run_greedy_reduce(i * 12345)
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol[:]
            
    # Convert block indices back to lists of points
    final_blocks = []
    if best_solution:
        for b_idx in best_solution:
            final_blocks.append(list(all_blocks[b_idx]))
    else:
        # Fallback: simple greedy without restarts if something went wrong
        # Or just return empty? No, must be valid.
        # Let's do one deterministic run
        sol = run_greedy_reduce(0)
        for b_idx in sol:
            final_blocks.append(list(all_blocks[b_idx]))

    return final_blocks
