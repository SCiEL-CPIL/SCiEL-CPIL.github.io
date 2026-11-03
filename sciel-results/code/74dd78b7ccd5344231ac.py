"""Improved Greedy Set Cover with Lazy Heap and Randomized Restarts."""
import itertools
import random
import heapq
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
    tsub_to_blocks = defaultdict(list)
    # For each block, list of t-subsets it contains
    block_to_tsubs = []
    
    for b_idx, block in enumerate(all_blocks):
        tsubs_in_block = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            tsub_to_blocks[idx].append(b_idx)
            tsubs_in_block.append(idx)
        block_to_tsubs.append(tsubs_in_block)
        
    init_gain = comb(k, t)
    
    def run_greedy(seed):
        rng = random.Random(seed)
        
        # Current state
        covered = np.zeros(n_tsubs, dtype=bool)
        # current_gain[i] is the true current gain of block i
        current_gain = np.full(n_blocks, init_gain, dtype=int)
        selected = set()
        selected_list = []
        
        # Heap entries: (-gain, random_tie_breaker, block_idx)
        # We use negative gain for max-heap behavior via min-heap
        # random_tie_breaker helps diversify solutions when gains are equal
        heap = []
        for b_idx in range(n_blocks):
            tie = rng.random()
            heapq.heappush(heap, (-init_gain, tie, b_idx))
            
        # Pre-compute a list of blocks to iterate for updates?
        # We need to update gains for all blocks sharing a t-subset with the newly covered one.
        
        while heap:
            # Pop until we find a valid, non-stale entry
            neg_gain, tie, b_idx = heapq.heappop(heap)
            gain_val = -neg_gain
            
            # Check if block is already selected
            if b_idx in selected:
                continue
                
            # Check if the gain in heap is stale (less than current true gain)
            # Note: Gains only decrease, so if heap_gain < current_gain, it's stale.
            # Actually, we push new entries when gain decreases.
            # So if the popped gain is less than current_gain[b_idx], it's an old entry.
            if gain_val < current_gain[b_idx]:
                continue
            
            # This is the best block
            selected.add(b_idx)
            selected_list.append(b_idx)
            
            # Update coverage and gains for all t-subsets in this block
            for ts_idx in block_to_tsubs[b_idx]:
                if not covered[ts_idx]:
                    covered[ts_idx] = True
                    # Decrease gain for all blocks containing this t-subset
                    for other_b_idx in tsub_to_blocks[ts_idx]:
                        if other_b_idx not in selected:
                            current_gain[other_b_idx] -= 1
                            # Push new entry with updated gain
                            new_tie = rng.random()
                            heapq.heappush(heap, (-current_gain[other_b_idx], new_tie, other_b_idx))
            
            # Check if all t-subsets are covered
            if np.all(covered):
                break
        
        return selected_list

    # Run multiple restarts to find a good solution
    best_solution = None
    best_len = float('inf')
    
    # Increase restarts since heap is faster
    # 120s limit. Let's try 50-100 restarts.
    n_restarts = 100
    
    for i in range(n_restarts):
        selected_indices = run_greedy(i * 1000 + 7)
        if len(selected_indices) < best_len:
            best_len = len(selected_indices)
            best_solution = selected_indices
            # Optional: break early if we hit a theoretical lower bound? 
            # Lower bound is 264. If we hit 264, we can stop.
            if best_len <= 264:
                break

    # Convert indices back to blocks
    if best_solution is None:
        return []
        
    return [list(all_blocks[i]) for i in best_solution]
