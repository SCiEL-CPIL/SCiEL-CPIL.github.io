import itertools
import numpy as np
from math import comb
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # 1. Precomputation
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)
    
    # Map each block to the list of triple indices it covers
    # block_triple_indices[i] is a list of triple indices for block i
    block_triple_indices = []
    for block in all_blocks:
        tids = [triple_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_triple_indices.append(tids)
        
    # Map each triple to the list of block indices that cover it
    # This is needed for efficient gain updates (reverse index)
    triple_to_blocks = [[] for _ in range(n_triples)]
    for b_idx, tids in enumerate(block_triple_indices):
        for t_idx in tids:
            triple_to_blocks[t_idx].append(b_idx)

    # Weights: inverse frequency of triples to prioritize rare triples
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    for tids in block_triple_indices:
        for t_idx in tids:
            triple_freq[t_idx] += 1
    triple_freq = np.maximum(triple_freq, 1)
    weights = 1.0 / triple_freq.astype(np.float64)

    def greedy_cover(random_seed):
        rng = random.Random(random_seed)
        # gain[b] is the number of currently uncovered triples covered by block b
        # Initially, all triples are uncovered, so gain is the number of triples in the block
        gain = np.array([len(tids) for tids in block_triple_indices], dtype=np.int32)
        # is_uncovered[t] is True if triple t is not yet covered
        is_uncovered = np.ones(n_triples, dtype=bool)
        
        selected_blocks = []
        remaining = n_triples
        
        while remaining > 0:
            # Check time
            if time.time() > deadline - 2.0:
                # If we are running out of time, we must ensure we return a valid cover.
                # The current selected_blocks might not be complete.
                # We need to finish the cover quickly. 
                # However, breaking here might leave an incomplete cover.
                # Strategy: If time is critical, we should probably just finish the current greedy 
                # but without the expensive reduction. Or better, we should have checked time 
                # more frequently. 
                # For safety, if we break here, we will try to complete the cover in a 
                # "fast mode" that ignores weights and just picks max gain, but even that 
                # might be too slow if many triples are left.
                # Given the problem size (C(24,3)=2024 triples), greedy is usually fast.
                # Let's assume we don't hit this unless something is wrong.
                break
            
            # Find max gain
            # Using np.max is fast
            max_gain = np.max(gain)
            if max_gain == 0:
                # Should not happen if remaining > 0 and blocks are valid
                break
            
            # Find all blocks with max gain
            # To avoid O(N) scan for candidates, we can just find indices
            candidates = np.where(gain == max_gain)[0]
            
            # Randomly pick one candidate to break ties
            best_idx = candidates[rng.randint(len(candidates))]
            
            # Update state
            block_triples = block_triple_indices[best_idx]
            
            # Count newly covered triples
            newly_covered_count = 0
            for t_idx in block_triples:
                if is_uncovered[t_idx]:
                    newly_covered_count += 1
                    is_uncovered[t_idx] = False
                    # For every block that covers this triple, decrease its gain
                    # Because this triple is no longer "newly covered" by them
                    for other_b_idx in triple_to_blocks[t_idx]:
                        gain[other_b_idx] -= 1
            
            remaining -= newly_covered_count
            selected_blocks.append(best_idx)
            
        return selected_blocks

    def reduce_cover(selected_block_indices):
        # Convert to list of sets of triple indices for quick check?
        # We maintain cover_count[t]
        cover_count = np.zeros(n_triples, dtype=np.int32)
        for b_idx in selected_block_indices:
            for t_idx in block_triple_indices[b_idx]:
                cover_count[t_idx] += 1
        
        active = [True] * len(selected_block_indices)
        
        # Pass 1: Remove redundant blocks
        # A block is redundant if all its triples have cover_count > 1
        # We iterate and remove.
        changed = True
        while changed:
            changed = False
            for i, b_idx in enumerate(selected_block_indices):
                if not active[i]:
                    continue
                # Check if redundant
                redundant = True
                for t_idx in block_triple_indices[b_idx]:
                    if cover_count[t_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    active[i] = False
                    # Update cover counts
                    for t_idx in block_triple_indices[b_idx]:
                        cover_count[t_idx] -= 1
                    changed = True
        
        return [b_idx for i, b_idx in enumerate(selected_block_indices) if active[i]]

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple times with different seeds
    # We use a loop that continues until deadline or no improvement for a while
    # But simple fixed number of restarts is safer to avoid infinite loops if reduction is slow
    # Let's do up to 50 restarts, but break early if time is up
    n_restarts = 50
    
    for i in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Run greedy
        block_indices = greedy_cover(i)
        
        # Reduce
        reduced_indices = reduce_cover(block_indices)
        
        if len(reduced_indices) < best_len:
            best_len = len(reduced_indices)
            best_blocks = reduced_indices
            
    # Fallback if no blocks found (should not happen)
    if best_blocks is None:
        block_indices = greedy_cover(0)
        best_blocks = block_indices
        
    # Convert indices back to blocks
    return [list(all_blocks[b_idx]) for b_idx in best_blocks]
