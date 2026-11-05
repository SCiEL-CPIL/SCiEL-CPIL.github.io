import itertools
import numpy as np
from math import comb
import time

def construct(v, k, t):
    # Deadline guard: must finish within 150s, use 135s as safety margin
    deadline = time.time() + 135.0

    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)
    
    # Store block indices for fast access
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            idx = triple_to_idx[sub]
            block_triple_indices[i, j] = idx
            triple_freq[idx] += 1
            
    # Weights: inverse frequency to prioritize rare triples
    triple_freq = np.maximum(triple_freq, 1)
    weights = 1.0 / triple_freq.astype(np.float64)
    
    def greedy_cover(random_seed):
        rng = np.random.RandomState(random_seed)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        remaining = n_triples
        
        while remaining > 0:
            # Check deadline
            if time.time() > deadline - 5.0:
                break
                
            # Calculate weighted gains
            active_weights = np.where(uncovered_mask, weights, 0.0)
            gains = np.sum(active_weights[block_triple_indices], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly pick one candidate to break ties
            best_idx = candidates[rng.randint(len(candidates))]
            
            # Update state
            block_triples = block_triple_indices[best_idx]
            newly_covered = np.sum(uncovered_mask[block_triples])
            remaining -= newly_covered
            uncovered_mask[block_triples] = False
            selected_blocks.append(list(all_blocks[best_idx]))
            
        return selected_blocks

    def local_search(blocks, random_seed):
        """Try to improve a covering by swapping blocks."""
        rng = np.random.RandomState(random_seed)
        if not blocks:
            return blocks
        
        # Build coverage count: how many blocks cover each triple
        triple_cover_count = np.zeros(n_triples, dtype=np.int32)
        for block in blocks:
            for sub in itertools.combinations(block, t):
                triple_cover_count[triple_to_idx[sub]] += 1
        
        # Count how many times each block index is used
        block_used = np.zeros(n_blocks, dtype=bool)
        for block in blocks:
            # Find the index of this block
            # This is slow; we need a better way
            pass
        
        # Actually, let's track block indices during greedy
        # For now, return blocks unchanged if we can't efficiently do local search
        return blocks

    # Run multiple times with different seeds to find a better covering
    best_blocks = None
    best_len = float('inf')
    
    # Adaptive restarts: run until deadline
    n_restarts = 0
    for i in range(100):  # Max 100 restarts
        if time.time() > deadline - 10.0:
            break
        if best_blocks is not None and len(best_blocks) <= 70:
            # If we're already at or below known bound, stop early
            break
            
        blocks = greedy_cover(i)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            n_restarts += 1
        elif n_restarts > 0 and (i % 10 == 0):
            # Every 10 iterations without improvement, try a different strategy
            pass
            
    return best_blocks
