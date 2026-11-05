import itertools
import numpy as np
from math import comb

def construct(v, k, t):
    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)
    
    # Store block indices for fast access
    # block_triple_indices[i, j] is the index of the j-th triple in block i
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_triple_indices[i, j] = triple_to_idx[sub]
            
    # Calculate frequency of each triple (how many blocks cover it)
    triple_freq = np.zeros(n_triples, dtype=np.float64)
    for i in range(n_blocks):
        for j in range(k_t):
            triple_freq[block_triple_indices[i, j]] += 1.0
            
    # Weights: inverse frequency. Harder triples (lower freq) get higher weight.
    # This biases the greedy choice towards blocks that cover "rare" triples,
    # which are more critical to cover early.
    triple_weights = 1.0 / (triple_freq + 1e-6)
    
    def greedy_cover(random_seed):
        rng = np.random.RandomState(random_seed)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        remaining = n_triples
        
        while remaining > 0:
            # Calculate weighted gains
            # eff_weights is the weight of each triple, 0 if covered
            eff_weights = triple_weights * uncovered_mask
            # gains[i] is the sum of weights of uncovered triples in block i
            gains = np.sum(eff_weights[block_triple_indices], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly pick one candidate to break ties randomly
            best_idx = candidates[rng.randint(len(candidates))]
            
            # Update state
            block_idx = block_triple_indices[best_idx]
            newly_covered = uncovered_mask[block_idx]
            remaining -= np.sum(newly_covered)
            uncovered_mask[block_idx] = False
            selected_blocks.append(list(all_blocks[best_idx]))
            
        return selected_blocks

    # Run multiple times with different seeds to find a better covering
    best_blocks = None
    best_len = float('inf')
    
    # 30 restarts provides a good balance. 
    # The weighted greedy is deterministic per seed (except tie-breaking).
    n_restarts = 30
    
    for i in range(n_restarts):
        blocks = greedy_cover(i)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks
