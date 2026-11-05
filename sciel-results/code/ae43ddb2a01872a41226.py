import itertools
import numpy as np
from math import comb
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)
    
    # Store block indices for fast access
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_triple_indices[i, j] = triple_to_idx[sub]
            
    # Precompute scarcity: how many blocks cover each triple
    # Use bincount for efficiency
    flat_indices = block_triple_indices.ravel()
    triple_scarcity = np.bincount(flat_indices, minlength=n_triples).astype(np.float64)
    # Normalize scarcity to [0,1] range for use in scoring
    max_scarcity = triple_scarcity.max() if n_triples > 0 else 1
    triple_scarcity_norm = triple_scarcity / max_scarcity if max_scarcity > 0 else np.zeros(n_triples)
    
    def greedy_cover(random_seed, use_scarcity=True):
        rng = np.random.RandomState(random_seed)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        remaining = n_triples
        
        while remaining > 0:
            if time.time() > deadline - 5.0:
                break
            
            # Calculate gains
            gains = np.sum(uncovered_mask[block_triple_indices], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            
            if len(candidates) > 1:
                if use_scarcity:
                    # Score candidates by scarcity of triples they cover
                    # Prefer blocks that cover rare triples
                    cand_triples = block_triple_indices[candidates]
                    # Only consider uncovered triples in scarcity calculation
                    cand_scarcity_scores = np.sum(
                        triple_scarcity_norm[cand_triples] * uncovered_mask[cand_triples], 
                        axis=1
                    )
                    # Add small random perturbation to break ties
                    perturbation = rng.random(len(candidates)) * 0.01
                    scores = cand_scarcity_scores + perturbation
                    best_local = np.argmax(scores)
                else:
                    best_local = rng.randint(len(candidates))
            else:
                best_local = 0
                
            best_idx = candidates[best_local]
            
            # Update state
            block_idx = block_triple_indices[best_idx]
            uncovered_mask[block_idx] = False
            remaining -= max_gain
            selected_blocks.append(best_idx)
            
        # Post-processing: Remove redundant blocks
        if selected_blocks:
            cover_count = np.zeros(n_triples, dtype=np.int32)
            for b_idx in selected_blocks:
                cover_count[block_triple_indices[b_idx]] += 1
            
            # Iterative removal of redundant blocks
            changed = True
            while changed and time.time() < deadline - 5.0:
                changed = False
                new_selected = []
                for b_idx in selected_blocks:
                    block_triples = block_triple_indices[b_idx]
                    if not np.any(cover_count[block_triples] == 1):
                        # This block is redundant, remove it
                        cover_count[block_triples] -= 1
                        changed = True
                    else:
                        new_selected.append(b_idx)
                selected_blocks = new_selected
                
        return [list(all_blocks[b_idx]) for b_idx in selected_blocks]
    
    # Run multiple times with different strategies
    best_blocks = None
    best_len = float('inf')
    
    n_restarts = 30
    
    for i in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Alternate between scarcity-based and random tie-breaking
        use_scarcity = (i % 2 == 0)
        blocks = greedy_cover(i, use_scarcity)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    # If we have time, try to improve the best solution by swapping blocks
    if best_blocks is not None and time.time() < deadline - 15.0:
        # Try local search: replace one block with another to reduce total count
        # This is complex, so we'll just do more restarts with different seeds
        for i in range(n_restarts, n_restarts + 20):
            if time.time() > deadline - 10.0:
                break
            blocks = greedy_cover(i, use_scarcity=True)
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
                
    return best_blocks
