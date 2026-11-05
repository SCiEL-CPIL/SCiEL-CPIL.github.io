import itertools
import numpy as np
from math import comb
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130
    
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)
    
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_triple_indices[i, j] = triple_to_idx[sub]
            
    # Precompute scarcity
    triple_scarcity = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_scarcity[block_triple_indices[i]] += 1
    
    # Weight: prefer covering rare triples
    # Use 1/sqrt(scarcity) as a smooth weight
    scarcity_weight = 1.0 / np.sqrt(triple_scarcity.astype(np.float64))
    # Normalize for numerical stability
    scarcity_weight = scarcity_weight / scarcity_weight.max()
    
    def greedy_cover(random_seed, mode='scarcity'):
        rng = np.random.RandomState(random_seed)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        remaining = n_triples
        
        while remaining > 0:
            if time.time() > deadline:
                break
            
            gains = np.sum(uncovered_mask[block_triple_indices], axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            
            if mode == 'scarcity' and len(candidates) > 1:
                # Among max-gain candidates, score by total scarcity weight of uncovered triples
                cand_scores = np.sum(
                    uncovered_mask[block_triple_indices[candidates]] * 
                    scarcity_weight[block_triple_indices[candidates]], 
                    axis=1
                )
                # Add small random perturbation to break ties
                cand_scores += rng.random(len(candidates)) * 0.01
                best_local = np.argmax(cand_scores)
                best_idx = candidates[best_local]
            else:
                best_idx = candidates[rng.randint(len(candidates))]
            
            block_idx = block_triple_indices[best_idx]
            newly_covered = np.sum(uncovered_mask[block_idx])
            uncovered_mask[block_idx] = False
            remaining -= newly_covered
            selected_blocks.append(best_idx)
            
        # Remove redundant blocks iteratively
        if selected_blocks:
            cover_count = np.zeros(n_triples, dtype=np.int32)
            for b_idx in selected_blocks:
                cover_count[block_triple_indices[b_idx]] += 1
            
            changed = True
            while changed:
                changed = False
                to_remove = []
                for b_idx in selected_blocks:
                    if not np.any(cover_count[block_triple_indices[b_idx]] == 1):
                        to_remove.append(b_idx)
                
                if to_remove:
                    changed = True
                    for b_idx in to_remove:
                        cover_count[block_triple_indices[b_idx]] -= 1
                    selected_blocks = [b for b in selected_blocks if b not in to_remove]
                    
        return [list(all_blocks[b_idx]) for b_idx in selected_blocks]

    best_blocks = None
    best_len = float('inf')
    
    n_restarts = 40
    
    for i in range(n_restarts):
        if time.time() > deadline - 5:
            break
        mode = 'scarcity' if (i % 2 == 0) else 'random'
        blocks = greedy_cover(i, mode=mode)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks
