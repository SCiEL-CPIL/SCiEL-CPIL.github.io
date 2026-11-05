import itertools
import numpy as np
from math import comb
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

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
    
    # Precompute triple occurrence frequency across all blocks
    # freq[triple_idx] = number of blocks containing that triple
    triple_occurrence = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_occurrence[block_triple_indices[i]] += 1
    # Inverse frequency weight: rarer triples get higher priority
    inv_freq = 1.0 / (triple_occurrence + 1).astype(np.float64)
    
    def greedy_cover(random_seed, use_weighting=True):
        rng = np.random.RandomState(random_seed)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        remaining = n_triples
        
        while remaining > 0:
            if time.time() > deadline - 2:
                break
            
            # Calculate gains: number of uncovered triples in each block
            gains = np.sum(uncovered_mask[block_triple_indices], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            if len(candidates) == 1:
                best_idx = candidates[0]
            else:
                if use_weighting:
                    # Weight candidates by sum of inverse frequencies of their uncovered triples
                    cand_triples = block_triple_indices[candidates]
                    # Only count uncovered triples in each candidate
                    cand_uncovered = uncovered_mask[cand_triples]
                    # Weight = sum of inv_freq for uncovered triples in candidate
                    weights = np.sum(inv_freq[cand_triples] * cand_uncovered, axis=1)
                    # Add small random noise for diversity
                    weights += rng.random(len(candidates)) * 0.01
                    best_local = np.argmax(weights)
                    best_idx = candidates[best_local]
                else:
                    best_idx = candidates[rng.randint(len(candidates))]
            
            # Update state
            block_idx = block_triple_indices[best_idx]
            uncovered_mask[block_idx] = False
            remaining -= max_gain
            selected_blocks.append(best_idx)
            
        return selected_blocks
    
    def local_search_polish(selected_block_indices, max_iters=500):
        """Try to remove blocks by checking if their triples are covered by others."""
        if not selected_block_indices:
            return selected_block_indices
        
        n_sel = len(selected_block_indices)
        # Build coverage count: for each triple, how many selected blocks cover it
        triple_count = np.zeros(n_triples, dtype=np.int32)
        for bi in selected_block_indices:
            triple_count[block_triple_indices[bi]] += 1
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > deadline - 2:
                break
            improved = False
            iters += 1
            # Try removing each block (in random order for diversity)
            order = list(range(n_sel))
            np.random.RandomState(iters * 12345 + 7).shuffle(order)
            for pos in order:
                if pos >= n_sel:
                    continue
                bi = selected_block_indices[pos]
                # Check if removing this block leaves any triple uncovered
                bi_triples = block_triple_indices[bi]
                # A triple is only covered if count > 1 (since we'd remove one)
                only_covered_by_this = np.sum(triple_count[bi_triples] == 1)
                if only_covered_by_this == 0:
                    # Safe to remove
                    triple_count[bi_triples] -= 1
                    selected_block_indices = [x for x in selected_block_indices if x != bi]
                    n_sel -= 1
                    improved = True
                    # Restart the loop since indices changed
                    break
        return selected_block_indices
    
    # Run multiple times with different seeds to find a better covering
    best_blocks = None
    best_len = float('inf')
    
    n_restarts = 30
    
    for i in range(n_restarts):
        if time.time() > deadline - 5:
            break
        blocks_idx = greedy_cover(i, use_weighting=(i % 2 == 0))
        if len(blocks_idx) < best_len:
            best_len = len(blocks_idx)
            best_blocks = blocks_idx
    
    if best_blocks is None:
        # Fallback: pure greedy
        best_blocks = greedy_cover(0, use_weighting=False)
        best_len = len(best_blocks)
    
    # Local search polish on the best solution
    if time.time() < deadline - 3:
        polished = local_search_polish(best_blocks)
        if len(polished) < best_len:
            best_blocks = polished
            best_len = len(best_blocks)
    
    # Convert indices back to actual blocks
    return [list(all_blocks[i]) for i in best_blocks]
