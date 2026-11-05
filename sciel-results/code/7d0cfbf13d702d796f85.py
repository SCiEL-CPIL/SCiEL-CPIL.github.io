import itertools
import numpy as np
from math import comb
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)
    
    # Represent each block's triple set as a bitmask (Python int)
    block_masks = []
    for i, block in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(block, t):
            idx = triple_to_idx[sub]
            mask |= (1 << idx)
        block_masks.append(mask)
    
    # Precompute triple frequency for weighting
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    for mask in block_masks:
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            triple_freq[idx] += 1
            m ^= lsb
    
    # Inverse frequency weight: rarer triples get higher priority
    inv_freq = 1.0 / (triple_freq + 1).astype(np.float64)
    
    def greedy_cover_bitset(random_seed, use_weighting=True):
        """Fast greedy using bitmask representation."""
        rng = np.random.RandomState(random_seed)
        uncovered_mask_int = (1 << n_triples) - 1  # All triples uncovered
        selected_blocks = []
        remaining = n_triples
        
        while remaining > 0:
            if time.time() > deadline - 3:
                break
            
            # Calculate gains: number of uncovered triples in each block
            # For each block, gain = popcount(block_mask & uncovered_mask_int)
            # We need to find the block with max gain
            best_gain = -1
            best_candidates = []
            
            for i, bm in enumerate(block_masks):
                gain = (bm & uncovered_mask_int).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if best_gain <= 0:
                break
            
            if len(best_candidates) == 1:
                best_idx = best_candidates[0]
            else:
                if use_weighting:
                    # Weight by sum of inverse frequencies of uncovered triples
                    best_local = 0
                    best_weight = -1
                    for ci in best_candidates:
                        w = 0.0
                        m = block_masks[ci] & uncovered_mask_int
                        while m:
                            lsb = m & (-m)
                            idx = lsb.bit_length() - 1
                            w += inv_freq[idx]
                            m ^= lsb
                        w += rng.random() * 0.01
                        if w > best_weight:
                            best_weight = w
                            best_local = ci
                    best_idx = best_local
                else:
                    best_idx = best_candidates[rng.randint(len(best_candidates))]
            
            # Update state
            uncovered_mask_int &= ~block_masks[best_idx]
            remaining -= best_gain
            selected_blocks.append(best_idx)
            
        return selected_blocks
    
    def local_search_polish(selected_block_indices, max_iters=200):
        """Try to remove redundant blocks."""
        if not selected_block_indices:
            return selected_block_indices
        
        n_sel = len(selected_block_indices)
        # Build coverage count: for each triple, how many selected blocks cover it
        triple_count = np.zeros(n_triples, dtype=np.int32)
        for bi in selected_block_indices:
            m = block_masks[bi]
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                triple_count[idx] += 1
                m ^= lsb
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > deadline - 3:
                break
            improved = False
            iters += 1
            order = list(range(n_sel))
            np.random.RandomState(iters * 12345 + 7).shuffle(order)
            for pos in order:
                if pos >= n_sel:
                    continue
                bi = selected_block_indices[pos]
                m = block_masks[bi]
                # Check if any triple is only covered by this block
                only_covered = 0
                mm = m
                while mm:
                    lsb = mm & (-mm)
                    idx = lsb.bit_length() - 1
                    if triple_count[idx] == 1:
                        only_covered = 1
                        break
                    mm ^= lsb
                if only_covered == 0:
                    # Safe to remove
                    mm = m
                    while mm:
                        lsb = mm & (-mm)
                        idx = lsb.bit_length() - 1
                        triple_count[idx] -= 1
                        mm ^= lsb
                    selected_block_indices = [x for x in selected_block_indices if x != bi]
                    n_sel -= 1
                    improved = True
                    break
        return selected_block_indices
    
    def vns_perturb_and_reoptimize(selected_block_indices, rng, n_perturb=3):
        """VNS-style: perturb by swapping out a random block for a random other block, then re-polish."""
        if len(selected_block_indices) < 2:
            return selected_block_indices
        
        result = list(selected_block_indices)
        for _ in range(n_perturb):
            if time.time() > deadline - 5:
                break
            # Remove a random block
            pos = rng.randint(len(result))
            removed = result.pop(pos)
            # Add a random different block
            new_block = rng.randint(n_blocks)
            while new_block in result:
                new_block = rng.randint(n_blocks)
            result.append(new_block)
        
        # Re-polish
        result = local_search_polish(result, max_iters=100)
        return result
    
    # Run multiple greedy restarts with bitset (fast)
    best_blocks = None
    best_len = float('inf')
    
    n_restarts = 40
    
    for i in range(n_restarts):
        if time.time() > deadline - 10:
            break
        blocks_idx = greedy_cover_bitset(i, use_weighting=(i % 2 == 0))
        if len(blocks_idx) < best_len:
            best_len = len(blocks_idx)
            best_blocks = blocks_idx
    
    if best_blocks is None:
        best_blocks = greedy_cover_bitset(0, use_weighting=False)
        best_len = len(best_blocks)
    
    # Local search polish on the best solution
    if time.time() < deadline - 5:
        polished = local_search_polish(best_blocks, max_iters=300)
        if len(polished) < best_len:
            best_blocks = polished
            best_len = len(polished)
    
    # VNS-style perturbation and re-optimization
    rng = np.random.RandomState(42)
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        perturbed = vns_perturb_and_reoptimize(best_blocks, rng, n_perturb=2)
        if len(perturbed) < best_len:
            best_len = len(perturbed)
            best_blocks = perturbed
    
    # Final polish
    if time.time() < deadline - 2:
        polished = local_search_polish(best_blocks, max_iters=200)
        if len(polished) < best_len:
            best_blocks = polished
            best_len = len(polished)
    
    # Convert indices back to actual blocks
    return [list(all_blocks[i]) for i in best_blocks]
