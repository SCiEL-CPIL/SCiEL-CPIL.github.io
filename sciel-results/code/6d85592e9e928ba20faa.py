"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import math


def _precompute(v, k, t):
    """Precompute masks for all k-blocks, the blocks themselves, and mapping from t-subsets to blocks."""
    num_t = math.comb(v, t)
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    block_masks = []
    blocks = []
    t_subset_to_blocks = [[] for _ in range(num_t)]
    
    for block in itertools.combinations(range(v), k):
        mask = 0
        for ts in itertools.combinations(block, t):
            ts_sorted = tuple(sorted(ts))
            idx = t_subset_to_idx[ts_sorted]
            mask |= (1 << idx)
            t_subset_to_blocks[idx].append(len(blocks))
        block_masks.append(mask)
        blocks.append(block)
        
    return block_masks, t_subset_to_blocks, t_subsets, blocks


def _greedy_cover_fast(v, k, t, block_masks, t_subset_to_blocks, num_t, rng, deadline):
    """Fast greedy using precomputed masks and candidate filtering."""
    all_uncovered = (1 << num_t) - 1
    uncovered = all_uncovered
    selected = []
    selected_set = set()
    
    while uncovered and time.time() < deadline:
        # Pick a random uncovered t-subset index
        while True:
            idx = rng.randint(0, num_t - 1)
            if (uncovered >> idx) & 1:
                break
                
        candidates = t_subset_to_blocks[idx]
        
        best_block = None
        best_gain = -1
        
        for b_idx in candidates:
            if b_idx in selected_set:
                continue
            gain = (block_masks[b_idx] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
                
        if best_block is None:
            break
            
        selected.append(best_block)
        selected_set.add(best_block)
        uncovered &= ~block_masks[best_block]
        
    return selected


def _local_search_fast(selected, v, k, t, block_masks, t_subset_to_blocks, num_t, rng, deadline):
    """Local search using masks: remove redundant, swap to improve structure."""
    selected = list(selected)
    selected_set = set(selected)
    
    coverage = [0] * num_t
    for b_idx in selected:
        mask = block_masks[b_idx]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            coverage[idx] += 1
            m ^= lsb
            
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Remove redundant blocks
        for i in range(len(selected)):
            if time.time() >= deadline:
                break
            b_idx = selected[i]
            mask = block_masks[b_idx]
            
            is_redundant = True
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if coverage[idx] > 1:
                    is_redundant = False
                    break
                m ^= lsb
                
            if is_redundant:
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    coverage[idx] -= 1
                    m ^= lsb
                    
                selected.pop(i)
                selected_set.remove(b_idx)
                improved = True
                break
        
        if improved:
            continue
            
        # Phase 2: Try swaps
        # Try to swap a block with another that covers its exclusive parts
        # We do a few random attempts per pass
        for _ in range(5):
            if time.time() >= deadline:
                break
            if not selected:
                break
            i = rng.randint(0, len(selected) - 1)
            b_idx = selected[i]
            mask = block_masks[b_idx]
            
            exclusive_mask = 0
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if coverage[idx] == 1:
                    exclusive_mask |= lsb
                m ^= lsb
                
            if exclusive_mask == 0:
                continue
                
            lsb = exclusive_mask & -exclusive_mask
            ts_idx = lsb.bit_length() - 1
            candidates = t_subset_to_blocks[ts_idx]
            
            found_swap = False
            for j_idx in candidates:
                if j_idx in selected_set:
                    continue
                if (block_masks[j_idx] & exclusive_mask) == exclusive_mask:
                    # Swap
                    m = mask
                    while m:
                        lsb_i = m & -m
                        idx_i = lsb_i.bit_length() - 1
                        coverage[idx_i] -= 1
                        m ^= lsb_i
                        
                    mask_j = block_masks[j_idx]
                    m = mask_j
                    while m:
                        lsb_j = m & -m
                        idx_j = lsb_j.bit_length() - 1
                        coverage[idx_j] += 1
                        m ^= lsb_j
                        
                    selected[i] = j_idx
                    selected_set.remove(b_idx)
                    selected_set.add(j_idx)
                    improved = True
                    found_swap = True
                    break
                    
            if found_swap:
                break

    return selected


def construct(v, k, t):
    deadline = time.time() + 115
    rng = random.Random(42)
    
    block_masks, t_subset_to_blocks, t_subsets, blocks = _precompute(v, k, t)
    num_t = len(t_subsets)
    
    best_indices = None
    best_count = float('inf')
    
    while time.time() < deadline - 10:
        selected_indices = _greedy_cover_fast(v, k, t, block_masks, t_subset_to_blocks, num_t, rng, deadline)
        selected_indices = _local_search_fast(selected_indices, v, k, t, block_masks, t_subset_to_blocks, num_t, rng, deadline - 2)
        
        if len(selected_indices) < best_count:
            best_count = len(selected_indices)
            best_indices = selected_indices
            
    if best_indices is None:
        return []
        
    return [blocks[i] for i in best_indices]
# EVOLVE-BLOCK-END
