"""RPBC: Randomized Parallel Batch Construction with vectorized gain matrices."""
import time
import random
import itertools

import numpy as np

def construct(v, k, t):
    deadline = time.time() + 105.0
    rng = random.Random(42)
    
    # Precompute all k-subsets as masks and their covered t-subsets
    all_k_subsets = []
    masks_by_k = {}
    for comb in itertools.combinations(range(v), k):
        mask = 0
        for p in comb:
            mask |= (1 << p)
        all_k_subsets.append(mask)
        masks_by_k[mask] = comb
    
    n_k = len(all_k_subsets)
    
    # Enumerate all t-subsets
    all_t_subsets = []
    for sub in itertools.combinations(range(v), t):
        sm = 0
        for p in sub:
            sm |= (1 << p)
        all_t_subsets.append(sm)
    n_t = len(all_t_subsets)
    
    # Map t-subset mask to index
    idx_of_t = {m: i for i, m in enumerate(all_t_subsets)}
    
    # Build coverage matrix: cov_matrix[k_idx, t_idx] = 1 if k-subset covers t-subset
    cov_matrix = np.zeros((n_k, n_t), dtype=np.int8)
    for k_idx, mask in enumerate(all_k_subsets):
        comb = masks_by_k[mask]
        for sub in itertools.combinations(comb, t):
            sm = 0
            for p in sub:
                sm |= (1 << p)
            cov_matrix[k_idx, idx_of_t[sm]] = 1
    
    # Initial state
    cover_count = np.zeros(n_t, dtype=np.int32)
    blocks = []
    blocks_set = set()
    used_k_indices = set()
    
    def deadline_hit():
        return time.time() > deadline
    
    def add_block(k_idx):
        mask = all_k_subsets[k_idx]
        blocks.append(mask)
        blocks_set.add(mask)
        used_k_indices.add(k_idx)
        cover_count += cov_matrix[k_idx]
        return cover_count
    
    # RPBC: Randomized Parallel Batch Construction
    batch_size = 15
    
    while not deadline_hit():
        # Identify uncovered t-subsets
        uncovered_mask = (cover_count == 0).astype(np.int8)
        if uncovered_mask.sum() == 0:
            break
        
        # Compute gain for each candidate k-subset: number of newly covered t-subsets
        # gain[k_idx] = sum over t_idx where cov_matrix[k_idx, t_idx] == 1 and cover_count[t_idx] == 0
        gains = (cov_matrix * uncovered_mask[None, :]).sum(axis=1)
        
        # Exclude already used blocks
        for idx in used_k_indices:
            gains[idx] = -1
        
        # If no positive gain, break (should not happen if valid covering exists)
        if gains.max() <= 0:
            break
        
        # Select top candidates with randomization for diversity
        # Find indices with positive gain
        positive_gains = np.where(gains > 0)[0]
        if len(positive_gains) == 0:
            break
        
        # Get top candidates by gain, with some randomization
        # Sort by gain descending, then take a random subset from the top
        sorted_idx = positive_gains[np.argsort(gains[positive_gains])[::-1]]
        
        # Take top batch_size * 2 candidates, then randomly select batch_size from them
        top_candidates = sorted_idx[:min(len(sorted_idx), batch_size * 2)]
        
        # Randomly select from top candidates
        if len(top_candidates) > batch_size:
            selected = rng.sample(list(top_candidates), batch_size)
        else:
            selected = list(top_candidates)
        
        # Add selected blocks in parallel (all at once)
        for k_idx in selected:
            if k_idx not in used_k_indices:
                add_block(k_idx)
        
        # Check if all covered
        if (cover_count == 0).sum() == 0:
            break
    
    # Repair: ensure all t-subsets are covered
    uncovered_idx = np.where(cover_count == 0)[0]
    for i in uncovered_idx:
        if deadline_hit():
            # Even if we run out of time, we must cover
            pass
        sm = all_t_subsets[i]
        # Find any k-subset that covers this t-subset
        # Prefer unused ones
        covering_k_indices = np.where(cov_matrix[:, i] == 1)[0]
        
        # Filter out used ones
        available = [idx for idx in covering_k_indices if idx not in used_k_indices]
        
        if available:
            # Pick randomly from available
            k_idx = rng.choice(available)
        else:
            # All covering k-subsets are used; pick one randomly (will be duplicate but we check)
            k_idx = rng.choice(list(covering_k_indices))
        
        mask = all_k_subsets[k_idx]
        if mask not in blocks_set:
            add_block(k_idx)
        elif k_idx not in used_k_indices:
            # This shouldn't happen if blocks_set is consistent, but just in case
            used_k_indices.add(k_idx)
    
    # Convert masks back to sorted lists of points
    result = []
    for mask in blocks:
        pts = [p for p in range(v) if (mask >> p) & 1]
        pts.sort()
        result.append(pts)
    return result
