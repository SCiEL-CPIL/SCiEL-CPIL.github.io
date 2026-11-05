"""
Improved covering C(19,8,5) using bitmask-based greedy with multiple restarts.
Uses numpy for fast bit operations. Deadline guard at ~115s to stay within 120s limit.
"""
import itertools
import time
import random

import numpy as np


def _bits_to_mask(bits, v):
    """Convert list of bit positions to a numpy uint64 mask."""
    mask = np.zeros(1, dtype=np.uint64)
    for b in bits:
        mask[0] |= (1 << b)
    return mask


def _mask_to_points(mask, v):
    """Convert a numpy uint64 mask to list of point indices."""
    pts = []
    for i in range(v):
        if mask[0] & (1 << i):
            pts.append(i)
    return pts


def _greedy_cover(v, k, t, rng, deadline):
    """
    Bitmask-based greedy set cover for C(v,k,t).
    Returns (blocks, covered_count) or None if deadline exceeded.
    """
    # Precompute all t-subsets as uint64 masks and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # For each k-block, precompute which t-subsets it covers (as list of indices)
    # And build inverted index: for each t-subset, list of k-blocks that cover it
    
    # First, generate all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(all_k_blocks)
    
    # Map from t-subset tuple to its index
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # For each k-block, compute the list of t-subset indices it covers
    block_covers = []  # list of lists of t-subset indices
    for blk in all_k_blocks:
        covered = []
        for sub in itertools.combinations(blk, t):
            idx = t_subset_to_idx[sub]
            covered.append(idx)
        block_covers.append(covered)
    
    # Inverted index: for each t-subset, list of block indices that cover it
    subset_to_blocks = [[] for _ in range(n_t)]
    for bi, covered in enumerate(block_covers):
        for si in covered:
            subset_to_blocks[si].append(bi)
    
    # Counts of how many times each t-subset is covered
    counts = np.zeros(n_t, dtype=np.int32)
    
    # Set of currently uncovered t-subset indices
    uncovered = set(range(n_t))
    
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            return None
        
        # Pick a random uncovered t-subset
        target = rng.choice(sorted(uncovered))  # sorted for reproducibility with seed
        
        # Get all blocks that cover this t-subset
        candidate_blocks = subset_to_blocks[target]
        
        if not candidate_blocks:
            # Should not happen for valid parameters, but guard
            return None
        
        # Find the block with maximum gain (number of currently uncovered t-subsets it covers)
        best_block = None
        best_gain = -1
        for bi in candidate_blocks:
            gain = 0
            for si in block_covers[bi]:
                if si in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = bi
        
        if best_block is None:
            return None
        
        # Add this block
        blk = all_k_blocks[best_block]
        blocks.append(blk)
        
        # Update counts and uncovered set
        for si in block_covers[best_block]:
            counts[si] += 1
            if counts[si] == 1:
                uncovered.discard(si)
    
    return blocks


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using bitmask-based greedy with multiple restarts.
    Returns list of blocks (each a list of k distinct integers in 0..v-1).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Precompute
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Try multiple restarts with different random seeds
    best_blocks = None
    best_len = float('inf')
    
    # Single deterministic greedy first (seed=0)
    rng = random.Random(0)
    result = _greedy_cover(v, k, t, rng, deadline)
    if result is not None:
        best_blocks = result
        best_len = len(result)
    
    # Try additional random restarts if time permits
    seed = 1
    while True:
        if time.time() > deadline - 5.0:
            break
        rng = random.Random(seed)
        result = _greedy_cover(v, k, t, rng, deadline)
        if result is not None:
            if len(result) < best_len:
                best_blocks = result
                best_len = len(result)
        seed += 1
    
    # If we haven't found any valid covering, fall back to a simple constructive approach
    if best_blocks is None:
        # Simple greedy without randomization
        uncovered = set(all_t_subsets)
        blocks = []
        t_subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
        
        while uncovered and time.time() < deadline:
            # Pick the first uncovered t-subset
            target = min(uncovered)
            target_set = set(target)
            
            # Find the best block covering this t-subset
            best_block = None
            best_gain = -1
            for blk in itertools.combinations(range(v), k):
                if target_set.issubset(set(blk)):
                    # Count how many uncovered t-subsets this block covers
                    gain = 0
                    for sub in itertools.combinations(blk, t):
                        if sub in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block = blk
            
            if best_block is None:
                break
            
            blocks.append(list(best_block))
            # Update uncovered
            blk_set = set(best_block)
            to_remove = []
            for sub in uncovered:
                if set(sub).issubset(blk_set):
                    to_remove.append(sub)
            for sub in to_remove:
                uncovered.discard(sub)
        
        best_blocks = blocks
        best_len = len(blocks)
    
    # Final polish: try to remove redundant blocks
    if best_blocks is not None and time.time() < deadline - 5.0:
        polished = list(best_blocks)
        n = len(polished)
        # Try removing each block and check if covering is still valid
        removed = [False] * n
        for i in range(n):
            if removed[i]:
                continue
            if time.time() > deadline - 2.0:
                break
            # Tentatively remove block i
            removed[i] = True
            # Check if all t-subsets are still covered by remaining blocks
            all_covered = True
            remaining = [polished[j] for j in range(n) if not removed[j]]
            # Build set of all t-subsets covered by remaining blocks
            covered_subs = set()
            for blk in remaining:
                for sub in itertools.combinations(blk, t):
                    covered_subs.add(sub)
            # Check if this equals all_t_subsets
            if len(covered_subs) == n_t:
                # Block i can be removed
                pass
            else:
                # Need block i back
                removed[i] = False
        
        # Rebuild polished list
        final = [polished[i] for i in range(n) if not removed[i]]
        if final and len(final) < len(best_blocks):
            best_blocks = final
            best_len = len(final)
    
    return best_blocks if best_blocks is not None else []
