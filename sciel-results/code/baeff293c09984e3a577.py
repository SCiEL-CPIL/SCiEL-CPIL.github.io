"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np
from collections import defaultdict


def _precompute(v, t):
    """Precompute all t-subsets, their index mapping, and reverse mapping."""
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {c: i for i, c in enumerate(all_tsubs)}
    # Reverse: index -> tuple
    idx_to_tsub = all_tsubs
    return all_tsubs, tsub_to_idx, idx_to_tsub, n_tsubs


def _precompute_tsub_indices(v, t):
    """Precompute mapping from sorted tuple to index for all t-subsets."""
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {c: i for i, c in enumerate(all_tsubs)}
    return all_tsubs, tsub_to_idx, n_tsubs


def _fast_gains_for_point(uncovered, tsub_to_idx, block, x, t):
    """Compute gain of adding point x to block - optimized for small t."""
    # For t=4, k=7: each (t-1)=3 subset of block combined with x gives a 4-subset
    # We need to count how many of these are currently uncovered
    block_set = set(block)
    if x in block_set:
        return 0
    
    # Generate all (t-1)-subsets of block
    t_minus_1 = t - 1
    count = 0
    for s in itertools.combinations(block, t_minus_1):
        tsub = tuple(sorted(s + (x,)))
        idx = tsub_to_idx.get(tsub)
        if idx is not None and uncovered[idx]:
            count += 1
    return count


def _greedy_construct(v, k, t, rng, deadline, tsub_to_idx, n_tsubs, idx_to_tsub):
    """Greedy covering with multi-candidate selection for better exploration."""
    uncovered = np.ones(n_tsubs, dtype=bool)
    blocks = []
    
    while uncovered.any():
        if deadline and time.time() > deadline:
            break
        
        # Pick a random uncovered t-subset to seed the block
        uncovered_indices = np.where(uncovered)[0]
        seed_idx = int(rng.choice(uncovered_indices))
        seed_tsub = idx_to_tsub[seed_idx]
        block = list(seed_tsub)
        
        while len(block) < k:
            if deadline and time.time() > deadline:
                break
            
            block_set = set(block)
            
            # Compute gains for all candidate points
            candidates = []
            for x in range(v):
                if x in block_set:
                    continue
                gain = _fast_gains_for_point(uncovered, tsub_to_idx, block, x, t)
                candidates.append((gain, x))
            
            if not candidates:
                break
            
            # Sort by gain descending
            candidates.sort(key=lambda c: c[0], reverse=True)
            best_gain = candidates[0][0]
            
            # Multi-candidate: pick from top candidates with weighted probability
            # This helps escape local optima
            top_n = min(3, len(candidates))
            top_candidates = candidates[:top_n]
            
            # Weight by gain (with minimum weight 1 to allow exploration)
            weights = [max(1, c[0]) for c in top_candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen_point = top_candidates[0][1]
            for i, (g, x) in enumerate(top_candidates):
                cum += weights[i]
                if r < cum:
                    chosen_point = x
                    break
            
            block.append(chosen_point)
        
        if len(block) == k:
            block.sort()
            block_tuple = tuple(block)
            blocks.append(block_tuple)
            # Update uncovered
            for c in itertools.combinations(block, t):
                idx = tsub_to_idx[c]
                uncovered[idx] = False
    
    return [list(b) for b in blocks]


def _local_search(blocks, v, k, t, tsub_to_idx, n_tsubs, deadline):
    """Try to remove redundant blocks using coverage matrix for efficiency."""
    if deadline and time.time() > deadline:
        return blocks
    
    blocks = [list(b) for b in blocks]
    n = len(blocks)
    
    if n < 2:
        return blocks
    
    # Precompute coverage vectors for each block
    coverage = []
    for block in blocks:
        cov = np.zeros(n_tsubs, dtype=bool)
        for c in itertools.combinations(block, t):
            idx = tsub_to_idx[c]
            cov[idx] = True
        coverage.append(cov)
    
    # Iteratively remove redundant blocks
    active = list(range(n))
    removed_count = 0
    changed = True
    
    while changed and time.time() < deadline:
        changed = False
        new_active = []
        
        for i in active:
            if time.time() > deadline:
                break
            
            # Check if block i is redundant: does the union of all other active blocks cover everything?
            # Equivalent: is there any t-subset covered ONLY by block i?
            # i.e., is coverage[i] a subset of (OR of all other coverage[j] for j != i in active)?
            
            # Compute union of all other active blocks' coverage
            other_union = np.zeros(n_tsubs, dtype=bool)
            for j in active:
                if j != i:
                    other_union |= coverage[j]
            
            # Block i is redundant if all t-subsets it covers are also covered by others
            if not np.any(coverage[i] & ~other_union):
                removed_count += 1
                changed = True
            else:
                new_active.append(i)
        
        active = new_active
    
    if removed_count > 0:
        blocks = [blocks[i] for i in active]
    
    return blocks


def _precompute(v, t):
    """Precompute all t-subsets, their index mapping, and reverse mapping."""
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {c: i for i, c in enumerate(all_tsubs)}
    idx_to_tsub = all_tsubs
    return all_tsubs, tsub_to_idx, idx_to_tsub, n_tsubs


def construct(v, k, t):
    """Construct a covering design C(v,k,t) using greedy with restarts and local search."""
    start_time = time.time()
    deadline = start_time + 115.0  # 115s deadline, 35s safety margin
    
    # Precompute
    all_tsubs, tsub_to_idx, idx_to_tsub, n_tsubs = _precompute(v, t)
    
    best_blocks = None
    best_len = float('inf')
    
    # First deterministic greedy with fixed seed
    baseline = _greedy_construct(v, k, t, random.Random(42), deadline, tsub_to_idx, n_tsubs, idx_to_tsub)
    if baseline:
        best_blocks = baseline
        best_len = len(baseline)
    
    if time.time() > deadline:
        return best_blocks if best_blocks else []
    
    # Random restarts with varied strategies
    restart_count = 0
    while time.time() < deadline - 25:  # Keep 25s for local search
        restart_count += 1
        rng = random.Random(restart_count * 12345 + 7)
        blocks = _greedy_construct(v, k, t, rng, deadline, tsub_to_idx, n_tsubs, idx_to_tsub)
        
        if blocks and len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
    
    # Local search on the best solution found
    if best_blocks and time.time() < deadline - 3:
        best_blocks = _local_search(best_blocks, v, k, t, tsub_to_idx, n_tsubs, deadline)
    
    return best_blocks if best_blocks else []
# EVOLVE-BLOCK-END
