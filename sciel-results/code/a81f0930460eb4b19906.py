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


def _vectorized_gain(uncovered, tsub_to_idx, block, x, t):
    """Compute gain of adding point x to block using vectorized operations."""
    # Get all (t-1)-subsets of current block
    t_minus_1_subs = list(itertools.combinations(block, t - 1))
    if not t_minus_1_subs:
        return 0
    
    # For each (t-1)-subset, form the t-subset with x and check if uncovered
    gains = []
    for s in t_minus_1_subs:
        tsub = tuple(sorted(s + (x,)))
        idx = tsub_to_idx.get(tsub)
        if idx is not None and uncovered[idx]:
            gains.append(1)
    return sum(gains)


def _greedy_construct(v, k, t, rng, deadline, tsub_to_idx, n_tsubs, idx_to_tsub):
    """Greedy covering with fast gain computation."""
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
            
            best_gain = -1
            best_point = None
            second_best_gain = -1
            second_best_point = None
            
            for x in range(v):
                if x in block_set:
                    continue
                # Gain = number of (t-1)-subsets S of block where S∪{x} is uncovered
                gain = _vectorized_gain(uncovered, tsub_to_idx, block, x, t)
                if gain > best_gain:
                    second_best_gain = best_gain
                    second_best_point = best_point
                    best_gain = gain
                    best_point = x
                elif gain > second_best_gain:
                    second_best_gain = gain
                    second_best_point = x
            
            if best_point is None:
                # Fallback: add any unused point
                for x in range(v):
                    if x not in block_set:
                        best_point = x
                        break
            if best_point is not None:
                # With some probability, pick the second-best point to add diversity
                if (second_best_point is not None and 
                    best_gain == second_best_gain and 
                    rng.random() < 0.3):
                    block.append(second_best_point)
                else:
                    block.append(best_point)
            else:
                break
        
        if len(block) == k:
            block.sort()
            blocks.append(tuple(block))
            # Update uncovered
            for c in itertools.combinations(block, t):
                idx = tsub_to_idx[c]
                uncovered[idx] = False
    
    return [list(b) for b in blocks]


def _local_search(blocks, v, k, t, tsub_to_idx, n_tsubs, deadline):
    """Try to remove redundant blocks."""
    if deadline and time.time() > deadline:
        return blocks
    
    blocks = [list(b) for b in blocks]
    n = len(blocks)
    
    # Precompute coverage mask for each block
    def block_coverage(block):
        cov = np.zeros(n_tsubs, dtype=bool)
        for c in itertools.combinations(block, t):
            idx = tsub_to_idx[c]
            cov[idx] = True
        return cov
    
    # Try removing each block
    removed = set()
    for i in range(n):
        if deadline and time.time() > deadline:
            break
        if i in removed:
            continue
        # Check if block i is redundant given all non-removed blocks
        remaining = [j for j in range(n) if j != i and j not in removed]
        if not remaining:
            # Only one block left, can't remove
            continue
        # Compute union of coverage of remaining blocks
        union_cov = np.zeros(n_tsubs, dtype=bool)
        for j in remaining:
            union_cov |= block_coverage(blocks[j])
        if union_cov.all():
            removed.add(i)
    
    if removed:
        blocks = [blocks[i] for i in range(n) if i not in removed]
    
    return blocks


def construct(v, k, t):
    """Construct a covering design C(v,k,t) using greedy with restarts and local search."""
    start_time = time.time()
    deadline = start_time + 110.0  # 110s deadline, 40s safety margin
    
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
    
    # Random restarts
    restart_count = 0
    while time.time() < deadline - 30:  # Keep 30s for local search
        restart_count += 1
        rng = random.Random(restart_count * 12345 + 7)
        blocks = _greedy_construct(v, k, t, rng, deadline, tsub_to_idx, n_tsubs, idx_to_tsub)
        
        if blocks and len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
    
    # Local search on the best solution found
    if best_blocks and time.time() < deadline - 5:
        best_blocks = _local_search(best_blocks, v, k, t, tsub_to_idx, n_tsubs, deadline)
    
    return best_blocks if best_blocks else []
# EVOLVE-BLOCK-END
