"""Combined covering design program: incremental greedy + efficient local search."""
import itertools
import time
import random
import numpy as np


def _precompute(v, t):
    """Precompute all t-subsets and their index mapping."""
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {c: i for i, c in enumerate(all_tsubs)}
    return all_tsubs, tsub_to_idx, n_tsubs


def _greedy_construct(v, k, t, rng, deadline, tsub_to_idx, n_tsubs, idx_to_tsub):
    """Greedy covering: build blocks one point at a time maximizing gain."""
    uncovered = np.ones(n_tsubs, dtype=bool)
    blocks = []
    
    while uncovered.any():
        if time.time() > deadline:
            break
        
        # Pick a random uncovered t-subset to seed the block
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
        seed_idx = int(rng.choice(uncovered_indices))
        seed_tsub = idx_to_tsub[seed_idx]
        block = list(seed_tsub)
        
        while len(block) < k:
            if time.time() > deadline:
                break
            
            block_set = set(block)
            # Precompute all (t-1)-subsets of current block
            t_minus_1_subs = list(itertools.combinations(block, t - 1))
            
            best_gain = -1
            best_point = None
            
            for x in range(v):
                if x in block_set:
                    continue
                # Gain = number of (t-1)-subsets S of block where S∪{x} is uncovered
                gain = 0
                for s in t_minus_1_subs:
                    tsub = tuple(sorted(s + (x,)))
                    idx = tsub_to_idx.get(tsub)
                    if idx is not None and uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None:
                for x in range(v):
                    if x not in block_set:
                        best_point = x
                        break
            if best_point is not None:
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
    """Remove redundant blocks using coverage counts, then try pair swaps."""
    if deadline and time.time() > deadline:
        return blocks
    
    blocks = [list(b) for b in blocks]
    n = len(blocks)
    if n == 0:
        return blocks
    
    # Compute t-subset indices for each block
    block_tsubs = []
    for b in blocks:
        tsubs = []
        for c in itertools.combinations(b, t):
            idx = tsub_to_idx[c]
            tsubs.append(idx)
        block_tsubs.append(tsubs)
    
    # Coverage count: how many blocks cover each t-subset
    cover_count = np.zeros(n_tsubs, dtype=np.int32)
    for tsubs in block_tsubs:
        for idx in tsubs:
            cover_count[idx] += 1
    
    # Remove redundant blocks iteratively
    kept = [True] * n
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
        for i in range(n):
            if not kept[i]:
                continue
            # Check if block i is redundant (all its t-subsets covered by others)
            redundant = True
            for idx in block_tsubs[i]:
                if cover_count[idx] == 1:
                    redundant = False
                    break
            if redundant:
                kept[i] = False
                for idx in block_tsubs[i]:
                    cover_count[idx] -= 1
                changed = True
    
    reduced = [blocks[i] for i in range(n) if kept[i]]
    reduced_tsubs = [block_tsubs[i] for i in range(n) if kept[i]]
    
    # Try replacing pairs of blocks with a single block
    n_reduced = len(reduced)
    if n_reduced > 2 and time.time() < deadline - 10:
        rng = random.Random(42)
        # Recompute cover_count for reduced set
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for tsubs in reduced_tsubs:
            for idx in tsubs:
                cover_count[idx] += 1
        
        # Try random pair swaps
        for trial in range(200):
            if time.time() > deadline:
                break
            if n_reduced < 2:
                break
            i, j = rng.sample(range(n_reduced), 2)
            # Get all t-subsets covered by blocks i and j
            tsubs_ij = set(reduced_tsubs[i] + reduced_tsubs[j])
            # Check if removing both would leave any t-subset uncovered
            can_remove = True
            for idx in tsubs_ij:
                if cover_count[idx] <= 2:  # Only covered by i and/or j
                    # Check if covered by any other block
                    other_count = cover_count[idx] - (1 if idx in reduced_tsubs[i] else 0) - (1 if idx in reduced_tsubs[j] else 0)
                    if other_count <= 0:
                        can_remove = False
                        break
            if can_remove:
                # Remove both blocks
                # Find a replacement block that covers all tsubs_ij
                # This is expensive, so skip for now
                pass
    
    return reduced


def construct(v, k, t):
    """Construct a covering design C(v,k,t) using greedy with restarts and local search."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Precompute
    all_tsubs, tsub_to_idx, n_tsubs = _precompute(v, t)
    idx_to_tsub = all_tsubs
    
    best_blocks = None
    best_len = float('inf')
    
    # First deterministic greedy with fixed seed
    baseline = _greedy_construct(v, k, t, random.Random(42), deadline, tsub_to_idx, n_tsubs, idx_to_tsub)
    if baseline:
        best_blocks = baseline
        best_len = len(baseline)
    
    if time.time() > deadline:
        return best_blocks if best_blocks else []
    
    # Random restarts with varying seeds
    restart_count = 0
    while time.time() < deadline - 20:  # Keep 20s for local search
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
