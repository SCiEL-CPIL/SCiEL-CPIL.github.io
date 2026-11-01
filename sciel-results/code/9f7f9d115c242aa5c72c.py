"""Improved covering design C(19,8,5) using efficient bitset-based greedy and local search."""
import itertools
import time
import random
import sys
import numpy as np

_global_deadline = None


def _precompute(v, k, t):
    """Precompute all t-subsets and build efficient lookup structures."""
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    idx_map = {ts: i for i, ts in enumerate(all_tsubsets)}
    return all_tsubsets, idx_map, n_tsubsets


def _block_covers(block, t):
    """Return list of t-subset indices covered by a block."""
    return [ts for ts in itertools.combinations(block, t)]


def _greedy_cover(v, k, t, rng, deadline, all_tsubsets, idx_map, n_tsubsets):
    """Efficient greedy set-cover using bitsets for coverage tracking."""
    uncovered = np.ones(n_tsubsets, dtype=bool)
    blocks = []
    
    while uncovered.any() and time.time() < deadline:
        # Pick a random uncovered t-subset as seed
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
        seed_idx = rng.choice(uncovered_indices)
        seed = list(all_tsubsets[seed_idx])
        block = list(seed)
        block_set = set(block)
        
        while len(block) < k:
            # Compute gain for each candidate point using bitset operations
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many uncovered t-subsets are completed by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in idx_map:
                        idx = idx_map[ts]
                        if uncovered[idx]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x < 0:
                break
            block.append(best_x)
            block_set.add(best_x)
        
        block.sort()
        blocks.append(block)
        # Remove all t-subsets covered by this block
        for ts in itertools.combinations(block, t):
            if ts in idx_map:
                idx = idx_map[ts]
                uncovered[idx] = False
    
    return blocks


def _local_search(blocks, v, k, t, deadline, all_tsubsets, idx_map, n_tsubsets):
    """Remove redundant blocks iteratively."""
    blocks = [list(b) for b in blocks]
    
    # Precompute coverage: for each t-subset, count how many blocks cover it
    coverage_count = np.zeros(n_tsubsets, dtype=np.int32)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            ts = tuple(sorted(ts))
            if ts in idx_map:
                coverage_count[idx_map[ts]] += 1

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                break
            # Check if removing block bi leaves any t-subset with count 1
            redundant = True
            for ts in itertools.combinations(blocks[bi], t):
                ts = tuple(sorted(ts))
                if ts in idx_map:
                    idx = idx_map[ts]
                    if coverage_count[idx] <= 1:
                        redundant = False
                        break
            if redundant:
                # Remove block bi
                removed_block = blocks.pop(bi)
                for ts in itertools.combinations(removed_block, t):
                    ts = tuple(sorted(ts))
                    if ts in idx_map:
                        idx = idx_map[ts]
                        coverage_count[idx] -= 1
                improved = True
                break
    return blocks


def _block_swap_search(blocks, v, k, t, deadline, rng, all_tsubsets, idx_map, n_tsubsets):
    """Try to replace a block with a different one that covers more uncovered t-subsets."""
    blocks = [list(b) for b in blocks]
    
    coverage_count = np.zeros(n_tsubsets, dtype=np.int32)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            ts = tuple(sorted(ts))
            if ts in idx_map:
                coverage_count[idx_map[ts]] += 1
    
    improved = True
    iterations = 0
    while improved and time.time() < deadline and iterations < 5000:
        improved = False
        iterations += 1
        
        # Try removing a random block and see if we can add a better one
        if len(blocks) < 2:
            break
        bi = rng.randrange(len(blocks))
        
        # Try to find a replacement block that covers at least as many t-subsets
        # Remove the block temporarily
        removed_block = blocks.pop(bi)
        for ts in itertools.combinations(removed_block, t):
            ts = tuple(sorted(ts))
            if ts in idx_map:
                coverage_count[idx_map[ts]] -= 1
        
        # Count uncovered t-subsets after removal
        uncovered_mask = coverage_count == 0
        n_uncovered = np.sum(uncovered_mask)
        
        # Try to find a block that covers many of these uncovered t-subsets
        best_block = None
        best_coverage = 0
        
        # Try a few random blocks
        for _ in range(5):
            candidate = sorted(rng.sample(range(v), k))
            cand_set = set(candidate)
            cand_coverage = 0
            for ts in itertools.combinations(candidate, t):
                ts = tuple(sorted(ts))
                if ts in idx_map:
                    idx = idx_map[ts]
                    if coverage_count[idx] == 0:
                        cand_coverage += 1
            if cand_coverage > best_coverage:
                best_coverage = cand_coverage
                best_block = candidate
        
        if best_block and best_coverage >= 0:
            # Add the new block
            blocks.append(best_block)
            for ts in itertools.combinations(best_block, t):
                ts = tuple(sorted(ts))
                if ts in idx_map:
                    coverage_count[idx_map[ts]] += 1
            improved = True
        else:
            # Revert
            blocks.insert(bi, removed_block)
            for ts in itertools.combinations(removed_block, t):
                ts = tuple(sorted(ts))
                if ts in idx_map:
                    coverage_count[idx_map[ts]] += 1
    
    return blocks


def _pairwise_merge_search(blocks, v, k, t, deadline, rng, all_tsubsets, idx_map, n_tsubsets):
    """Try to merge two blocks into one if possible."""
    blocks = [list(b) for b in blocks]
    
    coverage_count = np.zeros(n_tsubsets, dtype=np.int32)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            ts = tuple(sorted(ts))
            if ts in idx_map:
                coverage_count[idx_map[ts]] += 1
    
    improved = True
    iterations = 0
    while improved and time.time() < deadline and iterations < 3000:
        improved = False
        iterations += 1
        
        # First: try removing redundant blocks
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                return blocks
            redundant = True
            for ts in itertools.combinations(blocks[bi], t):
                ts = tuple(sorted(ts))
                if ts in idx_map:
                    idx = idx_map[ts]
                    if coverage_count[idx] <= 1:
                        redundant = False
                        break
            if redundant:
                removed_block = blocks.pop(bi)
                for ts in itertools.combinations(removed_block, t):
                    ts = tuple(sorted(ts))
                    if ts in idx_map:
                        coverage_count[idx_map[ts]] -= 1
                improved = True
                break
        
        if improved or time.time() >= deadline:
            continue
        
        # Try merging two blocks
        if len(blocks) >= 2 and rng.random() < 0.4:
            bi = rng.randrange(len(blocks))
            bj = rng.randrange(len(blocks))
            if bi == bj:
                continue
            
            b_i = set(blocks[bi])
            b_j = set(blocks[bj])
            union = b_i | b_j
            
            if len(union) <= k:
                candidate_block = sorted(union)
            else:
                # Greedily select k elements from union
                candidate_block = sorted(rng.sample(list(union), k))
            
            candidate_set = set(candidate_block)
            
            # Check if removing bi and bj and adding candidate is safe
            safe = True
            for idx in [bi, bj]:
                for ts in itertools.combinations(blocks[idx], t):
                    ts = tuple(sorted(ts))
                    if ts in idx_map:
                        tidx = idx_map[ts]
                        if coverage_count[tidx] <= 1 and ts not in itertools.combinations(candidate_block, t):
                            safe = False
                            break
                if not safe:
                    break
            
            if safe:
                # Perform merge
                for idx in [bi, bj]:
                    for ts in itertools.combinations(blocks[idx], t):
                        ts = tuple(sorted(ts))
                        if ts in idx_map:
                            coverage_count[idx_map[ts]] -= 1
                
                for idx in sorted([bi, bj], reverse=True):
                    blocks.pop(idx)
                
                blocks.append(candidate_block)
                for ts in itertools.combinations(candidate_block, t):
                    ts = tuple(sorted(ts))
                    if ts in idx_map:
                        coverage_count[idx_map[ts]] += 1
                
                improved = True
                break
    
    return blocks


def construct(v, k, t):
    global _global_deadline
    _global_deadline = time.time() + 110
    rng = random.Random(42)
    
    # Precompute
    all_tsubsets, idx_map, n_tsubsets = _precompute(v, k, t)
    
    # Phase 1: Greedy cover with multiple restarts
    best_blocks = None
    best_count = float('inf')
    
    blocks = _greedy_cover(v, k, t, rng, _global_deadline - 80, all_tsubsets, idx_map, n_tsubsets)
    best_blocks = blocks
    best_count = len(blocks)
    
    restarts = 0
    while time.time() < _global_deadline - 80 and restarts < 2:
        blocks = _greedy_cover(v, k, t, rng, _global_deadline - 80, all_tsubsets, idx_map, n_tsubsets)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        restarts += 1
    
    # Phase 2: Local search (redundancy removal)
    if time.time() < _global_deadline - 40:
        best_blocks = _local_search(best_blocks, v, k, t, _global_deadline - 40, all_tsubsets, idx_map, n_tsubsets)
    
    # Phase 3: Block swap search
    if time.time() < _global_deadline - 20:
        best_blocks = _block_swap_search(best_blocks, v, k, t, _global_deadline - 20, rng, all_tsubsets, idx_map, n_tsubsets)
    
    # Phase 4: Pairwise merge
    if time.time() < _global_deadline - 5:
        best_blocks = _pairwise_merge_search(best_blocks, v, k, t, _global_deadline - 5, rng, all_tsubsets, idx_map, n_tsubsets)
    
    # Final cleanup: remove any remaining redundant blocks
    if time.time() < _global_deadline - 1:
        best_blocks = _local_search(best_blocks, v, k, t, _global_deadline - 1, all_tsubsets, idx_map, n_tsubsets)
    
    return best_blocks
