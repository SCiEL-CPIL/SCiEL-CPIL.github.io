"""Covering design C(24,7,3) - improved greedy with local search."""
import itertools
import random
import time
import numpy as np
from collections import defaultdict


def _greedy_cover(v, k, t, rng, deadline):
    """Fast greedy covering using numpy for speed."""
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    uncovered = np.ones(len(all_tsubs), dtype=np.int32)
    blocks = []
    
    while uncovered.any():
        if time.time() > deadline:
            return None
        
        # Pick a random uncovered t-subset
        uncov_indices = np.where(uncovered > 0)[0]
        start_idx = uncov_indices[rng.randrange(len(uncov_indices))]
        start_tsub = all_tsubs[start_idx]
        block = list(start_tsub)
        block_set = set(block)
        
        while len(block) < k:
            # Find point that covers most uncovered t-subsets
            best_gain = -1
            best_point = -1
            
            # Precompute combinations of block of size t-1
            block_combos = list(itertools.combinations(block, t - 1))
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in block_combos:
                    ts = tuple(sorted(rest + (x,)))
                    idx = tsub_to_idx.get(ts)
                    if idx is not None and uncovered[idx] > 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1:
                candidates = [x for x in range(v) if x not in block_set]
                best_point = candidates[rng.randrange(len(candidates))]
            
            block.append(best_point)
            block_set.add(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            uncovered[idx] = 0
    
    return blocks


def _remove_redundant_blocks(blocks, v, t):
    """Remove blocks that don't cover any unique t-subset."""
    if not blocks:
        return blocks
    
    cover_count = defaultdict(int)
    block_tsubs = []
    
    for block in blocks:
        tsubs = list(itertools.combinations(block, t))
        block_tsubs.append(tsubs)
        for ts in tsubs:
            cover_count[ts] += 1
    
    kept_blocks = []
    for i, block in enumerate(blocks):
        all_redundant = True
        for ts in block_tsubs[i]:
            if cover_count[ts] == 1:
                all_redundant = False
                break
        if not all_redundant:
            kept_blocks.append(block)
    
    if len(kept_blocks) < len(blocks):
        cover_count2 = defaultdict(int)
        for block in kept_blocks:
            for ts in itertools.combinations(block, t):
                cover_count2[ts] += 1
        
        final_blocks = []
        for i, block in enumerate(kept_blocks):
            all_redundant = True
            for ts in itertools.combinations(block, t):
                if cover_count2[ts] == 1:
                    all_redundant = False
                    break
            if not all_redundant:
                final_blocks.append(block)
        
        return final_blocks
    
    return blocks


def _local_search(blocks, v, k, t, rng, deadline):
    """Local search to try to reduce block count."""
    if not blocks:
        return blocks
    
    current_blocks = [list(b) for b in blocks]
    
    # Compute coverage counts
    cover_count = defaultdict(int)
    for block in current_blocks:
        for ts in itertools.combinations(block, t):
            cover_count[ts] += 1
    
    # Try to remove redundant blocks
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for i in range(len(current_blocks) - 1, -1, -1):
            is_redundant = True
            for ts in itertools.combinations(current_blocks[i], t):
                if cover_count.get(ts, 0) <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                for ts in itertools.combinations(current_blocks[i], t):
                    cover_count[ts] -= 1
                current_blocks.pop(i)
                changed = True
                break
    
    return current_blocks


def _verify_covering(blocks, v, t):
    """Verify every t-subset is covered."""
    covered = set()
    for block in blocks:
        covered.update(itertools.combinations(block, t))
    all_tsubs = set(itertools.combinations(range(v), t))
    return all_tsubs.issubset(covered)


def construct(v, k, t):
    deadline = time.time() + 130  # Safety margin before 150s limit
    rng = random.Random(12345)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple randomized restarts
    max_restarts = 30
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        blocks = _greedy_cover(v, k, t, rng, deadline)
        if blocks is None:
            continue
        
        blocks = _remove_redundant_blocks(blocks, v, t)
        blocks = _local_search(blocks, v, k, t, rng, deadline)
        
        if _verify_covering(blocks, v, t):
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
        
        if best_count <= 76:
            break
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        best_blocks = blocks
    
    return best_blocks
