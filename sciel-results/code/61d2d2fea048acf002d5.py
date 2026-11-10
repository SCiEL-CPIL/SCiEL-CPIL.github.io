import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    # Precompute all t-subsets and their indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute coverage bitset for each possible k-block
    # There are C(19,7) = 50388 possible blocks
    # Each covers C(7,4)=35 t-subsets
    all_kblocks = list(itertools.combinations(range(v), k))
    block_coverage = {}
    for block in all_kblocks:
        cov = 0
        for ts in itertools.combinations(block, t):
            idx = tsubset_to_idx[ts]
            cov |= (1 << idx)
        block_coverage[block] = cov
    
    # Greedy construction
    uncovered = (1 << n_tsubsets) - 1
    blocks = []
    used_blocks = set()
    
    while uncovered and (time.time() - start_time) < deadline - 10:
        best_block = None
        best_gain = -1
        for block in all_kblocks:
            if block in used_blocks:
                continue
            gain = bin(uncovered & block_coverage[block]).count('1')
            if gain > best_gain:
                best_gain = gain
                best_block = block
            if gain == 0:
                continue
        if best_block is None or best_gain == 0:
            break
        blocks.append(list(best_block))
        used_blocks.add(best_block)
        uncovered &= ~block_coverage[best_block]
    
    if not uncovered:
        return blocks
    
    # Local search: try to improve
    # Phase 1: Remove redundant blocks
    changed = True
    while changed and (time.time() - start_time) < deadline - 5:
        changed = False
        for i in range(len(blocks)):
            # Check if block i is redundant
            # Temporarily remove it and check if all t-subsets it covers are still covered
            cov_i = block_coverage[tuple(blocks[i])]
            # Check if cov_i is subset of union of other blocks
            others_cov = 0
            for j in range(len(blocks)):
                if j != i:
                    others_cov |= block_coverage[tuple(blocks[j])]
            if (cov_i & ~others_cov) == 0:
                blocks.pop(i)
                changed = True
                break
    
    # Phase 2: Try to add blocks to cover remaining uncovered
    while uncovered and (time.time() - start_time) < deadline - 5:
        best_block = None
        best_gain = -1
        for block in all_kblocks:
            if tuple(block) in [tuple(b) for b in blocks]:
                continue
            gain = bin(uncovered & block_coverage[block]).count('1')
            if gain > best_gain:
                best_gain = gain
                best_block = block
        if best_block is None or best_gain == 0:
            break
        blocks.append(list(best_block))
        uncovered &= ~block_coverage[tuple(best_block)]
    
    # Verify and return
    return blocks
