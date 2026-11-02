"""VND heuristic for covering design C(v,k,t).
Uses bitset operations for rapid feasibility checks and systematic
1-exchange to m-for-m block swaps (m=2,3) to escape local minima.
"""
import itertools
import random
import numpy as np
from collections import defaultdict


def construct(v, k, t):
    """Construct a covering design C(v,k,t) using VND heuristic.
    
    Args:
        v: number of points
        k: block size
        t: subset size to cover
    
    Returns:
        List of blocks (each a list of k distinct integers in 0..v-1)
    """
    # Precompute all t-subsets as bitmasks
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index
    tsub_to_idx = {tuple(cs): i for i, cs in enumerate(all_tsubs)}
    
    # For each point, precompute which t-subsets it's in
    # point_tsubs[p] = list of t-subset indices containing point p
    point_tsubs = [[] for _ in range(v)]
    for idx, cs in enumerate(all_tsubs):
        for p in cs:
            point_tsubs[p].append(idx)
    
    # For each pair of points, precompute which t-subsets contain both
    # This is expensive for large v, so we'll compute on the fly or use a different approach
    
    # Greedy initial construction
    uncovered_mask = np.ones(num_tsubs, dtype=np.uint8)
    blocks = []
    
    while uncovered_mask.any():
        # Find first uncovered t-subset
        idx = int(np.argmax(uncovered_mask))
        if uncovered_mask[idx] == 0:
            break
        target = all_tsubs[idx]
        
        # Build block greedily starting with target
        block = list(target)
        block_set = set(block)
        
        # Track which t-subsets are covered by current block
        # A t-subset is covered if all its elements are in the block
        while len(block) < k:
            best_point = -1
            best_gain = -1
            for p in range(v):
                if p in block_set:
                    continue
                # Calculate gain: how many uncovered t-subsets would be covered
                # if we add p to the block
                # A t-subset is covered by block+p if it's a subset of block+p
                # and was previously uncovered
                gain = 0
                # For each t-subset containing p, check if the rest is in block
                for ts_idx in point_tsubs[p]:
                    if uncovered_mask[ts_idx] == 0:
                        continue
                    ts = all_tsubs[ts_idx]
                    # Check if all other elements of ts are in block
                    if all(x in block_set for x in ts if x != p):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
            
            if best_point == -1 or best_gain == 0:
                # No point adds coverage, just pick any
                for p in range(v):
                    if p not in block_set:
                        best_point = p
                        break
                if best_point == -1:
                    break
            
            block.append(best_point)
            block_set.add(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered mask
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[tuple(ts)]
            uncovered_mask[idx] = 0
    
    # VND phase: try to improve by replacing blocks
    # For each block, try to find a better replacement
    
    def block_coverage(block):
        """Return set of t-subset indices covered by this block."""
        cov = set()
        for ts in itertools.combinations(block, t):
            cov.add(tsub_to_idx[tuple(ts)])
        return cov
    
    # Compute coverage for all blocks
    block_covs = [block_coverage(b) for b in blocks]
    
    # Current uncovered
    current_uncovered = set(range(num_tsubs))
    for cov in block_covs:
        current_uncovered -= cov
    
    # VND: try 1-exchange, then 2-for-2, then 3-for-3
    for m in [1, 2, 3]:
        improved = True
        while improved:
            improved = False
            # Try to replace m consecutive blocks with m new blocks that cover more
            for start in range(len(blocks) - m + 1):
                if m == 1:
                    # Try replacing block[start] with a better one
                    old_block = blocks[start]
                    old_cov = block_covs[start]
                    
                    # Find t-subsets only covered by this block
                    only_covered = set()
                    for idx in old_cov:
                        if idx not in current_uncovered:
                            # Check if any other block covers it
                            covered_by_other = False
                            for j in range(len(blocks)):
                                if j == start:
                                    continue
                                if idx in block_covs[j]:
                                    covered_by_other = True
                                    break
                            if not covered_by_other:
                                only_covered.add(idx)
                    
                    # Build new block to cover only_covered plus maximize other coverage
                    # Start with a point from each uncovered t-subset in only_covered
                    new_block = []
                    new_block_set = set()
                    
                    # Greedy: pick points that cover most of only_covered
                    remaining = set(only_covered)
                    while len(new_block) < k and remaining:
                        best_p = -1
                        best_gain = -1
                        for p in range(v):
                            if p in new_block_set:
                                continue
                            gain = 0
                            for ts_idx in point_tsubs[p]:
                                if ts_idx in remaining:
                                    ts = all_tsubs[ts_idx]
                                    if all(x in new_block_set for x in ts if x != p):
                                        gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_p = p
                        if best_p == -1 or best_gain == 0:
                            break
                        new_block.append(best_p)
                        new_block_set.add(best_p)
                        # Update remaining
                        new_remaining = set()
                        for ts_idx in remaining:
                            ts = all_tsubs[ts_idx]
                            if all(x in new_block_set for x in ts):
                                pass  # covered
                            else:
                                new_remaining.add(ts_idx)
                        remaining = new_remaining
                    
                    # Fill rest with points that maximize additional coverage
                    while len(new_block) < k:
                        best_p = -1
                        best_gain = -1
                        for p in range(v):
                            if p in new_block_set:
                                continue
                            gain = 0
                            for ts_idx in point_tsubs[p]:
                                if ts_idx not in current_uncovered and ts_idx not in old_cov:
                                    # This would be new coverage
                                    ts = all_tsubs[ts_idx]
                                    if all(x in new_block_set for x in ts if x != p):
                                        gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_p = p
                        if best_p == -1:
                            break
                        new_block.append(best_p)
                        new_block_set.add(best_p)
                    
                    if len(new_block) < k:
                        continue
                    
                    new_block.sort()
                    new_cov = block_coverage(new_block)
                    
                    # Check if this is an improvement: new_cov should cover all of old_cov
                    # that aren't covered by other blocks, and ideally cover more
                    if old_cov.issubset(new_cov) or (new_cov - old_cov) and not (old_cov - new_cov):
                        # Accept if it covers at least as much
                        if len(new_cov) >= len(old_cov):
                            blocks[start] = new_block
                            block_covs[start] = new_cov
                            # Update current_uncovered
                            current_uncovered = set(range(num_tsubs))
                            for cov in block_covs:
                                current_uncovered -= cov
                            improved = True
                            break
                else:
                    # m-for-m exchange: try to find m new blocks that cover
                    # the union of t-subsets currently covered by blocks[start:start+m]
                    # and possibly more
                    
                    old_blocks = blocks[start:start+m]
                    old_covs = block_covs[start:start+m]
                    union_old = set()
                    for cov in old_covs:
                        union_old |= cov
                    
                    # Find t-subsets that are only covered by these m blocks
                    only_covered = set()
                    for idx in union_old:
                        if idx in current_uncovered:
                            continue
                        # Check if covered by any block outside this range
                        covered_by_other = False
                        for j in range(len(blocks)):
                            if start <= j < start + m:
                                continue
                            if idx in block_covs[j]:
                                covered_by_other = True
                                break
                        if not covered_by_other:
                            only_covered.add(idx)
                    
                    # Build m new blocks to cover only_covered
                    new_blocks = []
                    remaining = set(only_covered)
                    
                    for _ in range(m):
                        nb = []
                        nb_set = set()
                        while len(nb) < k and remaining:
                            best_p = -1
                            best_gain = -1
                            for p in range(v):
                                if p in nb_set:
                                    continue
                                gain = 0
                                for ts_idx in point_tsubs[p]:
                                    if ts_idx in remaining:
                                        ts = all_tsubs[ts_idx]
                                        if all(x in nb_set for x in ts if x != p):
                                            gain += 1
                                if gain > best_gain:
                                    best_gain = gain
                                    best_p = p
                            if best_p == -1 or best_gain == 0:
                                break
                            nb.append(best_p)
                            nb_set.add(best_p)
                            new_remaining = set()
                            for ts_idx in remaining:
                                ts = all_tsubs[ts_idx]
                                if all(x in nb_set for x in ts):
                                    pass
                                else:
                                    new_remaining.add(ts_idx)
                            remaining = new_remaining
                        
                        # Fill rest
                        while len(nb) < k:
                            best_p = -1
                            best_gain = -1
                            for p in range(v):
                                if p in nb_set:
                                    continue
                                gain = 0
                                for ts_idx in point_tsubs[p]:
                                    if ts_idx not in current_uncovered:
                                        ts = all_tsubs[ts_idx]
                                        if all(x in nb_set for x in ts if x != p):
                                            gain += 1
                                if gain > best_gain:
                                    best_gain = gain
                                    best_p = p
                            if best_p == -1:
                                break
                            nb.append(best_p)
                            nb_set.add(best_p)
                        
                        if len(nb) < k:
                            break
                        nb.sort()
                        new_blocks.append(nb)
                    
                    if len(new_blocks) < m:
                        continue
                    
                    new_covs = [block_coverage(nb) for nb in new_blocks]
                    union_new = set()
                    for cov in new_covs:
                        union_new |= cov
                    
                    # Check if new blocks cover all of only_covered
                    if only_covered.issubset(union_new):
                        # Accept
                        for i in range(m):
                            blocks[start + i] = new_blocks[i]
                            block_covs[start + i] = new_covs[i]
                        current_uncovered = set(range(num_tsubs))
                        for cov in block_covs:
                            current_uncovered -= cov
                        improved = True
                        break
        
        if not improved:
            break
    
    return blocks
