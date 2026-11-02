"""VND heuristic for covering design C(v,k,t).
Uses bitset operations for rapid feasibility checks and systematic
1-exchange to m-for-m block swaps (m=2,3) to escape local minima.
"""
import itertools
import random
import time
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
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer for evaluation overhead
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index
    tsub_to_idx = {tuple(cs): i for i, cs in enumerate(all_tsubs)}
    
    # Precompute for each point, the list of t-subsets it appears in
    point_tsubs = [[] for _ in range(v)]
    for idx, cs in enumerate(all_tsubs):
        for p in cs:
            point_tsubs[p].append(idx)
    
    # Precompute bitmask for each t-subset (as a set of points)
    tsub_points = [set(cs) for cs in all_tsubs]
    
    def block_covers_ts(block_set, ts_idx):
        """Check if block_set covers t-subset ts_idx."""
        ts = tsub_points[ts_idx]
        return ts.issubset(block_set)
    
    def get_block_coverage(block):
        """Return frozenset of t-subset indices covered by this block."""
        block_set = set(block)
        cov = set()
        for ts_idx, ts in enumerate(tsub_points):
            if ts.issubset(block_set):
                cov.add(ts_idx)
        return frozenset(cov)
    
    def greedy_build_block(must_cover, avoid_set, k_limit):
        """Build a block that covers as many of must_cover as possible.
        
        Args:
            must_cover: set of t-subset indices to prioritize
            avoid_set: set of points to avoid (optional)
            k_limit: max block size
        
        Returns:
            list of k points, or None if failed
        """
        block = []
        block_set = set()
        
        # First, try to include points from must_cover t-subsets
        remaining = set(must_cover)
        
        while len(block) < k_limit and remaining:
            best_p = -1
            best_gain = -1
            for p in range(v):
                if p in block_set:
                    continue
                if avoid_set and p in avoid_set:
                    continue
                gain = 0
                for ts_idx in point_tsubs[p]:
                    if ts_idx in remaining:
                        ts = tsub_points[ts_idx]
                        if ts.difference({p}).issubset(block_set):
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
            if best_p == -1 or best_gain == 0:
                break
            block.append(best_p)
            block_set.add(best_p)
            # Update remaining
            new_remaining = set()
            for ts_idx in remaining:
                ts = tsub_points[ts_idx]
                if not ts.issubset(block_set):
                    new_remaining.add(ts_idx)
            remaining = new_remaining
        
        # Fill rest with points that maximize coverage of any uncovered t-subsets
        while len(block) < k_limit:
            best_p = -1
            best_gain = -1
            for p in range(v):
                if p in block_set:
                    continue
                if avoid_set and p in avoid_set:
                    continue
                gain = 0
                for ts_idx in point_tsubs[p]:
                    ts = tsub_points[ts_idx]
                    if ts.difference({p}).issubset(block_set) and ts_idx not in remaining:
                        # This would newly cover ts_idx
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
            if best_p == -1:
                break
            block.append(best_p)
            block_set.add(best_p)
        
        if len(block) < k_limit:
            return None
        return sorted(block)
    
    # Greedy initial construction
    uncovered_mask = np.ones(num_tsubs, dtype=np.uint8)
    blocks = []
    
    while uncovered_mask.any():
        if time.time() - start_time > time_limit:
            break
        # Find first uncovered t-subset
        idx = int(np.argmax(uncovered_mask))
        if uncovered_mask[idx] == 0:
            break
        target = all_tsubs[idx]
        
        # Build block greedily starting with target
        block = list(target)
        block_set = set(block)
        
        while len(block) < k:
            best_point = -1
            best_gain = -1
            for p in range(v):
                if p in block_set:
                    continue
                gain = 0
                for ts_idx in point_tsubs[p]:
                    if uncovered_mask[ts_idx] == 0:
                        continue
                    ts = tsub_points[ts_idx]
                    if ts.difference({p}).issubset(block_set):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
            
            if best_point == -1 or best_gain == 0:
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
    
    # Compute coverage for all blocks
    block_covs = [get_block_coverage(b) for b in blocks]
    
    # Compute coverage count for each t-subset
    cov_count = np.zeros(num_tsubs, dtype=np.int32)
    for cov in block_covs:
        for idx in cov:
            cov_count[idx] += 1
    
    def update_cov_count(old_cov, new_cov, sign):
        """Update cov_count when replacing old_cov with new_cov."""
        for idx in old_cov:
            cov_count[idx] -= sign
        for idx in new_cov:
            cov_count[idx] += sign
    
    # VND phase
    def try_1_exchange(start):
        """Try to replace block[start] with a better one. Returns True if improved."""
        old_block = blocks[start]
        old_cov = block_covs[start]
        
        # Find t-subsets only covered by this block (cov_count == 1)
        only_covered = frozenset(idx for idx in old_cov if cov_count[idx] == 1)
        
        if not only_covered:
            return False
        
        # Build new block to cover only_covered
        new_block = greedy_build_block(only_covered, None, k)
        if new_block is None:
            return False
        
        new_cov = get_block_coverage(new_block)
        
        # Check if new block covers all of only_covered
        if not only_covered.issubset(new_cov):
            return False
        
        # Accept if it covers at least as much
        if len(new_cov) >= len(old_cov):
            update_cov_count(old_cov, new_cov, 1)
            blocks[start] = new_block
            block_covs[start] = new_cov
            return True
        return False
    
    def try_m_exchange(start, m):
        """Try to replace m consecutive blocks starting at start. Returns True if improved."""
        old_blocks = blocks[start:start+m]
        old_covs = [block_covs[start+i] for i in range(m)]
        
        # Union of old coverages
        union_old = set()
        for cov in old_covs:
            union_old |= cov
        
        # Find t-subsets only covered by these m blocks
        only_covered = set()
        for idx in union_old:
            if cov_count[idx] == m:  # Only covered by these m blocks
                only_covered.add(idx)
        
        if not only_covered:
            return False
        
        # Build m new blocks to cover only_covered
        remaining = set(only_covered)
        new_blocks = []
        
        for _ in range(m):
            nb = greedy_build_block(remaining, None, k)
            if nb is None:
                return False
            nb_set = set(nb)
            new_blocks.append(nb)
            # Update remaining
            new_remaining = set()
            for ts_idx in remaining:
                ts = tsub_points[ts_idx]
                if not ts.issubset(nb_set):
                    new_remaining.add(ts_idx)
            remaining = new_remaining
        
        # Check all only_covered are covered
        if remaining:
            return False
        
        # Compute new coverages
        new_covs = [get_block_coverage(nb) for nb in new_blocks]
        
        # Check total coverage doesn't decrease
        old_total = sum(len(c) for c in old_covs)
        new_total = sum(len(c) for c in new_covs)
        if new_total < old_total:
            return False
        
        # Update cov_count
        for i in range(m):
            update_cov_count(old_covs[i], new_covs[i], 1)
        
        for i in range(m):
            blocks[start + i] = new_blocks[i]
            block_covs[start + i] = new_covs[i]
        
        return True
    
    # VND: try 1-exchange, then 2-for-2, then 3-for-3
    for m in [1, 2, 3]:
        improved = True
        while improved:
            if time.time() - start_time > time_limit:
                break
            improved = False
            for start in range(len(blocks) - m + 1):
                if time.time() - start_time > time_limit:
                    break
                if m == 1:
                    if try_1_exchange(start):
                        improved = True
                        break
                else:
                    if try_m_exchange(start, m):
                        improved = True
                        break
        
        if not improved:
            break
    
    return blocks
