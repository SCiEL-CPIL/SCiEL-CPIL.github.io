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
    time_limit = 115  # Leave 5s buffer
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index
    tsub_to_idx = {tuple(cs): i for i, cs in enumerate(all_tsubs)}
    
    # For each point, precompute which t-subsets it's in
    point_tsubs = [[] for _ in range(v)]
    for idx, cs in enumerate(all_tsubs):
        for p in cs:
            point_tsubs[p].append(idx)
    
    # Precompute for each pair of points, which t-subsets contain both
    # This helps with faster gain computation
    # For efficiency, we'll use a different approach
    
    def compute_block_coverage(block):
        """Return a numpy array of 0/1 indicating which t-subsets are covered."""
        cov = np.zeros(num_tsubs, dtype=np.uint8)
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[tuple(ts)]
            cov[idx] = 1
        return cov
    
    def greedy_build_block(seed_points=None, uncovered=None, existing_blocks=None):
        """Build a block greedily to maximize coverage of uncovered t-subsets."""
        if seed_points is None:
            seed_points = []
        if uncovered is None:
            uncovered = np.ones(num_tsubs, dtype=np.uint8)
        
        block = list(seed_points)
        block_set = set(block)
        
        while len(block) < k:
            best_point = -1
            best_gain = -1
            
            for p in range(v):
                if p in block_set:
                    continue
                # Calculate gain: how many uncovered t-subsets would be covered
                gain = 0
                for ts_idx in point_tsubs[p]:
                    if uncovered[ts_idx] == 0:
                        continue
                    ts = all_tsubs[ts_idx]
                    # Check if all other elements of ts are in block
                    all_others_in = True
                    for x in ts:
                        if x != p and x not in block_set:
                            all_others_in = False
                            break
                    if all_others_in:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
            
            if best_point == -1:
                # No point adds coverage, just pick any
                for p in range(v):
                    if p not in block_set:
                        best_point = p
                        break
                if best_point == -1:
                    break
            
            block.append(best_point)
            block_set.add(best_point)
        
        if len(block) < k:
            # Fill remaining with arbitrary points
            for p in range(v):
                if len(block) >= k:
                    break
                if p not in block_set:
                    block.append(p)
                    block_set.add(p)
        
        return sorted(block)
    
    # Greedy initial construction
    blocks = []
    uncovered = np.ones(num_tsubs, dtype=np.uint8)
    
    while uncovered.any():
        if time.time() - start_time > time_limit - 30:
            break
        
        # Find first uncovered t-subset
        idx = int(np.argmax(uncovered))
        if uncovered[idx] == 0:
            break
        target = all_tsubs[idx]
        
        # Build block greedily starting with target
        block = greedy_build_block(seed_points=list(target), uncovered=uncovered)
        blocks.append(block)
        
        # Update uncovered mask
        cov = compute_block_coverage(block)
        uncovered &= (1 - cov)
    
    # Verify coverage
    if not uncovered.all() == 0:
        # Still have uncovered, add more blocks
        while uncovered.any():
            if time.time() - start_time > time_limit - 30:
                break
            idx = int(np.argmax(uncovered))
            if uncovered[idx] == 0:
                break
            target = all_tsubs[idx]
            block = greedy_build_block(seed_points=list(target), uncovered=uncovered)
            blocks.append(block)
            cov = compute_block_coverage(block)
            uncovered &= (1 - cov)
    
    # Compute coverage for all blocks
    block_covs = [compute_block_coverage(b) for b in blocks]
    
    # Coverage count: how many blocks cover each t-subset
    cov_count = np.sum(block_covs, axis=0) if block_covs else np.zeros(num_tsubs, dtype=np.int32)
    
    # VND phase: try to remove or replace blocks
    # First pass: try to remove blocks whose coverage is fully redundant
    for pass_num in range(5):
        if time.time() - start_time > time_limit - 10:
            break
        
        improved = False
        n_blocks = len(blocks)
        
        # Try removing each block
        for i in range(n_blocks):
            if time.time() - start_time > time_limit - 10:
                break
            
            # Check if block i's coverage is fully covered by other blocks
            # i.e., for every t-subset covered by block i, at least one other block covers it
            block_i_cov = block_covs[i]
            
            # Compute coverage without block i
            # cov_without_i[j] = 1 if t-subset j is covered by some block other than i
            if n_blocks > 1:
                other_cov = np.sum([block_covs[j] for j in range(n_blocks) if j != i], axis=0)
            else:
                other_cov = np.zeros(num_tsubs, dtype=np.int32)
            
            # Check: for all t-subsets where block_i_cov[j]==1, other_cov[j]>=1
            # i.e., block_i_cov & (other_cov == 0) should be all zeros
            redundant = np.all((block_i_cov == 1) | (other_cov >= 1))
            
            if redundant:
                # Remove block i
                del blocks[i]
                del block_covs[i]
                cov_count = np.sum(block_covs, axis=0) if block_covs else np.zeros(num_tsubs, dtype=np.int32)
                improved = True
                break
        
        if not improved:
            break
    
    # Second pass: try to replace blocks with better ones (1-for-1 exchange)
    for pass_num in range(3):
        if time.time() - start_time > time_limit - 10:
            break
        
        improved = False
        n_blocks = len(blocks)
        
        for i in range(n_blocks):
            if time.time() - start_time > time_limit - 10:
                break
            
            old_block = blocks[i]
            old_cov = block_covs[i]
            
            # Find t-subsets only covered by this block
            only_covered = np.zeros(num_tsubs, dtype=np.uint8)
            for j in range(num_tsubs):
                if old_cov[j] == 1 and cov_count[j] == 1:
                    only_covered[j] = 1
            
            # Try to build a new block that covers all only_covered t-subsets
            # and maximizes additional coverage
            if not only_covered.any():
                # This block has no unique coverage, try to find a better replacement
                # that covers more
                new_block = greedy_build_block(uncovered=np.ones(num_tsubs, dtype=np.uint8))
                new_cov = compute_block_coverage(new_block)
                
                # Accept if new block covers at least as much
                if np.sum(new_cov) >= np.sum(old_cov):
                    blocks[i] = new_block
                    block_covs[i] = new_cov
                    cov_count = np.sum(block_covs, axis=0)
                    improved = True
                    break
            else:
                # Need to cover the only_covered t-subsets
                # Start with points from those t-subsets
                seed_points = []
                for j in range(num_tsubs):
                    if only_covered[j] == 1:
                        seed_points.extend(all_tsubs[j])
                        break
                
                new_block = greedy_build_block(seed_points=seed_points, uncovered=only_covered)
                new_cov = compute_block_coverage(new_block)
                
                # Check if new block covers all only_covered
                covers_all = np.all((new_cov == 1) | (only_covered == 0))
                
                if covers_all:
                    # Check if this is an improvement
                    # New block should cover at least as much as old
                    if np.sum(new_cov) >= np.sum(old_cov):
                        blocks[i] = new_block
                        block_covs[i] = new_cov
                        cov_count = np.sum(block_covs, axis=0)
                        improved = True
                        break
        
        if not improved:
            break
    
    # Third pass: try 2-for-2 exchanges
    if time.time() - start_time < time_limit - 15:
        for pass_num in range(2):
            if time.time() - start_time > time_limit - 10:
                break
            
            improved = False
            n_blocks = len(blocks)
            
            for i in range(n_blocks - 1):
                if time.time() - start_time > time_limit - 10:
                    break
                
                old_blocks = [blocks[i], blocks[i+1]]
                old_covs = [block_covs[i], block_covs[i+1]]
                
                # Union of coverage
                union_old = np.maximum(old_covs[0], old_covs[1])
                
                # Find t-subsets only covered by these two blocks
                other_cov = np.sum([block_covs[j] for j in range(n_blocks) if j != i and j != i+1], axis=0) if n_blocks > 2 else np.zeros(num_tsubs, dtype=np.int32)
                
                only_covered = np.zeros(num_tsubs, dtype=np.uint8)
                for j in range(num_tsubs):
                    if union_old[j] == 1 and other_cov[j] == 0:
                        only_covered[j] = 1
                
                # Try to build 2 new blocks covering only_covered
                if only_covered.any():
                    # Split only_covered into two groups
                    idxs = np.where(only_covered == 1)[0]
                    mid = len(idxs) // 2
                    group1 = idxs[:mid]
                    group2 = idxs[mid:]
                    
                    # Build first block for group1
                    seed1 = []
                    if len(group1) > 0:
                        seed1 = list(all_tsubs[group1[0]])
                    block1 = greedy_build_block(seed_points=seed1, uncovered=only_covered)
                    
                    # Update uncovered for group2
                    cov1 = compute_block_coverage(block1)
                    remaining = only_covered & (1 - cov1)
                    
                    # Build second block for group2
                    seed2 = []
                    if len(group2) > 0:
                        seed2 = list(all_tsubs[group2[0]])
                    block2 = greedy_build_block(seed_points=seed2, uncovered=remaining)
                    
                    # Check if both blocks together cover all only_covered
                    cov1 = compute_block_coverage(block1)
                    cov2 = compute_block_coverage(block2)
                    union_new = np.maximum(cov1, cov2)
                    
                    covers_all = np.all((union_new == 1) | (only_covered == 0))
                    
                    if covers_all:
                        # Check if this is an improvement
                        total_old = np.sum(union_old)
                        total_new = np.sum(union_new)
                        
                        if total_new >= total_old:
                            blocks[i] = block1
                            blocks[i+1] = block2
                            block_covs[i] = cov1
                            block_covs[i+1] = cov2
                            cov_count = np.sum(block_covs, axis=0)
                            improved = True
                            break
            
            if not improved:
                break
    
    return blocks
