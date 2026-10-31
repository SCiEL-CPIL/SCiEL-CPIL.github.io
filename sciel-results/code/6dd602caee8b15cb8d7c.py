"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import math
import numpy as np


def _build_tsub_index(v, t):
    """Build mapping from t-subsets to indices."""
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {tsub: i for i, tsub in enumerate(all_tsubs)}
    return all_tsubs, tsub_to_idx, len(all_tsubs)


def _greedy_construct(v, k, t, all_tsubs, tsub_to_idx, num_tsubs, deadline, rng):
    """Greedy construction with randomization."""
    uncovered = np.ones(num_tsubs, dtype=bool)
    blocks = []
    uncovered_count = num_tsubs
    
    point_order = list(range(v))
    rng.shuffle(point_order)
    
    while uncovered_count > 0:
        if time.time() > deadline - 10:
            return None, None
        
        # Find an uncovered t-subset
        start_idx = -1
        for i in range(num_tsubs):
            if uncovered[i]:
                start_idx = i
                break
        if start_idx == -1:
            break
        
        initial_tsub = all_tsubs[start_idx]
        block = set(initial_tsub)
        uncovered[start_idx] = False
        uncovered_count -= 1
        
        # Greedily add points
        while len(block) < k:
            if time.time() > deadline - 10:
                return None, None
            
            best_gain = -1
            best_point = -1
            block_sorted = sorted(block)
            
            for x in point_order:
                if x in block:
                    continue
                
                gain = 0
                for rest in itertools.combinations(block_sorted, t - 1):
                    new_tsub = tuple(sorted(rest + (x,)))
                    if new_tsub in tsub_to_idx:
                        idx = tsub_to_idx[new_tsub]
                        if uncovered[idx]:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                elif gain == best_gain and best_point != -1:
                    if rng.random() < 0.5:
                        best_point = x
            
            if best_point == -1 or best_gain <= 0:
                break
            
            block.add(best_point)
            block_sorted = sorted(block)
            
            for tsub in itertools.combinations(block_sorted, t):
                idx = tsub_to_idx[tsub]
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
        
        if len(block) == k:
            blocks.append(sorted(block))
        else:
            break
    
    if uncovered_count == 0:
        return blocks, uncovered
    return None, None


def _local_optimize(blocks, v, k, t, all_tsubs, tsub_to_idx, num_tsubs, deadline):
    """Local optimization: try removing redundant blocks and replacing blocks."""
    if not blocks:
        return blocks
    
    # Track coverage count for each t-subset
    coverage_count = np.zeros(num_tsubs, dtype=np.int32)
    for block in blocks:
        for tsub in itertools.combinations(block, t):
            coverage_count[tsub_to_idx[tsub]] += 1
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Try removing each block if it doesn't uncover any t-subset
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            
            block = blocks[i]
            # Check if removing this block uncovers any t-subset
            can_remove = True
            for tsub in itertools.combinations(block, t):
                idx = tsub_to_idx[tsub]
                if coverage_count[idx] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                for tsub in itertools.combinations(block, t):
                    coverage_count[tsub_to_idx[tsub]] -= 1
                blocks.pop(i)
                improved = True
        
        # Try replacing blocks: for each block, see if we can find a better one
        for i in range(len(blocks)):
            if time.time() > deadline - 5:
                break
            
            current_block = set(blocks[i])
            # Count how many t-subsets are only covered by this block
            unique_coverage = 0
            for tsub in itertools.combinations(sorted(current_block), t):
                idx = tsub_to_idx[tsub]
                if coverage_count[idx] == 1:
                    unique_coverage += 1
            
            # Try to find a replacement that covers the same unique t-subsets
            # with potentially fewer total blocks (this is complex, so skip for now)
    
    return blocks


def construct(v, k, t):
    """
    Greedy covering design C(v, k, t) with deadline guard and local optimization.
    """
    deadline = time.time() + 110  # safety margin before 120s limit
    rng = random.Random(42)
    
    # Precompute all t-subsets
    all_tsubs, tsub_to_idx, num_tsubs = _build_tsub_index(v, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts with different random seeds
    max_restarts = 30
    restart = 0
    
    while restart < max_restarts:
        if time.time() > deadline - 15:
            break
        
        rng = random.Random(hash((restart, time.time())) % (2**32))
        
        blocks, uncovered = _greedy_construct(
            v, k, t, all_tsubs, tsub_to_idx, num_tsubs, deadline, rng
        )
        
        if blocks is not None:
            # Apply local optimization
            optimized_blocks = _local_optimize(
                blocks, v, k, t, all_tsubs, tsub_to_idx, num_tsubs, deadline
            )
            
            if len(optimized_blocks) < best_count:
                best_count = len(optimized_blocks)
                best_blocks = optimized_blocks
        
        restart += 1
    
    # If we never found a valid covering, fall back to simple greedy
    if best_blocks is None:
        uncovered_set = set(all_tsubs)
        blocks = []
        while uncovered_set and time.time() < deadline - 1:
            block = list(min(uncovered_set))
            while len(block) < k and time.time() < deadline - 1:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered_set)
                candidates = [x for x in range(v) if x not in block]
                if not candidates:
                    break
                best_x = max(candidates, key=gain)
                block.append(best_x)
            
            if len(block) == k:
                block.sort()
                blocks.append(block)
                uncovered_set.difference_update(itertools.combinations(block, t))
        
        if blocks:
            best_blocks = _local_optimize(
                blocks, v, k, t, all_tsubs, tsub_to_idx, num_tsubs, deadline
            )
        else:
            best_blocks = []
    
    return best_blocks
# EVOLVE-BLOCK-END
