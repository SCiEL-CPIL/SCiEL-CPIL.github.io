# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

import numpy as np

def _bitmask_combinations(v, k):
    """Yield bitmasks for all k-subsets of {0..v-1}."""
    for combo in itertools.combinations(range(v), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        yield mask

def _greedy_cover(v, k, t, deadline, rng):
    """Vectorized greedy covering with bitmask coverage tracking."""
    total_t = 0
    for i in range(v - t + 1):
        from math import comb
        total_t += comb(v - i - 1, t - 1)
    # Actually compute C(v,t)
    from math import comb
    total_t = comb(v, t)
    
    # Generate all t-subsets as bitmasks
    t_masks = list(_bitmask_combinations(v, t))
    n_t = len(t_masks)
    
    # Map each t-mask to an index
    t_to_idx = {m: i for i, m in enumerate(t_masks)}
    
    # Coverage count per t-subset
    cover_count = np.zeros(n_t, dtype=np.int32)
    
    blocks = []
    block_masks = []
    
    # Precompute for each point, the list of t-mask indices that include it?
    # That's expensive. Instead, for a block, compute all t-subsets within it.
    
    def block_t_indices(block_set):
        """Given a set of points, return list of t-mask indices covered."""
        indices = []
        for combo in itertools.combinations(block_set, t):
            mask = 0
            for x in combo:
                mask |= (1 << x)
            idx = t_to_idx.get(mask)
            if idx is not None:
                indices.append(idx)
        return indices
    
    # Greedy: repeatedly pick the block that covers the most uncovered t-subsets
    # To speed up, use a heuristic: start with random point, then greedily add points
    # that maximize newly covered t-subsets.
    
    # Precompute: for each pair of points, which t-subsets contain both? Too complex.
    # Instead: for a partial block (set of points), adding point x covers all t-subsets
    # that contain x and (t-1) other points from the current block.
    
    # For efficiency, precompute for each point x and each (t-1)-subset S of other points,
    # the t-mask index of S ∪ {x}.
    # This is C(v, t-1) * v entries. For v=19, t=5: C(19,4)*19 = 3876*19 = 73644. Manageable.
    
    # Precompute: for each point x, a dict mapping (t-1)-subset (as bitmask) to t-mask index
    # Actually, for a given partial block B and candidate x, the new t-subsets covered are
    # those of form S ∪ {x} where S is a (t-1)-subset of B.
    
    # Precompute t_minus_1 masks for each point
    # For each point x, precompute a dict: for each (t-1)-subset S not containing x,
    # map S_mask -> t_mask_index of S ∪ {x}
    
    # This is C(v-1, t-1) per point, total v*C(v-1,t-1) = v * C(19,4) = 19*3876 = 73644
    # For each (x, S), we need the t-mask index.
    
    # Let's precompute: point_tmap[x] = dict from S_mask to t_idx
    point_tmap = [dict() for _ in range(v)]
    for x in range(v):
        for S in itertools.combinations([y for y in range(v) if y != x], t - 1):
            S_mask = 0
            for y in S:
                S_mask |= (1 << y)
            full_mask = S_mask | (1 << x)
            idx = t_to_idx.get(full_mask)
            if idx is not None:
                point_tmap[x][S_mask] = idx
    
    uncovered_count = n_t
    start_time = time.time()
    
    while uncovered_count > 0:
        if time.time() > deadline - 15:
            break
        
        # Start a new block with a random point (or the point in the first uncovered t-subset)
        # Find an uncovered t-subset
        uncovered_idx = None
        for i in range(n_t):
            if cover_count[i] == 0:
                uncovered_idx = i
                break
        if uncovered_idx is None:
            break
        
        # Get the t-subset
        uncovered_mask = t_masks[uncovered_idx]
        # Pick a random point from this subset to start
        pts_in_subset = [i for i in range(v) if (uncovered_mask >> i) & 1]
        start_pt = rng.choice(pts_in_subset)
        
        block_set = {start_pt}
        
        # Greedily add points to maximize coverage
        while len(block_set) < k:
            best_x = -1
            best_gain = -1
            # Current (t-1)-subsets of block_set
            t_minus_1_subsets = list(itertools.combinations(block_set, t - 1))
            t_minus_1_masks = []
            for S in t_minus_1_subsets:
                mask = 0
                for y in S:
                    mask |= (1 << y)
                t_minus_1_masks.append(mask)
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for S_mask in t_minus_1_masks:
                    idx = point_tmap[x].get(S_mask)
                    if idx is not None and cover_count[idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1 or best_gain <= 0:
                # Pick a random point not in block
                candidates = [x for x in range(v) if x not in block_set]
                if candidates:
                    best_x = rng.choice(candidates)
                else:
                    break
            
            block_set.add(best_x)
        
        block = sorted(block_set)
        blocks.append(block)
        block_masks.append(block_set)
        
        # Update coverage
        for idx in block_t_indices(block_set):
            if cover_count[idx] == 0:
                uncovered_count -= 1
            cover_count[idx] += 1
    
    return blocks, cover_count, t_masks, t_to_idx, point_tmap, uncovered_count

def _local_search(blocks, v, k, t, deadline, rng, cover_count, t_masks, t_to_idx, point_tmap):
    """Try to remove redundant blocks and improve via swaps."""
    n_t = len(t_masks)
    n_blocks = len(blocks)
    
    # Recompute cover_count from scratch for safety
    cover_count = np.zeros(n_t, dtype=np.int32)
    for block in blocks:
        block_set = set(block)
        for combo in itertools.combinations(block_set, t):
            mask = 0
            for x in combo:
                mask |= (1 << x)
            idx = t_to_idx.get(mask)
            if idx is not None:
                cover_count[idx] += 1
    
    # Remove redundant blocks
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        new_blocks = []
        for i, block in enumerate(blocks):
            # Temporarily remove this block
            block_set = set(block)
            # Check if all t-subsets in this block are still covered
            redundant = True
            for combo in itertools.combinations(block_set, t):
                mask = 0
                for x in combo:
                    mask |= (1 << x)
                idx = t_to_idx.get(mask)
                if idx is not None and cover_count[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove it
                for combo in itertools.combinations(block_set, t):
                    mask = 0
                    for x in combo:
                        mask |= (1 << x)
                    idx = t_to_idx.get(mask)
                    if idx is not None:
                        cover_count[idx] -= 1
                changed = True
            else:
                new_blocks.append(block)
        blocks = new_blocks
    
    # Try swaps: replace one block with another that covers more uncovered or allows removal
    # This is complex; skip for now and just do redundant removal
    
    return blocks, cover_count

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    rng = random.Random(42)
    
    # First, do a greedy construction
    blocks, cover_count, t_masks, t_to_idx, point_tmap, uncovered_count = _greedy_cover(
        v, k, t, deadline, rng
    )
    
    if time.time() > deadline - 10:
        return blocks
    
    # Local search: remove redundant blocks
    blocks, cover_count = _local_search(
        blocks, v, k, t, deadline, rng, cover_count, t_masks, t_to_idx, point_tmap
    )
    
    return blocks
# EVOLVE-BLOCK-END
