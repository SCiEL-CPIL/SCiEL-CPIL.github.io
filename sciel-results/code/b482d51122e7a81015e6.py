"""Improved greedy covering for C(19,7,4) with deadline guard and vectorized gains."""
# EVOLVE-BLOCK-START
import itertools
import time
import heapq
from collections import defaultdict

import numpy as np


def _all_t_subsets(v, t):
    """Return a list of all t-subsets of {0..v-1} as tuples, plus a mapping."""
    subs = list(itertools.combinations(range(v), t))
    return subs


def _subsets_of_block(block, t):
    """Yield all t-subsets of a block."""
    return list(itertools.combinations(block, t))


def _fast_greedy(v, k, t, deadline):
    """
    Fast greedy: maintain uncovered set; at each step pick a block that
    maximizes the number of newly covered t-subsets.
    
    Strategy:
      1. Precompute all t-subsets and assign each an index.
      2. Maintain a boolean array 'covered' of length C(v,t).
      3. For candidate generation, use a heuristic: pick an uncovered t-subset,
         then build a block around it greedily by adding points that cover
         the most additional uncovered t-subsets.
      4. Use numpy for fast counting of newly covered subsets.
    """
    all_subs = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    
    covered = np.zeros(n_subsets, dtype=bool)
    blocks = []
    
    # Precompute for each point x and each (t-1)-subset, the t-subset formed
    # This helps quickly count how many uncovered subsets a new point would cover.
    
    # For each point x, precompute the list of t-subsets that contain x.
    # subsets_containing[x] = list of indices of t-subsets containing x
    subsets_containing = defaultdict(list)
    for idx, sub in enumerate(all_subs):
        for x in sub:
            subsets_containing[x].append(idx)
    
    # For efficiency, also precompute for each pair of points... but that's large.
    # Instead, use a simpler approach: for a partial block B, the t-subsets covered
    # by adding point x are those t-subsets S where S ⊆ B∪{x} and x ∈ S.
    # This equals: t-subsets of B∪{x} that include x.
    
    while True:
        if time.time() > deadline:
            break
        if covered.all():
            break
        
        # Find an uncovered t-subset to anchor
        # Use a random-ish approach: pick the first uncovered
        uncovered_indices = np.where(~covered)[0]
        if len(uncovered_indices) == 0:
            break
        
        # Pick an anchor: the uncovered subset with the most "flexibility"
        # For speed, just pick the first uncovered one
        anchor_idx = uncovered_indices[0]
        anchor = list(all_subs[anchor_idx])
        
        # Build block greedily: start with anchor, add points one by one
        block_set = set(anchor)
        block = list(anchor)
        
        while len(block) < k:
            if time.time() > deadline:
                break
            
            best_x = -1
            best_gain = -1
            
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                # A t-subset S is covered by B∪{x} if S ⊆ B∪{x} and S contains x
                # i.e., S ∩ B has size t-1 (the other t-1 elements from B)
                
                # For each (t-1)-subset of current block, check if combined with x forms an uncovered t-subset
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    # Form the t-subset: rest + (x,)
                    new_sub = tuple(sorted(rest + (x,)))
                    if new_sub in sub_to_idx:
                        idx = sub_to_idx[new_sub]
                        if not covered[idx]:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1 or best_gain == 0:
                # No point helps; pick an arbitrary unused point
                for x in range(v):
                    if x not in block_set:
                        best_x = x
                        break
                if best_x == -1:
                    break
            
            block.append(best_x)
            block_set.add(best_x)
        
        if len(block) < k:
            # Pad with arbitrary points
            for x in range(v):
                if x not in block_set:
                    block.append(x)
                    block_set.add(x)
                if len(block) == k:
                    break
        
        block.sort()
        
        # Mark all t-subsets of this block as covered
        for sub in itertools.combinations(block, t):
            if sub in sub_to_idx:
                covered[sub_to_idx[sub]] = True
        
        blocks.append(block)
    
    return blocks


def _greedy_with_priority(v, k, t, deadline):
    """
    Alternative: priority-queue based greedy.
    At each step, maintain a heap of candidate blocks ranked by marginal gain.
    """
    all_subs = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    
    covered = np.zeros(n_subsets, dtype=bool)
    blocks = []
    
    # Precompute subsets containing each point
    subsets_containing = defaultdict(list)
    for idx, sub in enumerate(all_subs):
        for x in sub:
            subsets_containing[x].append(idx)
    
    while True:
        if time.time() > deadline:
            break
        if covered.all():
            break
        
        # Find first uncovered
        uncovered_mask = ~covered
        if not uncovered_mask.any():
            break
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            break
        
        anchor_idx = uncovered_indices[0]
        anchor = list(all_subs[anchor_idx])
        
        block_set = set(anchor)
        block = list(anchor)
        
        while len(block) < k:
            if time.time() > deadline:
                break
            
            best_x = -1
            best_gain = -1
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_sub = tuple(sorted(rest + (x,)))
                    idx = sub_to_idx.get(new_sub, -1)
                    if idx >= 0 and not covered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1:
                for x in range(v):
                    if x not in block_set:
                        best_x = x
                        break
                if best_x == -1:
                    break
            
            block.append(best_x)
            block_set.add(best_x)
        
        if len(block) < k:
            for x in range(v):
                if x not in block_set:
                    block.append(x)
                    block_set.add(x)
                if len(block) == k:
                    break
        
        block.sort()
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx.get(sub, -1)
            if idx >= 0:
                covered[idx] = True
        
        blocks.append(block)
    
    return blocks


def _local_search_refine(blocks, v, k, t, deadline, max_rounds=3):
    """
    Try to remove redundant blocks or replace blocks to reduce count.
    Simple approach: check if any block's t-subsets are all covered by other blocks.
    """
    if not blocks:
        return blocks
    
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    n_subsets = len(all_subs)
    
    # For each block, compute the set of t-subset indices it covers
    block_subs = []
    for b in blocks:
        subs = set()
        for sub in itertools.combinations(b, t):
            idx = sub_to_idx.get(sub, -1)
            if idx >= 0:
                subs.add(idx)
        block_subs.append(subs)
    
    # Remove redundant blocks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                return blocks
            # Check if block i's subsets are all covered by other blocks
            others_cover = set()
            for j in range(len(blocks)):
                if j == i:
                    continue
                others_cover |= block_subs[j]
            if block_subs[i] <= others_cover:
                # Remove block i
                blocks.pop(i)
                block_subs.pop(i)
                changed = True
                break
    
    return blocks


def construct(v, k, t):
    start_time = time.time()
    # Deadline with safety margin (150s limit, use 130s for safety)
    deadline = start_time + 130.0
    
    # Run the fast greedy
    blocks = _fast_greedy(v, k, t, deadline)
    
    if time.time() > deadline:
        return blocks
    
    # Try local search refinement to remove redundant blocks
    blocks = _local_search_refine(blocks, v, k, t, deadline, max_rounds=3)
    
    return blocks
# EVOLVE-BLOCK-END
