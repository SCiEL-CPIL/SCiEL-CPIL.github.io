"""Stronger greedy + local-search covering for C(v,k,t)."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import sys
import math
from collections import defaultdict
from heapq import heappush, heappop

# We keep the same entry point: construct(v, k, t) -> list of blocks.


def _all_tsubsets(v, t):
    return list(itertools.combinations(range(v), t))


def _subset_to_block_index(subset, blocks):
    """Return list of block indices that contain this t-subset."""
    idx = []
    for bi, b in enumerate(blocks):
        if all(x in b for x in subset):
            idx.append(bi)
    return idx


def _greedy_construct(v, k, t, deadline, rng):
    """
    Greedy construction with randomized tie-breaking.
    Returns (blocks, uncovered_set).
    """
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []

    while uncovered:
        if time.time() > deadline:
            break
        # Pick a random uncovered t-subset as target for diversity
        target = rng.choice(list(uncovered))
        block = list(target)

        # Add points one by one to maximize coverage of uncovered t-subsets
        while len(block) < k:
            if time.time() > deadline:
                break
            best_x = -1
            best_gain = -1
            # For each candidate point, compute how many new t-subsets it covers
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                # New t-subsets formed by adding x to (t-1)-subsets of block
                for rest in itertools.combinations(block, t - 1):
                    new_sub = tuple(sorted(rest + (x,)))
                    if new_sub in uncovered:
                        gain += 1
                if gain > best_gain or (gain == best_gain and rng.random() < 0.3):
                    best_gain = gain
                    best_x = x
            if best_x == -1:
                break
            block.append(best_x)

        block.sort()
        blocks.append(block)
        # Remove all t-subsets covered by this block
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)

    return blocks, uncovered


def _remove_useless_blocks(blocks, v, t):
    """Remove blocks that are redundant (all their t-subsets covered by other blocks)."""
    if not blocks:
        return blocks
    # Build a mapping: for each t-subset, how many blocks cover it
    cover_count = defaultdict(int)
    for b in blocks:
        for sub in itertools.combinations(b, t):
            cover_count[sub] += 1

    new_blocks = []
    for b in blocks:
        # Check if all t-subsets of this block are covered by other blocks
        redundant = True
        for sub in itertools.combinations(b, t):
            if cover_count[sub] <= 1:
                # This subset is only covered by this block
                redundant = False
                break
        if not redundant:
            new_blocks.append(b)

    return new_blocks


def _local_search_1swap(blocks, v, t, deadline, max_iter=200):
    """
    Try to remove redundant blocks one at a time.
    """
    if not blocks:
        return blocks

    # Precompute for each t-subset, how many blocks cover it
    cover_count = defaultdict(int)
    for b in blocks:
        for sub in itertools.combinations(b, t):
            cover_count[sub] += 1

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            # Try removing block i
            can_remove = True
            for sub in itertools.combinations(blocks[i], t):
                if cover_count[sub] <= 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove it
                for sub in itertools.combinations(blocks[i], t):
                    cover_count[sub] -= 1
                blocks.pop(i)
                improved = True
                break  # Restart from beginning
    return blocks


def _try_replace_block(blocks, v, t, deadline, rng):
    """
    Try to replace a block with a better one that covers more uncovered subsets.
    """
    if not blocks or time.time() > deadline:
        return blocks

    # Find uncovered t-subsets
    covered = set()
    for b in blocks:
        for sub in itertools.combinations(b, t):
            covered.add(sub)

    all_subs = set(itertools.combinations(range(v), t))
    uncovered = all_subs - covered

    if not uncovered:
        return blocks

    # Try to improve by replacing each block with one that covers more uncovered
    best_blocks = blocks[:]
    for i in range(len(best_blocks)):
        if time.time() > deadline:
            break
        old_block = best_blocks[i]
        # Compute which uncovered subsets the old block covers
        old_covers = set()
        for sub in itertools.combinations(old_block, t):
            if sub in uncovered:
                old_covers.add(sub)

        # Try to find a new block that covers more uncovered subsets
        # Greedily build a new block
        new_block = []
        remaining_uncovered = uncovered - old_covers
        for _ in range(k):
            if time.time() > deadline:
                break
            best_x = -1
            best_gain = 0
            for x in range(v):
                if x in new_block:
                    continue
                gain = 0
                for rest in itertools.combinations(new_block, t - 1):
                    new_sub = tuple(sorted(rest + (x,)))
                    if new_sub in remaining_uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1:
                break
            new_block.append(best_x)

        new_block.sort()
        # Count how many uncovered subsets the new block covers
        new_covers = 0
        for sub in itertools.combinations(new_block, t):
            if sub in uncovered:
                new_covers += 1

        if new_covers > len(old_covers):
            best_blocks[i] = new_block

    return best_blocks


def _maxsat_local_search(blocks, v, k, t, deadline, rng, max_iter=500):
    """
    Max-SAT-inspired stochastic local search with relaxed swap constraint.
    
    Key improvement: Accept swaps that cover >= 60% of unique_loss (not 100%),
    then rely on a final fill step to restore validity. This enables more
    successful exchanges and better block overlap optimization.
    """
    if not blocks or time.time() > deadline:
        return blocks
    
    n_blocks = len(blocks)
    
    # Precompute: for each t-subset, which blocks cover it
    block_to_subs = [list(itertools.combinations(b, t)) for b in blocks]
    sub_to_blocks = defaultdict(list)
    for bi, subs in enumerate(block_to_subs):
        for sub in subs:
            sub_to_blocks[sub].append(bi)
    
    # Current state: all blocks selected
    selected = [True] * n_blocks
    
    # Count how many selected blocks cover each t-subset
    sub_cover_count = defaultdict(int)
    for sub, blks in sub_to_blocks.items():
        sub_cover_count[sub] = sum(1 for b in blks if selected[b])
    
    best_blocks = blocks[:]
    best_count = n_blocks
    
    # Relaxed threshold: accept swaps covering >= 60% of unique_loss
    swap_threshold = 0.6
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        candidates = [i for i in range(n_blocks) if selected[i]]
        if not candidates:
            break
        
        # Score each candidate by unique_loss (subsets only covered by this block)
        scored_candidates = []
        for i in candidates:
            unique_loss = 0
            for sub in block_to_subs[i]:
                if sub_cover_count[sub] <= 1:
                    unique_loss += 1
            scored_candidates.append((unique_loss, i))
        
        scored_candidates.sort(key=lambda x: (x[0], rng.random()))
        
        tried_any = False
        for unique_loss, i in scored_candidates[:min(5, len(scored_candidates))]:
            if time.time() > deadline:
                break
            
            if unique_loss == 0:
                # Safe to remove
                selected[i] = False
                for sub in block_to_subs[i]:
                    sub_cover_count[sub] -= 1
                best_blocks = [b for j, b in enumerate(blocks) if selected[j]]
                best_count = len(best_blocks)
                tried_any = True
                break
            else:
                # Risky: try swap with relaxed constraint
                if rng.random() < 0.5:  # 50% chance to try risky move
                    old_selected = selected[:]
                    old_cover = dict(sub_cover_count)
                    
                    selected[i] = False
                    for sub in block_to_subs[i]:
                        sub_cover_count[sub] -= 1
                    
                    # Find newly uncovered subsets
                    uncovered = [sub for sub in block_to_subs[i] if sub_cover_count[sub] == 0]
                    uncovered_set = set(uncovered)
                    
                    # Find best replacement block
                    best_new_block = -1
                    best_gain = 0
                    for j in range(n_blocks):
                        if selected[j]:
                            continue
                        gain = sum(1 for sub in block_to_subs[j] if sub in uncovered_set)
                        if gain > best_gain:
                            best_gain = gain
                            best_new_block = j
                    
                    # Relaxed: accept if gain >= swap_threshold * unique_loss
                    if best_new_block >= 0 and best_gain >= swap_threshold * unique_loss:
                        selected[best_new_block] = True
                        for sub in block_to_subs[best_new_block]:
                            sub_cover_count[sub] += 1
                        best_blocks = [b for j, b in enumerate(blocks) if selected[j]]
                        best_count = len(best_blocks)
                        tried_any = True
                        break
                    else:
                        # Rollback
                        selected = old_selected
                        sub_cover_count.clear()
                        sub_cover_count.update(old_cover)
        
        if not tried_any:
            # Stuck: try random safe removal
            if rng.random() < 0.1 and n_blocks > 0:
                i = rng.choice(range(n_blocks))
                if selected[i]:
                    unique_loss = sum(1 for sub in block_to_subs[i] if sub_cover_count[sub] <= 1)
                    if unique_loss == 0:
                        selected[i] = False
                        for sub in block_to_subs[i]:
                            sub_cover_count[sub] -= 1
                        best_blocks = [b for j, b in enumerate(blocks) if selected[j]]
                        best_count = len(best_blocks)
    
    return best_blocks


def _fill_uncovered(blocks, v, k, t, deadline, rng):
    """
    Fill any uncovered t-subsets by adding blocks.
    This ensures validity after relaxed local search.
    """
    # Compute which t-subsets are covered
    covered = set()
    for b in blocks:
        for sub in itertools.combinations(b, t):
            covered.add(sub)
    
    all_subs = set(itertools.combinations(range(v), t))
    uncovered = all_subs - covered
    
    if not uncovered:
        return blocks
    
    # Add blocks to cover uncovered subsets
    blocks = blocks[:]
    while uncovered and time.time() < deadline:
        target = min(uncovered)
        block = list(target)
        
        # Greedily add points to maximize coverage of uncovered
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_sub = tuple(sorted(rest + (x,)))
                    if new_sub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1:
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)
    
    return blocks


def construct(v, k, t):
    start_time = time.time()
    # Safety margin: stop at 115s to be safe within 120s limit
    deadline = start_time + 115.0
    rng = random.Random(42)

    # Phase 1: Greedy construction with multiple restarts
    # Use shorter local search per restart to enable more restarts
    best_blocks = None
    best_count = float('inf')
    
    # More restarts with shorter per-restart budget
    num_restarts = 5
    per_restart_deadline_margin = 15.0  # Leave 15s for final polish
    
    for restart in range(num_restarts):
        if time.time() > deadline - per_restart_deadline_margin:
            break
        
        # Each restart gets a tighter deadline
        restart_deadline = min(deadline - per_restart_deadline_margin, 
                               start_time + 12.0 + restart * 1.5)
        
        rng_restart = random.Random(42 + restart * 1000)
        blocks, uncovered = _greedy_construct(v, k, t, restart_deadline, rng_restart)
        
        if uncovered:
            # Incomplete coverage; try to complete it
            blocks = _fill_uncovered(blocks, v, k, t, restart_deadline, rng_restart)
        
        # Check if valid
        covered = set()
        for b in blocks:
            for sub in itertools.combinations(b, t):
                covered.add(sub)
        all_subs = set(itertools.combinations(range(v), t))
        uncovered = all_subs - covered
        
        if not uncovered:
            # Valid complete covering
            blocks = _remove_useless_blocks(blocks, v, t)
            # Shorter local search per restart
            local_deadline = min(restart_deadline, time.time() + 5.0)
            blocks = _local_search_1swap(blocks, v, t, local_deadline)
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks[:]
    
    # Use the best result if we have one
    if best_blocks is not None:
        blocks = best_blocks
    else:
        # Fallback: greedy + fill
        blocks, _ = _greedy_construct(v, k, t, deadline, rng)
        blocks = _fill_uncovered(blocks, v, k, t, deadline, rng)
        blocks = _remove_useless_blocks(blocks, v, t)

    # Phase 2: Max-SAT-inspired local search with relaxed constraint
    if time.time() < deadline - 10.0:  # Leave 10s margin for fill + cleanup
        blocks = _maxsat_local_search(blocks, v, k, t, deadline - 5.0, rng, max_iter=2000)
    
    # Phase 3: Fill any gaps from relaxed search (ensures validity)
    blocks = _fill_uncovered(blocks, v, k, t, deadline - 2.0, rng)
    
    # Final cleanup: remove redundant blocks
    blocks = _remove_useless_blocks(blocks, v, t)
    
    # One more pass of local search on the cleaned blocks
    if time.time() < deadline - 1.0:
        blocks = _local_search_1swap(blocks, v, t, deadline - 0.5)
        blocks = _remove_useless_blocks(blocks, v, t)

    # Ensure blocks are distinct and sorted
    seen = set()
    unique_blocks = []
    for b in blocks:
        b_sorted = sorted(b)
        key = tuple(b_sorted)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b_sorted)

    return unique_blocks
# EVOLVE-BLOCK-END
