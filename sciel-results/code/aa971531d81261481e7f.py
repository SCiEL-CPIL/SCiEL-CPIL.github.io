"""Bitset-accelerated greedy covering C(24,7,3) with local search pruning.

Uses a bitmask to track uncovered 3-subsets for fast greedy selection.
After the initial greedy construction, it performs a local search phase
to remove redundant blocks, minimizing the total count.
"""
# EVOLVE-BLOCK-START
import itertools
import time


def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0  # keep a safety margin under the 120 s limit

    # --- build id table for all t-subsets -------------------------------
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    id_of = {}
    for idx, combo in enumerate(all_t):
        id_of[combo] = idx

    # --- precompute coverage data for every candidate k-block ------------
    # masks[block] = bitmask of covered t-subsets
    # block_to_ids[block] = list of t-subset ids covered by block
    masks = {}
    block_to_ids = {}
    
    for block in itertools.combinations(range(v), k):
        m = 0
        ids = []
        for sub in itertools.combinations(block, t):
            idx = id_of[sub]
            m |= 1 << idx
            ids.append(idx)
        masks[block] = m
        block_to_ids[block] = ids

    # Prepare candidates list for faster iteration in greedy loop
    # List of (mask, block) tuples
    candidates = [(m, b) for b, m in masks.items()]

    # --- Phase 1: Greedy Set Cover ---------------------------------------
    uncovered = (1 << n_t) - 1
    selected_blocks = []
    used = set()

    while uncovered:
        if time.time() >= deadline - 10.0:  # Leave time for pruning
            break
        
        best_gain = 0
        best_block = None
        
        for mask, block in candidates:
            if block in used:
                continue
            gain = (uncovered & mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block = block
                if gain == n_t:
                    break
        
        if best_block is None or best_gain == 0:
            break
            
        used.add(best_block)
        selected_blocks.append(best_block)
        uncovered &= ~masks[best_block]

    # --- Phase 2: Local Search Pruning -----------------------------------
    # Try to remove blocks that are not essential.
    # A block can be removed if every t-subset it covers is also covered by
    # at least one other selected block.
    
    if selected_blocks and time.time() < deadline - 2.0:
        # Initialize coverage counts
        coverage_counts = [0] * n_t
        for b in selected_blocks:
            for idx in block_to_ids[b]:
                coverage_counts[idx] += 1
        
        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            # Iterate through a copy of indices to allow removal
            for i in range(len(selected_blocks)):
                b = selected_blocks[i]
                can_remove = True
                for idx in block_to_ids[b]:
                    if coverage_counts[idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove block i
                    selected_blocks.pop(i)
                    for idx in block_to_ids[b]:
                        coverage_counts[idx] -= 1
                    changed = True
                    break # Restart scan as counts changed

    return [list(b) for b in selected_blocks]
# EVOLVE-BLOCK-END
