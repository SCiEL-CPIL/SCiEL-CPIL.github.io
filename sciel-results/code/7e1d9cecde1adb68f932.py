"""Covering design C(v,k,t) via greedy + local search (block swaps/removals)."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin under 120s (and 150s kill)

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    # Map each t-subset to an index
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}

    # We'll track coverage as a list of ints: cover_count[i] = number of blocks covering subset i
    # Initially 0. We add blocks and increment.

    def time_ok():
        return time.time() < deadline

    # --- Phase 1: Greedy set cover ---
    uncovered = set(range(n_subsets))  # indices of uncovered t-subsets
    blocks = []

    # For speed, precompute for each point x, the list of t-subsets containing x
    # point_subsets[x] = list of subset indices that contain x
    point_subsets = [[] for _ in range(v)]
    for idx, s in enumerate(all_t_subsets):
        for x in s:
            point_subsets[x].append(idx)

    while uncovered and time_ok():
        # Pick the uncovered subset with minimum... actually pick one to anchor a block
        # Use the first uncovered (arbitrary) or the one with fewest candidate points?
        # Standard greedy: pick uncovered subset, then build block greedily.
        # Better: pick the uncovered t-subset that has the fewest points not yet "used heavily"
        # For simplicity and speed, pick an uncovered subset and build around it.
        
        # Choose an uncovered subset (the one with smallest index for determinism, 
        # but randomization can help; use min for now)
        anchor_idx = min(uncovered)
        anchor = all_t_subsets[anchor_idx]
        
        # Build a block containing anchor, adding points one by one
        block_set = set(anchor)
        while len(block_set) < k and time_ok():
            # For each candidate point x not in block_set, compute how many uncovered subsets
            # would be covered by adding x to the current partial block
            best_x = None
            best_gain = -1
            partial = list(block_set)
            for x in range(v):
                if x in block_set:
                    continue
                # Count uncovered t-subsets that contain x and (t-1) elements from partial
                # i.e., subsets s where x in s and |s ∩ partial| >= t-1
                # Since s has size t, this means s ⊆ partial ∪ {x} and x in s
                # Equivalent: s = {x} ∪ r where r ⊂ partial, |r| = t-1
                gain = 0
                for r in itertools.combinations(partial, t - 1):
                    s = tuple(sorted(r + (x,)))
                    if s in subset_to_idx:
                        si = subset_to_idx[s]
                        if si in uncovered:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None or best_gain == 0:
                # No point adds coverage; just pick an arbitrary unused point
                for x in range(v):
                    if x not in block_set:
                        best_x = x
                        break
                if best_x is None:
                    break
            block_set.add(best_x)
        
        if len(block_set) < k:
            # Pad with arbitrary points
            for x in range(v):
                if x not in block_set:
                    block_set.add(x)
                    if len(block_set) == k:
                        break
        
        block = sorted(block_set)
        blocks.append(block)
        
        # Update uncovered
        for s in itertools.combinations(block, t):
            si = subset_to_idx[s]
            uncovered.discard(si)

    if not uncovered:
        # We have a valid covering; proceed to local search
        pass
    else:
        # If time ran out during greedy, we still return what we have (may be invalid)
        # But let's try to finish quickly if possible
        while uncovered and time.time() < deadline - 5:
            anchor_idx = min(uncovered)
            anchor = all_t_subsets[anchor_idx]
            block_set = set(anchor)
            while len(block_set) < k:
                best_x = None
                best_gain = -1
                partial = list(block_set)
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for r in itertools.combinations(partial, t - 1):
                        s = tuple(sorted(r + (x,)))
                        if s in subset_to_idx and subset_to_idx[s] in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    for x in range(v):
                        if x not in block_set:
                            best_x = x
                            break
                block_set.add(best_x)
            block = sorted(block_set)
            blocks.append(block)
            for s in itertools.combinations(block, t):
                uncovered.discard(subset_to_idx[s])

    # --- Phase 2: Local search - remove redundant blocks ---
    # Build cover_count array
    cover_count = [0] * n_subsets
    for block in blocks:
        for s in itertools.combinations(block, t):
            cover_count[subset_to_idx[s]] += 1

    # Try to remove blocks that are redundant (all their subsets are covered by other blocks)
    removed_any = True
    while removed_any and time_ok():
        removed_any = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            # Check if block i is redundant
            redundant = True
            for s in itertools.combinations(blocks[i], t):
                si = subset_to_idx[s]
                if cover_count[si] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for s in itertools.combinations(blocks[i], t):
                    cover_count[subset_to_idx[s]] -= 1
                blocks.pop(i)
                removed_any = True

    # --- Phase 3: Local search - block swaps (replace one block with another to reduce total) ---
    # Try replacing a block with a different block that covers at least the same uncovered subsets
    # and allows removal of another block, or simply try random restarts of greedy on subsets.
    
    # A simpler effective strategy: try to find a block that can replace two existing blocks.
    # For each pair of blocks, check if there's a single block covering all subsets that only those two cover.
    # This is expensive; instead, do a limited number of swap attempts.
    
    # Alternative: random local search - pick a block, remove it, then greedily add blocks to restore coverage,
    # and accept if we end up with fewer or equal blocks.
    
    random.seed(42)
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    # Limit number of swap attempts based on time
    max_swaps = 50
    swap_attempts = 0
    while time.time() < deadline - 20 and swap_attempts < max_swaps:
        swap_attempts += 1
        if len(blocks) <= 1:
            break
        # Pick a random block to try removing
        idx = random.randrange(len(blocks))
        candidate_block = blocks[idx]
        
        # Temporarily remove it
        for s in itertools.combinations(candidate_block, t):
            cover_count[subset_to_idx[s]] -= 1
        
        # Find uncovered subsets
        newly_uncovered = [i for i in range(n_subsets) if cover_count[i] == 0]
        
        if not newly_uncovered:
            # Block was redundant; keep it removed
            blocks.pop(idx)
            best_blocks = list(blocks)
            best_count = len(blocks)
            continue
        
        # Try to cover newly_uncovered with one new block
        # Find the block that covers the most of newly_uncovered
        best_new_block = None
        best_new_gain = -1
        # Sample some candidate blocks
        candidates = set()
        for su in newly_uncovered:
            sub = all_t_subsets[su]
            for x in sub:
                # Candidate blocks must contain sub; generate one
                pass
        # Instead, greedily build a block around the first uncovered subset
        anchor_idx = newly_uncovered[0]
        anchor = all_t_subsets[anchor_idx]
        block_set = set(anchor)
        while len(block_set) < k:
            best_x = None
            best_gain = -1
            partial = list(block_set)
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for r in itertools.combinations(partial, t - 1):
                    s = tuple(sorted(r + (x,)))
                    if s in subset_to_idx:
                        si = subset_to_idx[s]
                        if si in set(newly_uncovered):
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                for x in range(v):
                    if x not in block_set:
                        best_x = x
                        break
            block_set.add(best_x)
        
        new_block = sorted(block_set)
        
        # Check if this new block covers all newly_uncovered
        new_block_subsets = set(subset_to_idx[s] for s in itertools.combinations(new_block, t))
        if set(newly_uncovered).issubset(new_block_subsets):
            # Accept the swap
            blocks.append(new_block)
            for si in new_block_subsets:
                cover_count[si] += 1
            best_blocks = list(blocks)
            best_count = len(blocks)
        else:
            # Reject; restore cover_count
            for s in itertools.combinations(candidate_block, t):
                cover_count[subset_to_idx[s]] += 1

    # Use the best found
    if len(blocks) < best_count:
        best_blocks = list(blocks)
    elif len(blocks) == best_count:
        best_blocks = list(blocks)
    
    return best_blocks
# EVOLVE-BLOCK-END
