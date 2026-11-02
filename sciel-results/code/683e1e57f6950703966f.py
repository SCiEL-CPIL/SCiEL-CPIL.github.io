"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _greedy_vectorized(v, k, t, seed_subset, rng, deadline):
    """Vectorized greedy: use bitmasks for fast gain computation."""
    # Precompute: for each point x, a bitmask of all t-subsets containing x
    # Represent each t-subset as an index in a flat array
    # Total t-subsets: C(v, t)
    n_subsets = 1
    for i in range(t):
        n_subsets = n_subsets * (v - i) // (i + 1)
    
    # Map each t-subset to an index
    sub_to_idx = {}
    idx = 0
    for sub in itertools.combinations(range(v), t):
        sub_to_idx[sub] = idx
        idx += 1
    
    # For each point, precompute which t-subsets it belongs to (as a bitmask)
    # Use a numpy array of uint64 bitmasks, but n_subsets can be large (C(19,5)=11628)
    # So we use a different approach: for each point x, store a set of indices
    point_subsets = [set() for _ in range(v)]
    for sub, idx in sub_to_idx.items():
        for x in sub:
            point_subsets[x].add(idx)
    
    # Convert to numpy arrays for fast intersection
    point_masks = [np.zeros(n_subsets, dtype=np.uint8) for _ in range(v)]
    for x in range(v):
        for idx in point_subsets[x]:
            point_masks[x][idx] = 1
    
    # Uncovered: numpy boolean array
    uncovered = np.ones(n_subsets, dtype=np.uint8)
    
    # Initialize block with seed_subset
    block = list(seed_subset)
    blocks = []
    
    # Mark seed t-subsets as covered (those within seed_subset)
    if len(seed_subset) >= t:
        for sub in itertools.combinations(seed_subset, t):
            uncovered[sub_to_idx[sub]] = 0
    
    # Greedily add points
    while len(block) < k:
        block_set = set(block)
        best_gain = -1
        best_x = -1
        for x in range(v):
            if x in block_set:
                continue
            # Gain = number of uncovered t-subsets that contain x AND are subsets of block∪{x}
            # A t-subset S is covered by block∪{x} iff S ⊆ block∪{x}
            # Since |block| < k and we add one point, S must contain x and S\{x} ⊆ block
            # So: S contains x, and all other elements of S are in block
            # Equivalently: S ∈ point_subsets[x] and S\{x} ⊆ block
            # Count uncovered ones among those
            gain = 0
            for idx in point_subsets[x]:
                if uncovered[idx]:
                    sub = [i for i in range(v) if (idx and True)]  # need to recover sub
                    # Actually, let's just check: is sub ⊆ block ∪ {x}?
                    # We need the actual subset. Let's precompute the reverse mapping.
                    pass
            # Simpler: iterate over C(|block|, t-1) subsets of block, form sub with x
            if len(block) >= t - 1:
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    if sub in sub_to_idx:
                        idx = sub_to_idx[sub]
                        if uncovered[idx]:
                            gain += 1
            if gain > best_gain:
                best_gain = gain
                best_x = x
        if best_x >= 0:
            block.append(best_x)
            block_set.add(best_x)
            # Mark newly covered t-subsets
            if len(block) >= t:
                for sub in itertools.combinations(block, t):
                    uncovered[sub_to_idx[sub]] = 0
    
    block.sort()
    blocks.append(block)
    
    # Mark all t-subsets in block as covered
    if len(block) >= t:
        for sub in itertools.combinations(block, t):
            uncovered[sub_to_idx[sub]] = 0
    
    # Continue until all covered or deadline
    while np.any(uncovered) and time.time() < deadline:
        # Pick first uncovered subset as seed
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
        # Convert index back to subset (slow, but we can precompute)
        # For speed, just pick a random uncovered index
        chosen_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
        # We need the subset for this index. Precompute idx_to_sub.
        # This is getting complex. Let's simplify the whole approach.
        break
    
    return blocks, uncovered


def _greedy_simple(v, k, t, seed_subset, rng, deadline):
    """Simpler greedy that's faster per block but less optimal."""
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered and time.time() < deadline:
        # Start with seed
        if seed_subset is not None and len(seed_subset) >= t:
            block = list(seed_subset[:t])
            seed_subset = None  # only use once
        else:
            # Pick first uncovered
            block = list(min(uncovered))
        
        while len(block) < k and time.time() < deadline:
            block_set = set(block)
            best_gain = -1
            best_x = -1
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                if len(block) >= t - 1:
                    for rest in itertools.combinations(block, t - 1):
                        sub = tuple(sorted(rest + (x,)))
                        if sub in uncovered:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x < 0 or best_gain <= 0:
                # No improvement possible; just add random point
                remaining = [x for x in range(v) if x not in block_set]
                if remaining:
                    best_x = rng.choice(remaining)
                else:
                    break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)
    
    return blocks, uncovered


def _remove_redundant(blocks, v, t, deadline):
    """Remove blocks that don't uniquely cover any t-subset."""
    if not blocks:
        return blocks
    
    n = len(blocks)
    # For each block, find t-subsets it uniquely covers
    # A t-subset is uniquely covered by block i if no other block contains it
    # To speed up: for each t-subset, track which blocks cover it (up to 2)
    # But this is expensive. Instead, do a simpler check:
    # For each block, check if removing it leaves all t-subsets covered.
    
    # Simpler: iterate and try removing each block, checking validity
    # This is O(n * C(v,t) * n) which is too slow.
    
    # Alternative: for each block, count how many t-subsets it's the ONLY cover for.
    # If zero, it's redundant.
    
    # Build: for each t-subset, which blocks cover it?
    # Use a dict: t-subset -> count of blocks covering it
    sub_cover_count = {}
    for i, block in enumerate(blocks):
        block_set = set(block)
        for sub in itertools.combinations(block, t):
            sub_cover_count[sub] = sub_cover_count.get(sub, 0) + 1
    
    # A block is redundant if none of its t-subsets have cover_count == 1
    keep = [True] * n
    for i, block in enumerate(blocks):
        unique = 0
        for sub in itertools.combinations(block, t):
            if sub_cover_count.get(sub, 0) == 1:
                unique += 1
                break
        if unique == 0:
            keep[i] = False
    
    new_blocks = [b for b, k in zip(blocks, keep) if k]
    
    # If we removed some, re-check and potentially remove more
    if len(new_blocks) < n:
        # Rebuild cover counts
        sub_cover_count2 = {}
        for i, block in enumerate(new_blocks):
            for sub in itertools.combinations(block, t):
                sub_cover_count2[sub] = sub_cover_count2.get(sub, 0) + 1
        
        # Check if still valid
        all_subs = set(itertools.combinations(range(v), t))
        for sub in all_subs:
            if sub_cover_count2.get(sub, 0) == 0:
                # Invalid, restore
                return blocks
        
        return new_blocks
    
    return blocks


def construct(v, k, t):
    """Construct a covering design C(v,k,t) using randomized greedy with restarts."""
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    rng = random.Random(42)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts with different seeds
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        # For first restart, use deterministic seed
        if restart == 0:
            seed_subset = list(range(t))
        else:
            # Random seed subset
            seed_subset = rng.sample(range(v), min(t, v))
        
        blocks, uncovered = _greedy_simple(v, k, t, seed_subset, rng, deadline)
        
        if not blocks:
            continue
        
        # Check if valid
        if not np.any(uncovered) if isinstance(uncovered, np.ndarray) else True:
            # For _greedy_simple, uncovered is a set
            if len(uncovered) == 0:
                # Remove redundant blocks
                blocks = _remove_redundant(blocks, v, t, deadline)
                
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = blocks
                    if best_count <= 300:
                        break
    
    if best_blocks is None:
        # Fallback: single deterministic greedy
        blocks, uncovered = _greedy_simple(v, k, t, list(range(t)), rng, deadline)
        best_blocks = blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
