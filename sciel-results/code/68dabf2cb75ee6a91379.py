"""Probabilistic method with alteration for C(19,7,4): random pool + MIS pruning."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

import numpy as np


def _build_subset_index(v, t):
    """Precompute all t-subsets and their indices."""
    all_subs = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    return all_subs, sub_to_idx, n_subsets


def _random_pool(v, k, t, n_blocks, seed=42):
    """Generate a random pool of k-blocks."""
    rng = random.Random(seed)
    blocks = set()
    while len(blocks) < n_blocks:
        block = tuple(sorted(rng.sample(range(v), k)))
        blocks.add(block)
    return list(blocks)


def _maximal_independent_set_prune(blocks, v, k, t, deadline, max_rounds=5):
    """
    Prune blocks using a maximal independent set heuristic on the dual hypergraph.
    
    Two blocks are 'adjacent' if they share at least one t-subset.
    We want a maximal set of non-adjacent blocks (no two share a t-subset)
    that still covers all t-subsets.
    
    Actually, we want to REMOVE blocks whose t-subsets are all covered by remaining blocks.
    Use greedy removal: repeatedly find a block whose removal doesn't uncover any t-subset.
    """
    all_subs, sub_to_idx, n_subsets = _build_subset_index(v, t)
    
    # For each block, compute the set of t-subset indices it covers
    block_subs = []
    for b in blocks:
        subs = set()
        for sub in itertools.combinations(b, t):
            idx = sub_to_idx.get(sub, -1)
            if idx >= 0:
                subs.add(idx)
        block_subs.append(subs)
    
    # Coverage count: how many blocks cover each t-subset
    coverage_count = np.zeros(n_subsets, dtype=np.int32)
    for subs in block_subs:
        for idx in subs:
            coverage_count[idx] += 1
    
    # Greedy removal: remove blocks that are 'redundant' (all their subsets are covered by others)
    # Process in order of lowest contribution (fewest unique subsets)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Find a redundant block: one where removing it doesn't uncover any t-subset
        # A block is redundant if for every t-subset it covers, coverage_count > 1
        for i in range(len(blocks)):
            if time.time() > deadline:
                return blocks
            is_redundant = True
            for idx in block_subs[i]:
                if coverage_count[idx] <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                # Remove this block
                for idx in block_subs[i]:
                    coverage_count[idx] -= 1
                blocks.pop(i)
                block_subs.pop(i)
                changed = True
                break
    
    return blocks


def _probabilistic_cover(v, k, t, deadline):
    """
    Probabilistic method with alteration:
    1. Generate a large random pool of blocks.
    2. Prune using MIS heuristic to remove redundant blocks.
    3. If not all t-subsets are covered, add greedy blocks for uncovered ones.
    4. Prune again.
    """
    all_subs, sub_to_idx, n_subsets = _build_subset_index(v, t)
    
    # Step 1: Generate random pool. 
    # Expected number of blocks needed: each block covers C(k,t) t-subsets out of C(v,t).
    # Probability a specific t-subset is in a random block: C(v-t, k-t)/C(v,k)
    # For v=19, k=7, t=4: C(15,3)/C(19,7) = 455/50388 ≈ 0.00903
    # To cover all ~3876 t-subsets with high probability, need ~3876 * ln(3876)/0.00903 ≈ 2.5M blocks
    # That's too many. Use a smaller pool and supplement with greedy.
    
    # Use a moderate pool size
    pool_size = 5000
    blocks = _random_pool(v, k, t, pool_size, seed=12345)
    
    # Step 2: Prune
    blocks = _maximal_independent_set_prune(blocks, v, k, t, deadline)
    
    # Step 3: Check coverage and add greedy blocks for uncovered t-subsets
    covered = np.zeros(n_subsets, dtype=bool)
    for b in blocks:
        for sub in itertools.combinations(b, t):
            idx = sub_to_idx.get(sub, -1)
            if idx >= 0:
                covered[idx] = True
    
    if not covered.all():
        # Greedy supplement: add blocks to cover remaining t-subsets
        uncovered_indices = np.where(~covered)[0]
        while len(uncovered_indices) > 0 and time.time() < deadline:
            # Pick first uncovered t-subset as anchor
            anchor_idx = uncovered_indices[0]
            anchor = list(all_subs[anchor_idx])
            
            # Build a block around the anchor greedily
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
            uncovered_indices = np.where(~covered)[0]
    
    # Step 4: Prune again
    blocks = _maximal_independent_set_prune(blocks, v, k, t, deadline)
    
    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Use probabilistic method with alteration
    blocks = _probabilistic_cover(v, k, t, deadline)
    
    return blocks
# EVOLVE-BLOCK-END
