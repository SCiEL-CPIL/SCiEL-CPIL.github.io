"""
Covering design C(24,7,3) via bitset-accelerated randomized greedy + redundancy removal.
Strategy:
  1. Precompute an index for every 3-subset of 24 points (2024 total).
  2. Run multiple randomized greedy covering passes with different random seeds.
  3. Use numpy bitsets to speed up gain computation.
  4. After each greedy pass, perform iterative redundancy removal (delete a block if
     every triple it covers is also covered by another block).
  5. Keep the best (fewest blocks) result within the time budget.
"""

import itertools
import time
import random
import numpy as np

# Precompute all 3-subsets and assign each an integer index
_all_triples = list(itertools.combinations(range(24), 3))
_triple_to_idx = {triple: i for i, triple in enumerate(_all_triples)}
_NUM_TRIPLES = len(_all_triples)  # 2024

# Precompute, for each 7-subset, the list of triple indices it covers
# We'll build this lazily or use a fast lookup.


def _block_triple_indices(block):
    """Return list of triple indices covered by a 7-element block (sorted tuple)."""
    block = tuple(sorted(block))
    return [_triple_to_idx[triple] for triple in itertools.combinations(block, 3)]


def _greedy_cover_fast(v, k, t, rng):
    """
    Randomized greedy covering using numpy bitsets for speed.
    Returns a list of blocks (each a sorted list of k ints).
    """
    # uncovered_mask: numpy bool array of length _NUM_TRIPLES
    uncovered_mask = np.ones(_NUM_TRIPLES, dtype=bool)
    blocks = []

    while np.any(uncovered_mask):
        # Pick the first uncovered triple to anchor the block
        unc_idx = np.flatnonzero(uncovered_mask)
        if len(unc_idx) == 0:
            break
        # Randomly pick one uncovered triple for diversity
        anchor_idx = unc_idx[rng.randint(0, len(unc_idx) - 1)]
        anchor = _all_triples[anchor_idx]
        block = list(anchor)

        # Greedily add points to maximize coverage of uncovered triples
        while len(block) < k:
            best_x = -1
            best_gain = -1
            block_set = set(block)
            for x in range(v):
                if x in block_set:
                    continue
                # Compute how many uncovered triples are covered by adding x
                # A triple (a,b,c) is covered if {a,b,c} ⊆ block ∪ {x}
                # i.e., at least t-1=2 elements are in block and the rest in {x}
                gain = 0
                # For each (t-1)-subset of current block, check if (subset + x) is an uncovered triple
                for rest in itertools.combinations(block, t - 1):
                    triple = tuple(sorted(rest + (x,)))
                    if triple in _triple_to_idx:
                        idx = _triple_to_idx[triple]
                        if uncovered_mask[idx]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)

        block.sort()
        blocks.append(block)

        # Update uncovered mask: mark all triples in this block as covered
        for idx in _block_triple_indices(block):
            uncovered_mask[idx] = False

    return blocks


def _remove_redundant_blocks(blocks):
    """
    Iteratively remove blocks that are redundant (every triple they cover is
    covered by at least one other block).
    Returns the reduced list of blocks.
    """
    if len(blocks) <= 1:
        return list(blocks)

    # Build a count of how many blocks cover each triple
    triple_count = np.zeros(_NUM_TRIPLES, dtype=np.int32)
    block_triples = []  # list of lists of triple indices for each block
    for block in blocks:
        tidx = _block_triple_indices(block)
        block_triples.append(tidx)
        for idx in tidx:
            triple_count[idx] += 1

    # Iteratively remove redundant blocks
    removed = [False] * len(blocks)
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            if removed[i]:
                continue
            # Check if every triple in block i is covered by >= 2 blocks
            redundant = True
            for idx in block_triples[i]:
                if triple_count[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                removed[i] = True
                changed = True
                for idx in block_triples[i]:
                    triple_count[idx] -= 1

    return [blocks[i] for i in range(len(blocks)) if not removed[i]]


def construct(v, k, t):
    """
    Construct a covering design: choose as few k-subsets of v points as possible
    so that every t-subset is contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10-second safety margin before 120s limit

    best_blocks = None
    best_len = float('inf')

    # First, run a deterministic greedy to get a baseline
    rng = random.Random(42)
    blocks = _greedy_cover_fast(v, k, t, rng)
    blocks = _remove_redundant_blocks(blocks)
    if len(blocks) < best_len:
        best_len = len(blocks)
        best_blocks = blocks

    # Run multiple randomized greedy passes
    seed = 12345
    while time.time() < deadline:
        rng = random.Random(seed)
        blocks = _greedy_cover_fast(v, k, t, rng)
        blocks = _remove_redundant_blocks(blocks)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        seed += 1

    return best_blocks if best_blocks is not None else []
