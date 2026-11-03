"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import sys

import numpy as np


def _combinations_to_indices(v, t):
    """Map each t-combination (as a tuple of sorted ints) to a dense index 0..C(v,t)-1."""
    # Precompute binomial coefficients
    # C(n, r) for needed values
    def nCr(n, r):
        if r < 0 or r > n:
            return 0
        r = min(r, n - r)
        num = 1
        den = 1
        for i in range(r):
            num *= (n - i)
            den *= (i + 1)
        return num // den

    total = nCr(v, t)
    # Use a dictionary for mapping; for v=19,t=4 this is 3876 entries, fine
    mapping = {}
    idx = 0
    for combo in itertools.combinations(range(v), t):
        mapping[combo] = idx
        idx += 1
    return mapping, total


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Precompute all t-subsets and their indices
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {c: i for i, c in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)

    # Precompute for each block (k-subset) the set of t-subset indices it covers
    # We can't precompute all C(v,k) blocks if too many, but for v=19,k=7: C(19,7)=50388
    # Each block covers C(7,4)=35 t-subsets. Total edges ~ 50388*35 = 1.76M, manageable.
    # But storing a set of 35 ints for each of 50388 blocks is ~ 1.76M ints total, ok.
    
    # Actually, let's use a more memory-efficient approach:
    # For each point, precompute which t-subsets contain it.
    # For each pair of points, etc. But let's just compute on the fly with caching.

    # Better: precompute for each k-subset (as a frozenset) the list of t-subset indices.
    # But 50388 frozensets of size 7, each mapped to a list of 35 ints.
    # Let's use a dict: key = tuple(sorted(block)), value = np.array of t-subset indices.

    # Actually, for the greedy algorithm with local search, we need fast gain computation.
    # Let's precompute: for each subset S of points (represented as a bitmask or frozenset),
    # the set of t-subsets covered by S. But that's 2^19 = 524288 subsets, too many.

    # Alternative: represent uncovered t-subsets as a bitset (numpy array of uint64 or similar).
    # n_tsubs = 3876 for v=19,t=4. 3876 bits = 61 uint64 words. Very manageable.

    # Let's use a Python int as a bitset, or numpy array.
    # For speed, let's precompute for each k-subset the bitmask of t-subsets it covers.

    # Precompute all k-subsets and their coverage masks
    # C(19,7) = 50388, each mask is 3876 bits. Using Python int: fine.
    
    # But building all 50388 masks might take time. Let's see.
    # Each mask: for each of C(7,4)=35 t-subsets in the block, set the bit.
    # 50388 * 35 = 1.76M operations, fast.

    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # Precompute coverage mask for each k-subset
    # mask[i] is a Python int with bit j set if t-subset j is covered by k-subset i
    coverage_masks = [0] * n_ksubs
    for i, block in enumerate(all_ksubs):
        mask = 0
        for tsub in itertools.combinations(block, t):
            mask |= (1 << tsub_index[tsub])
        coverage_masks[i] = mask

    # Greedy construction with local search
    uncovered_mask = (1 << n_tsubs) - 1  # all t-subsets uncovered
    blocks = []
    used_blocks = set()

    # Greedy: repeatedly pick the block that covers the most uncovered t-subsets
    while uncovered_mask and time.time() < deadline:
        # Find the best block
        best_gain = -1
        best_idx = -1
        for i in range(n_ksubs):
            if i in used_blocks:
                continue
            gain = bin(uncovered_mask & coverage_masks[i]).count('1')
            if gain > best_gain:
                best_gain = gain
                best_idx = i
            if time.time() > deadline:
                break
        if best_idx == -1:
            break
        blocks.append(all_ksubs[best_idx])
        used_blocks.add(best_idx)
        uncovered_mask &= ~coverage_masks[best_idx]

    # Local search: try to swap blocks to reduce total count
    # For each pair (i, j) where i is in blocks and j is not, try swapping
    # if it doesn't increase the number of uncovered t-subsets
    # This is complex; let's do a simpler local search:
    # Try removing each block and see if the covering is still valid.
    # If valid, remove it and continue.

    # Check if removing a block keeps the covering valid
    def is_covered_without(block_idx):
        """Check if all t-subsets are still covered without blocks[block_idx]."""
        # Rebuild the union of all other blocks' coverage
        union_mask = 0
        for i, b in enumerate(blocks):
            if i == block_idx:
                continue
            # Find the index of b in all_ksubs
            # This is slow; let's store indices instead
            pass
        # This approach is too slow. Let's store block indices.
        return True  # placeholder

    # Let's refactor to store block indices
    # Actually, let's do a simpler approach: after greedy, try to remove blocks one by one
    # by checking if the remaining blocks still cover everything.

    # Rebuild: store blocks as indices into all_ksubs
    block_indices = [b for b in blocks]
    # Wait, blocks are already the actual block tuples. Let's keep them.

    # For local search, let's try a different approach:
    # Iterate through blocks in reverse order, and for each, check if it's redundant.
    # A block is redundant if all t-subsets it covers are also covered by other blocks.
    
    # Precompute for each t-subset, which blocks cover it
    # This is expensive. Let's do a simpler check:
    # For a candidate block to remove, for each t-subset it covers, check if another block covers it.
    
    # This is O(k * n_tsubs_per_block * n_blocks) per removal check, which is too slow.

    # Given time constraints, let's just return the greedy result.
    # The greedy should be decent.
    
    return [list(b) for b in blocks]
# EVOLVE-BLOCK-END
