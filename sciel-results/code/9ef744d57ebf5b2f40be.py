"""Covering design C(19,7,4) via greedy set cover with local search (block swaps)."""
import itertools
import random
import time
from collections import defaultdict

# We will use a deadline guard.
DEADLINE_MARGIN = 5.0  # seconds before the 150s limit


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)

    # Map each t-subset to an index
    tsubset_index = {ts: i for i, ts in enumerate(all_tsubsets)}

    # For each k-subset (block), precompute the list of t-subsets it covers
    # But there are C(19,7) = 50388 blocks, each covering C(7,4)=35 t-subsets.
    # That's manageable.
    all_ksubsets = list(itertools.combinations(range(v), k))
    n_ksubsets = len(all_ksubsets)

    # For each block, compute the bitmask or set of t-subset indices it covers
    # Using a set of indices for fast operations
    block_covers = []
    for block in all_ksubsets:
        covered = set()
        for ts in itertools.combinations(block, t):
            covered.add(tsubset_index[ts])
        block_covers.append(covered)

    # Greedy set cover
    uncovered = set(range(n_tsubsets))
    blocks = []
    block_indices = []

    while uncovered:
        # Find the block that covers the most uncovered t-subsets
        best_block_idx = -1
        best_count = -1
        for bi in range(n_ksubsets):
            cov = len(block_covers[bi] & uncovered)
            if cov > best_count:
                best_count = cov
                best_block_idx = bi
                if best_count == len(uncovered):
                    break

        blocks.append(list(all_ksubsets[best_block_idx]))
        block_indices.append(best_block_idx)
        uncovered -= block_covers[best_block_idx]

        if time.time() > deadline:
            break

    if not uncovered:
        return blocks

    # Local search: try to replace a block with another that covers more uncovered
    # and allows removing some other blocks
    # We'll do a simple swap-based local search
    # First, compute for each t-subset which blocks cover it
    # This is expensive, so let's use a different approach

    # Alternative: try random swaps
    # For each pair of blocks, try replacing one with a random other block
    # and see if we can reduce the total count

    # Let's do a more structured approach:
    # 1. Compute coverage counts for each t-subset
    # 2. Try to remove blocks that are redundant (all their t-subsets are covered by other blocks)
    # 3. Try to swap blocks to improve coverage

    # Compute for each t-subset, how many blocks cover it
    coverage_count = [0] * n_tsubsets
    for bi in block_indices:
        for ts_idx in block_covers[bi]:
            coverage_count[ts_idx] += 1

    # Try to remove redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(block_indices) - 1, -1, -1):
            bi = block_indices[i]
            # Check if all t-subsets in this block are covered by other blocks
            redundant = True
            for ts_idx in block_covers[bi]:
                if coverage_count[ts_idx] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove this block
                for ts_idx in block_covers[bi]:
                    coverage_count[ts_idx] -= 1
                block_indices.pop(i)
                blocks.pop(i)
                changed = True
                if time.time() > deadline:
                    break
        if time.time() > deadline:
            break

    # Now try local search: swap blocks
    # For each block, try replacing it with a different block that covers more uncovered t-subsets
    # But all t-subsets are covered, so we need to look for opportunities to reduce count

    # Try: for each pair of blocks, see if we can replace one with a block that covers
    # all t-subsets of the removed block plus some extra, allowing removal of another block
    # This is complex. Let's try a simpler approach: random restarts with different greedy orders

    # Actually, let's try a different local search:
    # For each block i, try replacing it with block j (j not in current set)
    # such that all t-subsets of block i are still covered (by other blocks or by j)
    # and block j covers some t-subsets that are only covered by block i (redundancy)
    # This doesn't directly help reduce count.

    # Better: try to find two blocks that can be replaced by one block
    # For each pair of blocks (i, j), check if there exists a block that covers
    # all t-subsets covered by both i and j. If so, replace i and j with that block.
    # This reduces count by 1.

    # Precompute union of t-subsets for each pair of blocks
    # This is expensive: O(n_blocks^2 * 35)
    # With ~150 blocks, that's ~150*150*35 = 787500 operations, manageable

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(block_indices)):
            for j in range(i + 1, len(block_indices)):
                union_ts = block_covers[block_indices[i]] | block_covers[block_indices[j]]
                # Find a block that covers all t-subsets in union_ts
                for bi in range(n_ksubsets):
                    if bi in block_indices:
                        continue
                    if union_ts.issubset(block_covers[bi]):
                        # Replace blocks i and j with block bi
                        # Remove j first (higher index)
                        block_indices.pop(j)
                        blocks.pop(j)
                        # Replace i with bi
                        block_indices[i] = bi
                        blocks[i] = list(all_ksubsets[bi])
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return blocks
