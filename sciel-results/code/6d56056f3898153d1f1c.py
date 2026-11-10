# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Uses randomized whole-block greedy with local search (redundant block removal).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Precompute all t-subsets and index them
    all_t = list(itertools.combinations(range(v), t))
    t_index = {ts: i for i, ts in enumerate(all_t)}
    n_t = len(all_t)

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, list of t-subset indices it covers
    coverage_map = []
    for block in all_blocks:
        covered = [t_index[ts] for ts in itertools.combinations(block, t)]
        coverage_map.append(covered)

    # For each t-subset, list of block indices that contain it
    blocks_containing = [[] for _ in range(n_t)]
    for b_idx, covered in enumerate(coverage_map):
        for t_idx in covered:
            blocks_containing[t_idx].append(b_idx)

    # Build a mapping from block tuple to its index for O(1) lookup
    block_to_idx = {block: i for i, block in enumerate(all_blocks)}

    def greedy_construct(rng, deadline):
        """
        Randomized greedy: repeatedly pick the block covering the most uncovered t-subsets.
        Uses randomization in tie-breaking.
        """
        uncovered = bytearray([1] * n_t)  # 1 = uncovered
        blocks = []
        remaining_uncovered = n_t

        while remaining_uncovered > 0:
            if time.time() > deadline:
                break

            best_block = -1
            best_gain = -1
            best_gain_list = None

            # Randomize the order of blocks to break ties
            block_order = list(range(n_blocks))
            rng.shuffle(block_order)

            for b_idx in block_order:
                if time.time() > deadline:
                    break

                # Compute gain: number of uncovered t-subsets in this block
                gain = 0
                for t_idx in coverage_map[b_idx]:
                    if uncovered[t_idx]:
                        gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    best_gain_list = coverage_map[b_idx]
                elif gain == best_gain and gain > 0:
                    # Tie-breaking: random
                    if rng.random() < 0.5:
                        best_block = b_idx
                        best_gain_list = coverage_map[b_idx]

            if best_block == -1 or best_gain <= 0:
                break  # No improvement possible

            blocks.append(all_blocks[best_block])

            # Update uncovered
            for t_idx in best_gain_list:
                if uncovered[t_idx]:
                    uncovered[t_idx] = 0
                    remaining_uncovered -= 1

        # Verify completeness
        if remaining_uncovered > 0:
            return None

        return blocks

    def local_search_remove(blocks, rng, deadline):
        """
        Remove redundant blocks: a block is redundant if all its t-subsets are covered by other blocks.
        Uses the precomputed block_to_idx mapping for O(1) lookups.
        """
        if not blocks:
            return blocks

        # Convert blocks to indices using precomputed mapping
        block_indices = [block_to_idx[b] for b in blocks]

        # Compute coverage count: how many times each t-subset is covered
        coverage_count = [0] * n_t
        for b_idx in block_indices:
            for t_idx in coverage_map[b_idx]:
                coverage_count[t_idx] += 1

        new_block_indices = list(block_indices)
        changed = True

        while changed:
            changed = False
            # Try removing each block (iterate in reverse for safe removal)
            for i in range(len(new_block_indices) - 1, -1, -1):
                if time.time() > deadline:
                    break

                b_idx = new_block_indices[i]
                is_redundant = True
                for t_idx in coverage_map[b_idx]:
                    if coverage_count[t_idx] == 1:
                        is_redundant = False
                        break

                if is_redundant:
                    # Remove this block
                    new_block_indices.pop(i)
                    for t_idx in coverage_map[b_idx]:
                        coverage_count[t_idx] -= 1
                    changed = True

        return [all_blocks[i] for i in new_block_indices]

    # Main loop: try multiple restarts
    best_blocks = None

    restart_count = 0
    while time.time() < deadline - 10.0:
        restart_count += 1
        rng_restart = random.Random(42 + restart_count)

        blocks = greedy_construct(rng_restart, deadline)

        if blocks is not None:
            # Local search
            blocks = local_search_remove(blocks, rng_restart, deadline)

            if best_blocks is None or len(blocks) < len(best_blocks):
                best_blocks = blocks
                if time.time() > deadline - 5.0:
                    break

        # If we've used a lot of time, stop
        if time.time() > deadline - 15.0:
            break

    # Fallback: deterministic greedy if nothing found
    if best_blocks is None:
        uncovered = bytearray([1] * n_t)
        remaining = n_t
        blocks = []
        while remaining > 0:
            if time.time() > deadline:
                break
            # Find any uncovered t-subset
            t_idx = -1
            for i in range(n_t):
                if uncovered[i]:
                    t_idx = i
                    break
            if t_idx == -1:
                break
            # Find the block covering the most uncovered t-subsets that contains t_sub
            best_block = -1
            best_gain = -1
            for b_idx in blocks_containing[t_idx]:
                gain = 0
                for ti in coverage_map[b_idx]:
                    if uncovered[ti]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx

            if best_block != -1:
                blocks.append(all_blocks[best_block])
                for ti in coverage_map[best_block]:
                    if uncovered[ti]:
                        uncovered[ti] = 0
                        remaining -= 1

        if remaining > 0:
            return blocks

        best_blocks = blocks

    return best_blocks
# EVOLVE-BLOCK-END
