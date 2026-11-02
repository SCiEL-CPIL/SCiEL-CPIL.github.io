"""Covering design C(v,k,t) via greedy with time limit and local improvement."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Precompute all t-subsets as frozensets for fast membership
    all_t_subsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    total_t_subsets = len(all_t_subsets)

    # Map each t-subset to an index for fast tracking
    t_subset_index = {s: i for i, s in enumerate(all_t_subsets)}

    # Track uncovered t-subsets: 1 means uncovered
    uncovered = bytearray([1] * total_t_subsets)
    uncovered_count = total_t_subsets

    blocks = []

    def time_left():
        return deadline - time.time()

    def cover_block_t_subsets(block_set):
        """Return list of indices of t-subsets covered by block_set."""
        covered = []
        for c in itertools.combinations(block_set, t):
            idx = t_subset_index[frozenset(c)]
            if uncovered[idx]:
                covered.append(idx)
        return covered

    def greedy_one_block():
        """Build one block greedily."""
        # Pick the first uncovered t-subset to start
        first_uncovered = -1
        for i in range(total_t_subsets):
            if uncovered[i]:
                first_uncovered = i
                break
        if first_uncovered < 0:
            return None

        start_set = set(all_t_subsets[first_uncovered])
        block = list(start_set)
        block_set = set(block)

        while len(block) < k and time_left() > 1.0:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    s = frozenset(rest) | {x}
                    idx = t_subset_index[s]
                    if uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None or best_gain <= 0:
                # Just add a random point if no gain
                remaining = [x for x in range(v) if x not in block_set]
                if remaining:
                    best_x = random.choice(remaining)
                else:
                    break
            block.append(best_x)
            block_set.add(best_x)

        block.sort()
        return block

    # Phase 1: Greedy construction
    while uncovered_count > 0 and time_left() > 5.0:
        block = greedy_one_block()
        if block is None:
            break
        blocks.append(block)
        block_set = set(block)
        # Update uncovered
        for c in itertools.combinations(block_set, t):
            idx = t_subset_index[frozenset(c)]
            if uncovered[idx]:
                uncovered[idx] = 0
                uncovered_count -= 1

    if uncovered_count == 0:
        return blocks

    # Phase 2: Try to improve by removing redundant blocks
    # If time permits, do a few rounds of block removal
    if time_left() > 10.0:
        for _ in range(3):  # limited rounds
            if time_left() < 5.0:
                break
            # Rebuild coverage map
            covered_by = [set() for _ in range(total_t_subsets)]
            for bi, block in enumerate(blocks):
                block_set = set(block)
                for c in itertools.combinations(block_set, t):
                    idx = t_subset_index[frozenset(c)]
                    covered_by[idx].add(bi)

            # Try removing each block
            removed_any = False
            for bi in range(len(blocks) - 1, -1, -1):
                if time_left() < 5.0:
                    break
                # Check if block bi is the only cover for any t-subset
                block_set = set(blocks[bi])
                essential = False
                for c in itertools.combinations(block_set, t):
                    idx = t_subset_index[frozenset(c)]
                    if len(covered_by[idx]) == 1:
                        essential = True
                        break
                if not essential:
                    # Remove block bi
                    del blocks[bi]
                    removed_any = True
                    # Update covered_by
                    for c in itertools.combinations(block_set, t):
                        idx = t_subset_index[frozenset(c)]
                        covered_by[idx].discard(bi)
                    # Adjust indices for blocks after bi
                    for idx in range(total_t_subsets):
                        for j in list(covered_by[idx]):
                            if j > bi:
                                covered_by[idx].discard(j)
                                covered_by[idx].add(j - 1)
                    break  # restart the loop after removal

            if not removed_any:
                break

    # Final validation: ensure all t-subsets are covered
    if time_left() > 1.0:
        final_uncovered = bytearray([1] * total_t_subsets)
        for block in blocks:
            block_set = set(block)
            for c in itertools.combinations(block_set, t):
                idx = t_subset_index[frozenset(c)]
                final_uncovered[idx] = 0

        # If something is uncovered (shouldn't happen), add blocks to cover it
        for i in range(total_t_subsets):
            if final_uncovered[i]:
                # Add a block covering this t-subset
                ts = set(all_t_subsets[i])
                block = list(ts)
                while len(block) < k:
                    x = random.randint(0, v - 1)
                    if x not in block:
                        block.append(x)
                block.sort()
                blocks.append(block)

    return blocks
# EVOLVE-BLOCK-END
