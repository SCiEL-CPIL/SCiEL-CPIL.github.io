"""Covering design C(19,7,4) solver with randomized greedy + local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys


def construct(v, k, t):
    # Deadline guard: evaluator kills after 150s wall-clock.
    deadline = time.time() + 130  # safety margin

    # Precompute all t-subsets as frozensets for fast lookup
    all_tsubsets = list(itertools.combinations(range(v), t))
    total_tsubsets = len(all_tsubsets)

    def blocks_cover(block):
        """Return the set of t-subsets covered by a given block (list of k points)."""
        return set(itertools.combinations(block, t))

    def uncovered_count(blocks):
        """Count how many t-subsets are NOT covered by the given blocks."""
        covered = set()
        for b in blocks:
            covered.update(itertools.combinations(b, t))
        return total_tsubsets - len(covered)

    def is_valid(blocks):
        """Check if every t-subset is covered."""
        covered = set()
        for b in blocks:
            covered.update(itertools.combinations(b, t))
        return len(covered) == total_tsubsets

    def greedy_init(rng):
        """Randomized greedy covering."""
        uncovered = set(all_tsubsets)
        blocks = []
        while uncovered:
            # Pick a random uncovered t-subset to seed the block
            seed = rng.choice(list(uncovered))
            block = list(seed)
            block_set = set(block)
            # Add points one at a time, choosing the one that maximizes new coverage
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    # Count how many uncovered t-subsets would be newly covered
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        cand = tuple(sorted(rest + (x,)))
                        if cand in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
                block_set.add(best_x)
            block.sort()
            blocks.append(block)
            # Remove all t-subsets covered by this block
            for ts in itertools.combinations(block, t):
                uncovered.discard(ts)
        return blocks

    def local_search(blocks, rng):
        """Local search: try removing redundant blocks and swapping points."""
        blocks = [list(b) for b in blocks]
        improved = True
        max_iters = 500
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1

            # Phase 1: Remove redundant blocks
            for i in range(len(blocks)):
                remaining = [blocks[j] for j in range(len(blocks)) if j != i]
                if is_valid(remaining):
                    del blocks[i]
                    improved = True
                    break

            if improved:
                continue

            # Phase 2: Try to swap a point in a block to improve coverage
            # We look for a block where replacing one point with another
            # doesn't lose coverage but might allow removing another block later.
            # Simpler: try replacing a point in block i with a point from block j
            # to see if we can merge or reduce.
            # Alternative: try dropping a point from a block and re-adding optimally.
            # Let's try: for each block, for each point in it, try replacing with
            # a different point and check if the cover is still valid.
            for i in range(len(blocks)):
                if time.time() > deadline - 10:
                    return blocks
                for pi in range(len(blocks[i])):
                    old_point = blocks[i][pi]
                    for new_point in range(v):
                        if new_point == old_point or new_point in blocks[i]:
                            continue
                        # Try the swap
                        blocks[i][pi] = new_point
                        if is_valid(blocks):
                            improved = True
                            break
                        else:
                            blocks[i][pi] = old_point
                    if improved:
                        break
                if improved:
                    break

        return blocks

    # Main: multi-start local search
    best_blocks = None
    best_len = float('inf')

    # First, get a baseline greedy solution (deterministic-ish)
    rng = random.Random(42)
    blocks = greedy_init(rng)
    best_blocks = blocks
    best_len = len(blocks)

    # Try multiple random restarts
    attempt = 0
    while time.time() < deadline - 15:
        attempt += 1
        rng = random.Random(42 + attempt * 7919)
        blocks = greedy_init(rng)
        if len(blocks) < best_len:
            best_blocks = [list(b) for b in blocks]
            best_len = len(blocks)

        # Local search on the best so far
        if len(blocks) <= best_len:
            ls_blocks = local_search(blocks, rng)
            if len(ls_blocks) < best_len:
                best_blocks = ls_blocks
                best_len = len(best_blocks)

        # Periodically also do local search on the current best
        if attempt % 5 == 0:
            ls_blocks = local_search(best_blocks, rng)
            if len(ls_blocks) < best_len:
                best_blocks = ls_blocks
                best_len = len(best_blocks)

    # Final polish: try to remove more redundant blocks
    for _ in range(3):
        for i in range(len(best_blocks)):
            remaining = [best_blocks[j] for j in range(len(best_blocks)) if j != i]
            if is_valid(remaining):
                best_blocks = remaining
                break

    return best_blocks
# EVOLVE-BLOCK-END
