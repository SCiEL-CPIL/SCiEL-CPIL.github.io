"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin before 120s limit

    # Precompute all t-subsets and index them
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsubset_index = {ts: i for i, ts in enumerate(all_tsubsets)}

    # For each point, precompute which t-subsets it participates in
    # point_to_tsubsets[x] = list of indices of t-subsets containing x
    point_to_tsubsets = defaultdict(set)
    for i, ts in enumerate(all_tsubsets):
        for x in ts:
            point_to_tsubsets[x].add(i)

    def greedy_construct():
        """Greedy covering with efficient gain computation."""
        uncovered = set(range(n_tsubsets))
        blocks = []

        while uncovered:
            if time.time() > deadline - 5.0:
                break

            # Start block with the first uncovered t-subset
            first_ts = min(uncovered)
            block = list(all_tsubsets[first_ts])

            while len(block) < k:
                if time.time() > deadline - 5.0:
                    break

                best_point = -1
                best_gain = -1

                for x in range(v):
                    if x in block:
                        continue
                    # Gain: number of currently uncovered t-subsets that would be covered
                    # A t-subset is covered by adding x to block iff it contains x and
                    # all other t-1 elements are already in block
                    gain = 0
                    # For each (t-1)-subset of block, check if combined with x is uncovered
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in tsubset_index:
                            idx = tsubset_index[candidate]
                            if idx in uncovered:
                                gain += 1

                    if gain > best_gain:
                        best_gain = gain
                        best_point = x

                if best_point == -1:
                    break
                block.append(best_point)

            block.sort()
            # Remove covered t-subsets
            for ts in itertools.combinations(block, t):
                idx = tsubset_index[ts]
                uncovered.discard(idx)

            blocks.append(block)

        return blocks

    def remove_redundant_blocks(blocks):
        """Remove blocks that are redundant (their t-subsets are all covered by other blocks)."""
        if not blocks:
            return blocks

        # Build coverage: for each t-subset index, which blocks cover it
        # For efficiency, track count per t-subset
        coverage_count = [0] * n_tsubsets
        for block in blocks:
            for ts in itertools.combinations(block, t):
                coverage_count[tsubset_index[ts]] += 1

        # A block is redundant if for every t-subset it covers, there's another block covering it
        # i.e., coverage_count[ts] >= 2 for all ts in the block
        kept = []
        for block in blocks:
            redundant = True
            for ts in itertools.combinations(block, t):
                if coverage_count[tsubset_index[ts]] <= 1:
                    redundant = False
                    break
            if not redundant:
                kept.append(block)
            else:
                # This block is redundant; decrement counts
                for ts in itertools.combinations(block, t):
                    coverage_count[tsubset_index[ts]] -= 1

        return kept

    def local_search(blocks, max_iter=50):
        """Try to replace a block with a better one that covers more uncovered subsets."""
        if not blocks:
            return blocks

        # Track uncovered after removing each block? That's expensive.
        # Instead: for each block, try to find a replacement that covers at least as many
        # unique t-subsets.
        # Simpler: just try removing each block and see if coverage is maintained.
        # If yes, remove it.
        changed = True
        while changed and time.time() < deadline - 10.0:
            changed = False
            new_blocks = list(blocks)
            for i in range(len(new_blocks)):
                if time.time() > deadline - 10.0:
                    break
                # Try removing block i
                candidate = new_blocks[:i] + new_blocks[i+1:]
                # Check if all t-subsets are still covered
                # Build coverage set from candidate
                covered = set()
                for block in candidate:
                    for ts in itertools.combinations(block, t):
                        covered.add(tsubset_index[ts])
                if len(covered) == n_tsubsets:
                    new_blocks = candidate
                    changed = True
                    break
            blocks = new_blocks

        return blocks

    # Run greedy construction
    blocks = greedy_construct()

    # Remove redundant blocks
    if time.time() < deadline - 20.0:
        blocks = remove_redundant_blocks(blocks)

    # Local search: try to remove more blocks
    if time.time() < deadline - 10.0:
        blocks = local_search(blocks)

    # Verify coverage (safety check)
    covered = set()
    for block in blocks:
        for ts in itertools.combinations(block, t):
            covered.add(tsubset_index[ts])

    # If we didn't cover everything, fall back to full greedy
    if len(covered) < n_tsubsets:
        blocks = greedy_construct()

    return blocks
# EVOLVE-BLOCK-END
