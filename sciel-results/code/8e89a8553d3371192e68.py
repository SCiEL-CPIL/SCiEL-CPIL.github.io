"""Covering design C(24,7,3): fast randomized greedy with local search reduction."""
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 142.0  # safety margin before 150s kill

    # Precompute all t-subsets for faster membership checks
    all_tsubs = list(itertools.combinations(range(v), t))
    all_tsub_set = set(all_tsubs)

    # Greedy covering with randomized restarts and local search
    best_blocks = None
    best_count = float('inf')

    rng = random.Random(42)

    for restart in range(200):  # cap restarts
        if time.time() > deadline:
            break

        # Randomized greedy: pick a random uncovered t-subset to start each block
        uncovered = set(all_tsub_set)
        blocks = []

        while uncovered:
            if time.time() > deadline:
                break

            # Pick a random uncovered triple to start
            start_triple = rng.choice(list(uncovered))
            block = list(start_triple)
            block_set = set(block)

            while len(block) < k:
                if time.time() > deadline:
                    break

                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    # Count how many uncovered t-subsets this point would cover
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x

                if best_x == -1:
                    break
                block.append(best_x)
                block_set.add(best_x)

            if len(block) < k:
                break

            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))

        if not uncovered and len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)

        # If we've found a good solution, do local search to improve it
        if best_blocks is not None and len(best_blocks) <= 80:
            improved = _local_search(best_blocks, v, k, t, all_tsub_set, rng, deadline)
            if improved is not None and len(improved) < best_count:
                best_blocks = improved
                best_count = len(best_blocks)

    # Fallback: deterministic greedy if nothing found
    if best_blocks is None:
        uncovered = set(all_tsub_set)
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            block_set = set(block)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    break
                block.append(best_x)
                block_set.add(best_x)
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        best_blocks = blocks

    return best_blocks


def _local_search(blocks, v, k, t, all_tsub_set, rng, deadline):
    """Try to reduce the number of blocks by removing redundant ones and replacing blocks."""
    blocks = [list(b) for b in blocks]
    n = len(blocks)

    # Build a coverage map: for each t-subset, which blocks cover it
    # Use bitmasks for efficiency
    def _build_coverage_map(blks):
        cov = defaultdict(list)
        for i, blk in enumerate(blks):
            for tsub in itertools.combinations(blk, t):
                cov[tsub].append(i)
        return cov

    # Pass 1: Remove redundant blocks (all their t-subsets covered by others)
    for _pass in range(5):  # Multiple passes to catch cascading redundancies
        if time.time() > deadline:
            return blocks
        if not blocks:
            break

        cov = _build_coverage_map(blocks)
        n = len(blocks)
        removed = False

        for i in range(n):
            if time.time() > deadline:
                return blocks

            # Check if block i is redundant: all its t-subsets are covered by at least one other block
            redundant = True
            for tsub in itertools.combinations(blocks[i], t):
                # Need at least one block other than i covering this tsub
                covering = [j for j in cov[tsub] if j != i]
                if not covering:
                    redundant = False
                    break

            if redundant:
                # Remove block i
                blocks.pop(i)
                n -= 1
                removed = True
                # Rebuild coverage map after removal
                cov = _build_coverage_map(blocks)

        if not removed:
            break

    # Pass 2: Try to replace each block with a better one (greedy reconstruction)
    if time.time() > deadline:
        return blocks

    for _pass in range(3):
        if time.time() > deadline:
            return blocks

        cov = _build_coverage_map(blocks)
        n = len(blocks)
        improved = False

        for i in range(n):
            if time.time() > deadline:
                return blocks

            # Try replacing block i: remove it, then greedily rebuild
            # to cover what it was uniquely covering
            old_block = blocks[i]
            blocks.pop(i)

            # Find t-subsets that are now uncovered (were only covered by old_block)
            newly_uncovered = set()
            for tsub in itertools.combinations(old_block, t):
                covering = [j for j in cov[tsub] if j != i]
                if not covering:
                    newly_uncovered.add(tsub)

            if not newly_uncovered:
                # Old block was redundant, keep it removed
                improved = True
                continue

            # Greedily build a new block to cover newly_uncovered
            # Start with a t-subset from newly_uncovered
            start_triple = list(next(iter(newly_uncovered)))
            new_block = list(start_triple)
            new_block_set = set(new_block)
            uncovered = set(newly_uncovered)

            while len(new_block) < k and uncovered:
                if time.time() > deadline:
                    break

                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in new_block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(new_block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x

                if best_x == -1:
                    break
                new_block.append(best_x)
                new_block_set.add(best_x)

            if len(new_block) < k:
                # Couldn't build a valid block, restore old
                blocks.insert(i, old_block)
                continue

            new_block.sort()

            # Check if new block covers all newly_uncovered
            new_tsubs = set(itertools.combinations(new_block, t))
            if newly_uncovered.issubset(new_tsubs):
                blocks.insert(i, new_block)
                improved = True
            else:
                # New block doesn't cover everything, restore old
                blocks.insert(i, old_block)

        if not improved:
            break

    return blocks
