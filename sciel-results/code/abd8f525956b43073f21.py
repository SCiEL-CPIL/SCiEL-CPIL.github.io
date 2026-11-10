"""Covering design C(19,7,4) – improved greedy with efficient local search."""
import itertools
import time
import random
import math
from collections import defaultdict, Counter

_DEADLINE_S = 145.0
_START_TIME = time.time()


def _time_left():
    return max(0.0, _DEADLINE_S - (time.time() - _START_TIME))


def _greedy_construct(v, k, t):
    """
    Greedy construction with point-frequency bias to avoid overusing popular points.
    """
    all_tsubs = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsubs)
    blocks = []
    point_freq = [0] * v

    while uncovered:
        if _time_left() < 2.0:
            break

        # Pick the uncovered t-subset with the rarest points (min sum of frequencies)
        best_target = min(uncovered, key=lambda tsub: sum(point_freq[p] for p in tsub))
        target_set = set(best_target)
        remaining_points = [p for p in range(v) if p not in target_set]

        best_block = None
        best_gain = -1
        # Sample a limited number of candidate blocks for speed
        n_candidates = 200
        for _ in range(n_candidates):
            extra = random.sample(remaining_points, k - t)
            block = tuple(sorted(list(best_target) + list(extra)))
            gain = sum(1 for tsub in itertools.combinations(block, t) if tsub in uncovered)
            if gain > best_gain:
                best_gain = gain
                best_block = block

        if best_block is None:
            break

        blocks.append(list(best_block))
        for tsub in itertools.combinations(best_block, t):
            uncovered.discard(tsub)
        for p in best_block:
            point_freq[p] += 1

    # Fill any remaining gaps (should be few if we sampled well)
    if uncovered and _time_left() > 2.0:
        for tsub in list(uncovered):
            remaining = [p for p in range(v) if p not in set(tsub)]
            extra = random.sample(remaining, k - t)
            block = tuple(sorted(list(tsub) + list(extra)))
            blocks.append(list(block))
            for t2 in itertools.combinations(block, t):
                uncovered.discard(t2)

    return blocks


def _remove_redundant_blocks(blocks, v, k, t):
    """Remove blocks that don't cover any unique t-subset."""
    if not blocks:
        return blocks

    tsub_to_blocks = defaultdict(set)
    for i, block in enumerate(blocks):
        for tsub in itertools.combinations(block, t):
            tsub_to_blocks[tsub].add(i)

    changed = True
    while changed:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            redundant = True
            for tsub in itertools.combinations(blocks[i], t):
                if len(tsub_to_blocks[tsub]) == 1:
                    redundant = False
                    break
            if redundant:
                for tsub in itertools.combinations(blocks[i], t):
                    tsub_to_blocks[tsub].discard(i)
                blocks.pop(i)
                # Re-index
                for tsub, bset in list(tsub_to_blocks.items()):
                    new_bset = set()
                    for j in bset:
                        if j < i:
                            new_bset.add(j)
                        elif j > i:
                            new_bset.add(j - 1)
                    tsub_to_blocks[tsub] = new_bset
                changed = True
                break

    return blocks


def _local_search(blocks, v, k, t, time_budget=100.0):
    """
    Local search: try to replace each block with one that covers its unique t-subsets
    while allowing more redundant blocks to be removed.
    """
    if not blocks:
        return blocks

    # Build tsub -> block indices
    tsub_to_blocks = defaultdict(set)
    for i, block in enumerate(blocks):
        for tsub in itertools.combinations(block, t):
            tsub_to_blocks[tsub].add(i)

    best_blocks = list(blocks)
    best_len = len(blocks)
    start_time = time.time()

    # Iterate multiple passes
    while time.time() - start_time < time_budget:
        progress = False

        for i in range(len(blocks)):
            if _time_left() < 5.0:
                return best_blocks

            # Find unique t-subsets for block i
            unique_tsubs = [tsub for tsub in itertools.combinations(blocks[i], t)
                           if len(tsub_to_blocks[tsub]) == 1]

            if not unique_tsubs:
                # Redundant block, try to remove
                for tsub in itertools.combinations(blocks[i], t):
                    tsub_to_blocks[tsub].discard(i)
                blocks.pop(i)
                # Re-index
                for tsub, bset in list(tsub_to_blocks.items()):
                    new_bset = set()
                    for j in bset:
                        if j < i:
                            new_bset.add(j)
                        elif j > i:
                            new_bset.add(j - 1)
                    tsub_to_blocks[tsub] = new_bset
                progress = True
                break  # Restart pass after modification

            # Collect must-include points from unique t-subsets
            must_include = set()
            for tsub in unique_tsubs:
                must_include.update(tsub)

            if len(must_include) > k:
                continue

            remaining_points = [p for p in range(v) if p not in must_include]
            need_more = k - len(must_include)

            # Try to find a replacement block
            for _ in range(50):
                if need_more == 0:
                    new_block = tuple(sorted(must_include))
                else:
                    extra = random.sample(remaining_points, need_more)
                    new_block = tuple(sorted(list(must_include) + list(extra)))

                # Check if new block covers all unique t-subsets
                valid = all(set(tsub).issubset(set(new_block)) for tsub in unique_tsubs)
                if not valid:
                    continue

                # Check if this replacement allows removing other blocks
                old_block = blocks[i]
                # Count how many blocks become redundant if we replace
                new_tsub_count = defaultdict(int)
                for j, b in enumerate(blocks):
                    if j == i:
                        continue
                    for tsub in itertools.combinations(b, t):
                        new_tsub_count[tsub] += 1

                # Check which blocks (other than i) become redundant
                removable = 0
                for j, b in enumerate(blocks):
                    if j == i:
                        continue
                    is_removable = True
                    for tsub in itertools.combinations(b, t):
                        if tsub_to_blocks[tsub].issubset({i, j}) and len(tsub_to_blocks[tsub]) == 2:
                            # This tsub is only covered by i and j; if we remove i, j must cover it
                            # But j still covers it, so it's fine
                            pass
                        elif j not in tsub_to_blocks[tsub]:
                            is_removable = False
                            break
                        elif len(tsub_to_blocks[tsub]) == 1 and i in tsub_to_blocks[tsub]:
                            is_removable = False
                            break
                    if is_removable:
                        removable += 1

                # Accept if we can remove at least one other block
                if removable >= 1:
                    # Perform the replacement
                    blocks[i] = list(new_block)
                    for tsub in itertools.combinations(old_block, t):
                        tsub_to_blocks[tsub].discard(i)
                    for tsub in itertools.combinations(new_block, t):
                        tsub_to_blocks[tsub].add(i)

                    # Remove redundant blocks
                    changed = True
                    while changed:
                        changed = False
                        for j in range(len(blocks) - 1, -1, -1):
                            redundant = True
                            for tsub in itertools.combinations(blocks[j], t):
                                if len(tsub_to_blocks[tsub]) == 1:
                                    redundant = False
                                    break
                            if redundant:
                                for tsub in itertools.combinations(blocks[j], t):
                                    tsub_to_blocks[tsub].discard(j)
                                blocks.pop(j)
                                for tsub, bset in list(tsub_to_blocks.items()):
                                    new_bset = set()
                                    for m in bset:
                                        if m < j:
                                            new_bset.add(m)
                                        elif m > j:
                                            new_bset.add(m - 1)
                                    tsub_to_blocks[tsub] = new_bset
                                changed = True
                                break

                    progress = True
                    break

        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)

        if not progress:
            # Try some random perturbations
            for _ in range(10):
                if _time_left() < 5.0:
                    return best_blocks

                # Randomly swap two blocks
                if len(blocks) < 2:
                    break
                i, j = random.sample(range(len(blocks)), 2)
                blocks[i], blocks[j] = blocks[j], blocks[i]
                # Rebuild tsub_to_blocks
                tsub_to_blocks = defaultdict(set)
                for idx, block in enumerate(blocks):
                    for tsub in itertools.combinations(block, t):
                        tsub_to_blocks[tsub].add(idx)

    return best_blocks


def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Returns a list of blocks, each a list of k distinct integers in 0..v-1.
    """
    global _START_TIME
    _START_TIME = time.time()

    # Phase 1: Greedy construction
    blocks = _greedy_construct(v, k, t)

    if _time_left() < 5.0:
        return blocks

    # Phase 2: Remove redundant blocks
    blocks = _remove_redundant_blocks(blocks, v, k, t)

    if _time_left() < 15.0:
        return blocks

    # Phase 3: Local search refinement
    time_budget = min(100.0, _time_left() - 10.0)
    if time_budget > 5.0:
        blocks = _local_search(blocks, v, k, t, time_budget)

    # Phase 4: Final cleanup
    if _time_left() > 5.0:
        blocks = _remove_redundant_blocks(blocks, v, k, t)

    return blocks
