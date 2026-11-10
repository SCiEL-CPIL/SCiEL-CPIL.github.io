"""Simulated annealing covering design C(19,7,4) solver.

Strategy:
1. Fast greedy construction with incremental coverage counting.
2. Redundant block removal.
3. Simulated annealing: state is the set of blocks; moves include
   removing a block (if coverage stays complete), swapping a block for
   another that covers its unique subsets, and occasionally adding a block.
   Accept worse solutions probabilistically to escape local minima.
4. Deadline guard to return best result before timeout.
"""

import itertools
import time
import random
import math
from collections import defaultdict


def _build_sub_index(v, t):
    """Map each t-subset to a unique index; return (list_of_subsets, sub_index, point_sub_map)."""
    subs = list(itertools.combinations(range(v), t))
    sub_index = {s: i for i, s in enumerate(subs)}
    point_sub_map = {}
    for idx, sub in enumerate(subs):
        for x in sub:
            rest = tuple(p for p in sub if p != x)
            point_sub_map[(x, rest)] = idx
    return subs, sub_index, point_sub_map


def _greedy_construct(v, k, t, deadline):
    """Fast greedy covering using bitset gain computation."""
    subs, sub_index, point_sub_map = _build_sub_index(v, t)
    n_subs = len(subs)
    uncovered = set(range(n_subs))
    blocks = []

    while uncovered and time.time() < deadline:
        min_idx = min(uncovered)
        min_sub = subs[min_idx]
        block = list(min_sub)
        block_set = set(block)

        while len(block) < k and time.time() < deadline:
            best_x = -1
            best_gain = -1

            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    rest_tuple = tuple(rest)
                    idx = point_sub_map.get((x, rest_tuple), -1)
                    if idx >= 0 and idx in uncovered:
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

        for sub in itertools.combinations(block, t):
            idx = sub_index.get(sub, -1)
            if idx >= 0:
                uncovered.discard(idx)

    return blocks


def _remove_redundant_blocks(blocks, v, k, t, deadline):
    """Remove blocks that don't cover any unique t-subset."""
    if not blocks:
        return blocks

    sub_count = defaultdict(int)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            sub_count[sub] += 1

    kept = []
    for block in blocks:
        redundant = True
        for sub in itertools.combinations(block, t):
            if sub_count[sub] == 1:
                redundant = False
                break
        if not redundant:
            kept.append(block)

    return kept


def _compute_sub_indices(block, sub_index, t):
    """Return list of sub indices covered by a block."""
    indices = []
    for sub in itertools.combinations(block, t):
        idx = sub_index.get(sub, -1)
        if idx >= 0:
            indices.append(idx)
    return indices


def _is_valid_covering(blocks, sub_count, n_subs):
    """Check if all subsets are covered (all counts >= 1)."""
    for c in sub_count:
        if c < 1:
            return False
    return True


def _simulated_annealing(blocks, v, k, t, deadline):
    """Simulated annealing to minimize block count while maintaining coverage."""
    subs, sub_index, point_sub_map = _build_sub_index(v, t)
    n_subs = len(subs)

    # Initialize coverage counts
    sub_count = [0] * n_subs
    for block in blocks:
        for idx in _compute_sub_indices(block, sub_index, t):
            sub_count[idx] += 1

    # Verify initial validity
    if not _is_valid_covering(blocks, sub_count, n_subs):
        return blocks

    best_blocks = blocks[:]
    best_len = len(blocks)
    current_blocks = blocks[:]

    # Annealing parameters
    t_start = 1.0
    t_end = 0.01
    cooling_rate = 0.995
    max_iter = 100000

    rng = random.Random(42)
    current_temp = t_start
    iter_count = 0

    while current_temp > t_end and iter_count < max_iter and time.time() < deadline:
        iter_count += 1

        # Choose a random move
        move_type = rng.random()
        if move_type < 0.4 and len(current_blocks) > 0:
            # Try removing a block
            idx = rng.randint(0, len(current_blocks) - 1)
            block = current_blocks[idx]
            block_indices = _compute_sub_indices(block, sub_index, t)

            # Check if removal keeps coverage valid
            can_remove = True
            for si in block_indices:
                if sub_count[si] <= 1:
                    can_remove = False
                    break

            if can_remove:
                for si in block_indices:
                    sub_count[si] -= 1
                current_blocks.pop(idx)

                new_len = len(current_blocks)
                if new_len < best_len:
                    best_blocks = current_blocks[:]
                    best_len = new_len
                current_temp *= cooling_rate
                continue
            else:
                # Revert (nothing changed)
                pass

        elif move_type < 0.8 and len(current_blocks) > 0:
            # Try swapping a block
            idx = rng.randint(0, len(current_blocks) - 1)
            old_block = current_blocks[idx]
            old_indices = _compute_sub_indices(old_block, sub_index, t)

            # Find unique subsets covered by old_block
            unique_indices = [si for si in old_indices if sub_count[si] == 1]

            # Generate a candidate block that covers the unique subsets
            # The candidate must contain all points that are in unique subsets
            # but not covered by other blocks
            required_points = set()
            for si in unique_indices:
                required_points.update(subs[si])

            if len(required_points) > k:
                # Can't cover all unique subsets with one block
                current_temp *= cooling_rate
                continue

            # Try to build a candidate block containing required_points
            # Fill remaining slots with random points
            if len(required_points) <= k:
                remaining = [p for p in range(v) if p not in required_points]
                fill_needed = k - len(required_points)
                if fill_needed > 0 and len(remaining) >= fill_needed:
                    fill = rng.sample(remaining, fill_needed)
                    candidate = sorted(list(required_points) + fill)
                elif len(required_points) == k:
                    candidate = sorted(required_points)
                else:
                    current_temp *= cooling_rate
                    continue
            else:
                current_temp *= cooling_rate
                continue

            # Check if candidate is the same as old
            if candidate == old_block:
                current_temp *= cooling_rate
                continue

            # Compute new coverage if we swap
            new_indices = _compute_sub_indices(candidate, sub_index, t)
            new_index_set = set(new_indices)

            # Simulate the swap
            valid = True
            for si in old_indices:
                if sub_count[si] == 1 and si not in new_index_set:
                    valid = False
                    break

            if not valid:
                current_temp *= cooling_rate
                continue

            # Apply the swap
            for si in old_indices:
                sub_count[si] -= 1
            for si in new_indices:
                sub_count[si] += 1
            current_blocks[idx] = candidate

            # After swap, try to remove the old block's position if it becomes redundant
            # Actually we just swapped, so no removal here. But we can try removing
            # other blocks that became redundant.
            # For now, just accept the swap and continue.

            # Check if we can now remove any block
            for j in range(len(current_blocks)):
                b = current_blocks[j]
                b_indices = _compute_sub_indices(b, sub_index, t)
                can_remove = True
                for si in b_indices:
                    if sub_count[si] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for si in b_indices:
                        sub_count[si] -= 1
                    current_blocks.pop(j)
                    new_len = len(current_blocks)
                    if new_len < best_len:
                        best_blocks = current_blocks[:]
                        best_len = new_len
                    break

            current_temp *= cooling_rate

        else:
            # Occasionally, try a random replacement of a block with a random block
            if len(current_blocks) > 0:
                idx = rng.randint(0, len(current_blocks) - 1)
                old_block = current_blocks[idx]
                old_indices = _compute_sub_indices(old_block, sub_index, t)
                unique_indices = [si for si in old_indices if sub_count[si] == 1]

                if not unique_indices:
                    # Block has no unique coverage, just remove it
                    for si in old_indices:
                        sub_count[si] -= 1
                    current_blocks.pop(idx)
                    new_len = len(current_blocks)
                    if new_len < best_len:
                        best_blocks = current_blocks[:]
                        best_len = new_len
                    current_temp *= cooling_rate
                    continue

                # Generate random candidate
                candidate = sorted(rng.sample(range(v), k))
                new_indices = _compute_sub_indices(candidate, sub_index, t)
                new_index_set = set(new_indices)

                # Check if candidate covers all unique subsets
                valid = True
                for si in unique_indices:
                    if si not in new_index_set:
                        valid = False
                        break

                if valid:
                    for si in old_indices:
                        sub_count[si] -= 1
                    for si in new_indices:
                        sub_count[si] += 1
                    current_blocks[idx] = candidate
                    current_temp *= cooling_rate
                else:
                    current_temp *= cooling_rate
            else:
                current_temp *= cooling_rate

    # Restore best solution
    # Rebuild sub_count from best_blocks
    sub_count = [0] * n_subs
    for block in best_blocks:
        for idx in _compute_sub_indices(block, sub_index, t):
            sub_count[idx] += 1

    return best_blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for safety

    # Phase 1: Greedy construction
    blocks = _greedy_construct(v, k, t, deadline)

    if time.time() > deadline:
        return blocks

    # Phase 2: Remove redundant blocks
    blocks = _remove_redundant_blocks(blocks, v, k, t, deadline)

    if time.time() > deadline:
        return blocks

    # Phase 3: Simulated annealing
    blocks = _simulated_annealing(blocks, v, k, t, deadline)

    # Phase 4: Final verification - ensure every t-subset is covered
    subs, sub_index, _ = _build_sub_index(v, t)
    covered = [False] * len(subs)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            idx = sub_index.get(sub, -1)
            if idx >= 0:
                covered[idx] = True

    if not all(covered):
        # Fallback: add blocks to cover missing subsets
        missing = [subs[i] for i in range(len(subs)) if not covered[i]]
        for msub in missing:
            remaining = [p for p in range(v) if p not in msub]
            extra = random.sample(remaining, k - t)
            block = sorted(list(msub) + extra)
            if block not in blocks:
                blocks.append(block)

    return blocks
