# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import comb


def construct(v, k, t):
    """Covering design C(v,k,t) via fast bitmask greedy set cover."""
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Precompute all t-subsets as bitmasks
    all_tsubs = []
    tsub_to_idx = {}
    for c in combinations(range(v), t):
        mask = 0
        for x in c:
            mask |= (1 << x)
        idx = len(all_tsubs)
        all_tsubs.append(mask)
        tsub_to_idx[mask] = idx

    total_t = len(all_tsubs)

    # For each point, precompute which t-subsets contain it
    # tsub_contains_point[p] = set of t-subset indices that include point p
    tsub_contains_point = [set() for _ in range(v)]
    for idx, mask in enumerate(all_tsubs):
        for p in range(v):
            if mask & (1 << p):
                tsub_contains_point[p].add(idx)

    # uncovered: use a list of booleans for O(1) access
    uncovered = bytearray([1]) * total_t
    uncovered_count = total_t

    blocks = []
    rng = random.Random(42)
    used_blocks = set()

    while uncovered_count > 0:
        if time.time() > deadline - 5.0:
            break

        # Pick an uncovered t-subset to seed a new block
        # Use a random sample to find one with high "clustering" potential
        # (many uncovered t-subsets nearby)
        # For speed, just pick a random uncovered one
        # Find a random uncovered index
        while True:
            idx = rng.randint(0, total_t - 1)
            if uncovered[idx]:
                break

        seed_mask = all_tsubs[idx]
        seed_points = [p for p in range(v) if seed_mask & (1 << p)]

        # Build block greedily: start with seed, add points one by one
        block_mask = seed_mask
        block_points = list(seed_points)
        remaining_slots = k - len(block_points)

        for _ in range(remaining_slots):
            if time.time() > deadline - 5.0:
                break
            if uncovered_count == 0:
                break

            # For each candidate point x not in block, compute gain:
            # number of currently-uncovered t-subsets that would be covered
            # by adding x to block
            best_x = None
            best_gain = -1
            for x in range(v):
                if block_mask & (1 << x):
                    continue
                # t-subsets of (block ∪ {x}) that contain x and are uncovered
                # = choose t-1 from block_points, combine with x
                gain = 0
                for rest in combinations(block_points, t - 1):
                    s_mask = (1 << x)
                    for p in rest:
                        s_mask |= (1 << p)
                    if s_mask in tsub_to_idx:
                        s_idx = tsub_to_idx[s_mask]
                        if uncovered[s_idx]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x is None or best_gain <= 0:
                # Pick any unused point
                for x in range(v):
                    if not (block_mask & (1 << x)):
                        best_x = x
                        break
                if best_x is None:
                    break

            block_mask |= (1 << best_x)
            block_points.append(best_x)

        # Finalize block
        if len(block_points) == k:
            block_sorted = tuple(sorted(block_points))
            if block_sorted not in used_blocks:
                used_blocks.add(block_sorted)
                blocks.append(list(block_sorted))
                # Remove all covered t-subsets
                for c in combinations(block_sorted, t):
                    s_mask = 0
                    for p in c:
                        s_mask |= (1 << p)
                    s_idx = tsub_to_idx[s_mask]
                    if uncovered[s_idx]:
                        uncovered[s_idx] = 0
                        uncovered_count -= 1

    # If we didn't finish in time, do a fast emergency pass
    if uncovered_count > 0 and time.time() < deadline - 2.0:
        while uncovered_count > 0 and time.time() < deadline - 0.5:
            # Find any uncovered t-subset
            found = False
            for idx in range(total_t):
                if uncovered[idx]:
                    found = True
                    break
            if not found:
                break

            seed_mask = all_tsubs[idx]
            seed_points = [p for p in range(v) if seed_mask & (1 << p)]
            block_mask = seed_mask
            block_points = list(seed_points)

            while len(block_points) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if block_mask & (1 << x):
                        continue
                    gain = 0
                    for rest in combinations(block_points, t - 1):
                        s_mask = (1 << x)
                        for p in rest:
                            s_mask |= (1 << p)
                        if s_mask in tsub_to_idx:
                            s_idx = tsub_to_idx[s_mask]
                            if uncovered[s_idx]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    for x in range(v):
                        if not (block_mask & (1 << x)):
                            best_x = x
                            break
                    if best_x is None:
                        break
                block_mask |= (1 << best_x)
                block_points.append(best_x)

            if len(block_points) == k:
                block_sorted = tuple(sorted(block_points))
                if block_sorted not in used_blocks:
                    used_blocks.add(block_sorted)
                    blocks.append(list(block_sorted))
                    for c in combinations(block_sorted, t):
                        s_mask = 0
                        for p in c:
                            s_mask |= (1 << p)
                        s_idx = tsub_to_idx[s_mask]
                        if uncovered[s_idx]:
                            uncovered[s_idx] = 0
                            uncovered_count -= 1

    return blocks
# EVOLVE-BLOCK-END
