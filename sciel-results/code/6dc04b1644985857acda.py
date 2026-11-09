"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def construct(v, k, t):
    deadline = time.time() + 130.0  # safety margin before 150s kill
    rng = random.Random(42)

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # For each point, precompute the list of t-subset indices it belongs to
    point_tsubs = [[] for _ in range(v)]
    for idx, ts in enumerate(all_tsubs):
        for p in ts:
            point_tsubs[p].append(idx)

    # Greedy covering with reverse indexing
    def greedy_cover(seed=None):
        if seed is not None:
            local_rng = random.Random(seed)
        else:
            local_rng = rng

        uncovered = np.ones(n_tsubs, dtype=np.int32)  # 1 if uncovered
        blocks = []

        while uncovered.any():
            # Pick an uncovered t-subset (randomize among uncovered for diversity)
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            start_idx = uncovered_indices[local_rng.randint(0, len(uncovered_indices) - 1)]
            start_tsub = all_tsubs[start_idx]

            block = list(start_tsub)
            block_set = set(block)

            # Build the set of t-subsets covered by this block so far
            block_tsubs = set()
            for combo in itertools.combinations(block, t):
                block_tsubs.add(tsub_to_idx[combo])

            while len(block) < k:
                # Find the point that covers the most uncovered t-subsets
                best_point = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    # New t-subsets covered by adding x
                    new_tsubs = 0
                    for combo in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(combo + (x,)))
                        idx = tsub_to_idx[ts]
                        if uncovered[idx]:
                            new_tsubs += 1
                    if new_tsubs > best_gain:
                        best_gain = new_tsubs
                        best_point = x

                if best_point == -1:
                    # Pick any remaining point
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_point = remaining[local_rng.randint(0, len(remaining) - 1)]
                    else:
                        break

                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            blocks.append(tuple(block))

            # Update uncovered
            for combo in itertools.combinations(block, t):
                idx = tsub_to_idx[combo]
                uncovered[idx] = 0

        return [list(b) for b in blocks]

    # Local search: try to remove blocks
    def local_search_remove(blocks, seed=None):
        if seed is not None:
            local_rng = random.Random(seed)
        else:
            local_rng = rng

        blocks = [list(b) for b in blocks]
        improved = True
        while improved:
            improved = False
            # Count coverage of each t-subset
            cover_count = np.zeros(n_tsubs, dtype=np.int32)
            for block in blocks:
                for combo in itertools.combinations(block, t):
                    idx = tsub_to_idx[combo]
                    cover_count[idx] += 1

            # Try removing each block if it doesn't break coverage
            for i in range(len(blocks) - 1, -1, -1):
                block = blocks[i]
                # Check if all t-subsets in this block have cover_count > 1
                can_remove = True
                for combo in itertools.combinations(block, t):
                    idx = tsub_to_idx[combo]
                    if cover_count[idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for combo in itertools.combinations(block, t):
                        idx = tsub_to_idx[combo]
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    improved = True

        return blocks

    # Simulated annealing: 1-for-1 exchanges
    def simulated_annealing(blocks, seed=None, max_iters=50000):
        if seed is not None:
            local_rng = random.Random(seed)
        else:
            local_rng = rng

        blocks = [list(b) for b in blocks]
        n_blocks = len(blocks)

        # Build coverage count
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for block in blocks:
            for combo in itertools.combinations(block, t):
                idx = tsub_to_idx[combo]
                cover_count[idx] += 1

        # Precompute for each block the list of t-subset indices it covers
        block_tsub_indices = []
        for block in blocks:
            idxs = []
            for combo in itertools.combinations(block, t):
                idxs.append(tsub_to_idx[combo])
            block_tsub_indices.append(idxs)

        current_cost = n_blocks
        best_blocks = [b[:] for b in blocks]
        best_cost = n_blocks

        temperature = 1.0
        cooling = 0.9995

        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                break

            # Try: replace a random point in a random block with a different point
            i = local_rng.randint(0, n_blocks - 1)
            block = blocks[i]
            pos = local_rng.randint(0, k - 1)
            old_point = block[pos]

            # Choose a new point not already in the block
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                continue
            new_point = candidates[local_rng.randint(0, len(candidates) - 1)]

            # Compute change in coverage
            # Remove old point's contribution, add new point's contribution
            delta_cover = {}  # tsub_idx -> change in cover count

            # T-subsets involving old_point that are in this block
            for combo in itertools.combinations(block, t):
                if old_point in combo:
                    idx = tsub_to_idx[combo]
                    delta_cover[idx] = delta_cover.get(idx, 0) - 1

            # T-subsets involving new_point with the other t-1 points in the block
            other_points = [x for x in block if x != old_point]
            for combo in itertools.combinations(other_points, t - 1):
                ts = tuple(sorted(combo + (new_point,)))
                idx = tsub_to_idx[ts]
                delta_cover[idx] = delta_cover.get(idx, 0) + 1

            # Check validity: no t-subset should go to 0
            valid = True
            for idx, delta in delta_cover.items():
                if cover_count[idx] + delta < 1:
                    valid = False
                    break

            if valid:
                # Accept the move
                block[pos] = new_point
                block.sort()
                for idx, delta in delta_cover.items():
                    cover_count[idx] += delta
                # Update block_tsub_indices
                new_idxs = []
                for combo in itertools.combinations(block, t):
                    new_idxs.append(tsub_to_idx[combo])
                block_tsub_indices[i] = new_idxs

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = [b[:] for b in blocks]

            temperature *= cooling

        return best_blocks

    # Run greedy
    initial_blocks = greedy_cover(seed=42)
    initial_count = len(initial_blocks)

    # Local search to remove redundant blocks
    blocks = local_search_remove(initial_blocks, seed=43)
    best_count = len(blocks)
    best_result = [b[:] for b in blocks]

    # Simulated annealing
    if time.time() < deadline - 10.0:
        sa_blocks = simulated_annealing(blocks, seed=44, max_iters=100000)
        if len(sa_blocks) < best_count:
            best_count = len(sa_blocks)
            best_result = [b[:] for b in sa_blocks]

    # Try a few more greedy runs with different seeds if time permits
    for seed in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]:
        if time.time() > deadline - 15.0:
            break
        g_blocks = greedy_cover(seed=seed)
        g_blocks = local_search_remove(g_blocks, seed=seed + 100)
        if len(g_blocks) < best_count:
            best_count = len(g_blocks)
            best_result = [b[:] for b in g_blocks]
            if time.time() < deadline - 10.0:
                sa_blocks = simulated_annealing(g_blocks, seed=seed + 200, max_iters=50000)
                if len(sa_blocks) < best_count:
                    best_count = len(sa_blocks)
                    best_result = [b[:] for b in sa_blocks]

    return best_result
# EVOLVE-BLOCK-END
