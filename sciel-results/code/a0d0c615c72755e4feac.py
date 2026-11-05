import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s kill

    # Precomputation
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    id_to_triplet = [None] * n_triplets
    for i, trip in enumerate(all_triplets):
        id_to_triplet[i] = trip

    # Precompute all blocks and their masks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Precompute mask for each block
    # mask[b] is an integer with bits set for each t-subset in block b
    block_masks = np.zeros(n_blocks, dtype=object)
    block_triplet_ids = [None] * n_blocks  # list of triplet ids for each block

    for bi, block in enumerate(all_blocks):
        mask = 0
        ids = []
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
            ids.append(tid)
        block_masks[bi] = mask
        block_triplet_ids[bi] = ids
        # Also store as a frozenset for fast membership
        block_triplet_ids_set = [None] * n_blocks  # placeholder, fill below

    # Precompute set versions for fast lookup in optimizer
    block_triplet_ids_set = [set(ids) for ids in block_triplet_ids]

    # For fast lookup: given a set of points, find blocks containing them
    # Build index: for each point, list of block indices containing it
    point_to_blocks = defaultdict(list)
    for bi, block in enumerate(all_blocks):
        for p in block:
            point_to_blocks[p].append(bi)

    def get_blocks_containing(points):
        """Return list of block indices containing all points in the set."""
        if len(points) == 0:
            return list(range(n_blocks))
        if len(points) > k:
            return []
        # Start with blocks containing the first point, intersect with others
        first = next(iter(points))
        candidate_set = set(point_to_blocks[first])
        for p in points:
            if p == first:
                continue
            candidate_set &= set(point_to_blocks[p])
            if not candidate_set:
                return []
        return list(candidate_set)

    def get_blocks_containing_frozenset(points):
        """Return list of block indices containing all points (points as iterable)."""
        pts = list(points)
        if len(pts) == 0:
            return list(range(n_blocks))
        if len(pts) > k:
            return []
        first = pts[0]
        candidate_set = set(point_to_blocks[first])
        for p in pts[1:]:
            candidate_set &= set(point_to_blocks[p])
            if not candidate_set:
                return []
        return list(candidate_set)

    # Greedy Construction with max-gain selection
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = (1 << n_triplets) - 1
        blocks = []
        used_block_indices = set()

        while uncovered:
            # Pick a random uncovered triplet to guide the search
            idx = -1
            for _ in range(20):
                cand = random.randint(0, n_triplets - 1)
                if (uncovered >> cand) & 1:
                    idx = cand
                    break
            if idx == -1:
                lsb = uncovered & -uncovered
                idx = lsb.bit_length() - 1

            start_triplet = id_to_triplet[idx]

            # Find best block containing start_triplet using max gain
            best_block_idx = -1
            best_gain = 0
            best_mask = 0

            # Get blocks containing all 3 points of start_triplet
            candidate_indices = get_blocks_containing(start_triplet)

            for bi in candidate_indices:
                if bi in used_block_indices:
                    continue
                b_mask = block_masks[bi]
                gain = (b_mask & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = bi
                    best_mask = b_mask

            if best_block_idx == -1:
                # Fallback: allow reuse or pick any block covering the triplet
                for bi in candidate_indices:
                    b_mask = block_masks[bi]
                    gain = (b_mask & uncovered).bit_count()
                    if gain > 0:
                        best_block_idx = bi
                        best_mask = b_mask
                        break

            if best_block_idx == -1:
                # Should not happen if covering is possible
                break

            blocks.append(all_blocks[best_block_idx])
            used_block_indices.add(best_block_idx)
            uncovered &= ~best_mask

        return blocks

    # Local Search Optimizer
    def optimize(blocks, time_limit):
        counts = np.zeros(n_triplets, dtype=np.int32)

        # Precompute triplet id lists for each block in the current solution
        def get_block_triplet_ids(b):
            return [triplet_to_id[triplet] for triplet in itertools.combinations(b, t)]

        def recalc_counts(b_list):
            counts[:] = 0
            for b in b_list:
                for tid in get_block_triplet_ids(b):
                    counts[tid] += 1

        recalc_counts(blocks)

        no_improve_count = 0
        max_no_improve = 50

        while time.time() < time_limit:
            improved = False
            n = len(blocks)

            # 1. Try Remove Redundant
            for i in range(n):
                b = blocks[i]
                redundant = True
                for tid in get_block_triplet_ids(b):
                    if counts[tid] == 1:
                        redundant = False
                        break
                if redundant:
                    for tid in get_block_triplet_ids(b):
                        counts[tid] -= 1
                    blocks.pop(i)
                    improved = True
                    break

            # 2. Try 2-for-1 replacement
            if not improved and n >= 2:
                # Compute unique load for each block
                unique_loads = []
                for i in range(n):
                    b = blocks[i]
                    load = 0
                    for tid in get_block_triplet_ids(b):
                        if counts[tid] == 1:
                            load += 1
                    unique_loads.append((load, i))

                unique_loads.sort(reverse=True)

                # Try pairs from top-loaded blocks
                tried_pairs = set()
                max_pairs = 25
                for load1, i in unique_loads[:max_pairs]:
                    if load1 == 0:
                        continue
                    for load2, j in unique_loads[:max_pairs]:
                        if i == j or i > j:
                            continue
                        if (i, j) in tried_pairs:
                            continue
                        tried_pairs.add((i, j))

                        b1 = blocks[i]
                        b2 = blocks[j]

                        # Get triplet ids
                        ids_b1 = set(get_block_triplet_ids(b1))
                        ids_b2 = set(get_block_triplet_ids(b2))
                        all_involved = ids_b1 | ids_b2

                        needed_triplets = []
                        points_needed = set()
                        for tid in all_involved:
                            if counts[tid] == 1:
                                needed_triplets.append(tid)
                                points_needed.update(id_to_triplet[tid])

                        if not needed_triplets:
                            # Both redundant
                            # Remove j first (higher index) to keep indices valid
                            for tid in get_block_triplet_ids(b2):
                                counts[tid] -= 1
                            blocks.pop(j)
                            for tid in get_block_triplet_ids(b1):
                                counts[tid] -= 1
                            blocks.pop(i)
                            improved = True
                            break

                        if len(points_needed) > k:
                            continue

                        # Find replacement block
                        candidate_indices = get_blocks_containing(points_needed)
                        found = False
                        for bi in candidate_indices:
                            new_block = all_blocks[bi]
                            if set(new_block) == set(b1) or set(new_block) == set(b2):
                                continue
                            new_ids = block_triplet_ids_set[bi]
                            covers = all(tid in new_ids for tid in needed_triplets)
                            if covers:
                                # Perform swap
                                for tid in get_block_triplet_ids(b2):
                                    counts[tid] -= 1
                                blocks.pop(j)
                                for tid in get_block_triplet_ids(b1):
                                    counts[tid] -= 1
                                blocks.pop(i)
                                blocks.append(new_block)
                                for tid in block_triplet_ids[bi]:
                                    counts[tid] += 1
                                improved = True
                                found = True
                                break

                        if improved:
                            break
                        if time.time() > time_limit - 3:
                            break

            # 3. Try 1-for-1 replacement (swap)
            if not improved and n >= 1:
                # Try multiple random blocks
                for _ in range(5):
                    i = random.randint(0, n - 1)
                    b1 = blocks[i]
                    ids_b1 = set(get_block_triplet_ids(b1))

                    needed_triplets = []
                    points_needed = set()
                    for tid in ids_b1:
                        if counts[tid] == 1:
                            needed_triplets.append(tid)
                            points_needed.update(id_to_triplet[tid])

                    if needed_triplets and len(points_needed) <= k:
                        candidate_indices = get_blocks_containing(points_needed)
                        for bi in candidate_indices:
                            new_block = all_blocks[bi]
                            if set(new_block) == set(b1):
                                continue
                            new_ids = block_triplet_ids_set[bi]
                            covers = all(tid in new_ids for tid in needed_triplets)
                            if covers:
                                for tid in get_block_triplet_ids(b1):
                                    counts[tid] -= 1
                                blocks.pop(i)
                                blocks.append(new_block)
                                for tid in block_triplet_ids[bi]:
                                    counts[tid] += 1
                                improved = True
                                break

                    if improved:
                        break

            if not improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
            else:
                no_improve_count = 0

        return blocks

    best_blocks = None
    best_len = float('inf')

    # Run multiple restarts
    n_restarts = 12
    for seed in range(n_restarts):
        if time.time() > deadline - 10:
            break

        # Greedy init
        init_blocks = greedy_construct(seed)

        # Optimize with remaining time
        opt_blocks = optimize(init_blocks, deadline - 5)

        # Verify validity
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in opt_blocks:
            for triplet in itertools.combinations(b, t):
                counts[triplet_to_id[triplet]] += 1

        if np.min(counts) > 0:
            if len(opt_blocks) < best_len:
                best_len = len(opt_blocks)
                best_blocks = opt_blocks

    if best_blocks is None:
        best_blocks = greedy_construct(0)

    return list(best_blocks)


if __name__ == "__main__":
    pass
