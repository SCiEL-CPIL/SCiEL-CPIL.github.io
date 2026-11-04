import itertools
import time
import random
import numpy as np
from collections import defaultdict
from itertools import combinations

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 143  # 7s safety margin

    # Precomputation
    all_triplets = list(combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}

    # Map triplet id to the triplet itself for reverse lookup
    id_to_triplet = [None] * n_triplets
    for i, trip in enumerate(all_triplets):
        id_to_triplet[i] = trip

    # Precompute all C(k,t) triplets within a block given its elements
    def get_block_triplet_ids(block):
        """Return list of triplet IDs for all t-subsets of block."""
        return [triplet_to_id[trip] for trip in combinations(block, t)]

    # Enumerate blocks containing a set of points
    def blocks_containing(points):
        points = frozenset(points)
        if len(points) > k:
            return
        remaining = [p for p in range(v) if p not in points]
        need = k - len(points)
        if need < 0:
            return
        for comb in combinations(remaining, need):
            yield tuple(sorted(points | set(comb)))

    # Greedy Construction with ratio-based selection
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = (1 << n_triplets) - 1
        blocks = []

        while uncovered:
            # Pick a random uncovered triplet to guide the search
            idx = -1
            for _ in range(15):
                cand = random.randint(0, n_triplets - 1)
                if (uncovered >> cand) & 1:
                    idx = cand
                    break
            if idx == -1:
                lsb = uncovered & -uncovered
                idx = lsb.bit_length() - 1

            start_triplet = id_to_triplet[idx]

            # Find best block containing start_triplet using ratio: gain / cost
            best_block = None
            best_ratio = -1
            best_mask = 0
            best_gain = 0

            # Enumerate all blocks containing start_triplet
            for block in blocks_containing(start_triplet):
                # Compute mask
                b_mask = 0
                for triplet in combinations(block, t):
                    tid = triplet_to_id[triplet]
                    b_mask |= (1 << tid)
                gain = (b_mask & uncovered).bit_count()
                cost = b_mask.bit_count()
                ratio = gain / max(cost, 1)

                if ratio > best_ratio or (ratio == best_ratio and gain > best_gain):
                    best_ratio = ratio
                    best_block = block
                    best_mask = b_mask
                    best_gain = gain

            blocks.append(best_block)
            uncovered &= ~best_mask

        return blocks

    # Local Search Optimizer
    def optimize(blocks, time_limit):
        # Maintain counts
        counts = np.zeros(n_triplets, dtype=np.int32)

        def recalc_counts(b_list):
            counts[:] = 0
            for b in b_list:
                for triplet in combinations(b, t):
                    counts[triplet_to_id[triplet]] += 1

        recalc_counts(blocks)

        # Precompute block masks for faster operations
        def compute_block_mask(block):
            mask = 0
            for triplet in combinations(block, t):
                tid = triplet_to_id[triplet]
                mask |= (1 << tid)
            return mask

        # Optimization Loop
        no_improve_count = 0
        max_no_improve = 20

        while time.time() < time_limit:
            improved = False
            n = len(blocks)
            if n == 0:
                break

            # 1. Try Remove Redundant
            for i in range(n):
                b = blocks[i]
                redundant = True
                for triplet in combinations(b, t):
                    if counts[triplet_to_id[triplet]] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for triplet in combinations(b, t):
                        counts[triplet_to_id[triplet]] -= 1
                    blocks.pop(i)
                    improved = True
                    break

            # 2. Try 2-for-1 (fixed pop order)
            if not improved:
                n = len(blocks)
                # Identify blocks with high unique load
                unique_loads = []
                for i in range(n):
                    b = blocks[i]
                    load = 0
                    for triplet in combinations(b, t):
                        if counts[triplet_to_id[triplet]] == 1:
                            load += 1
                    unique_loads.append((load, i))

                # Sort by load descending, try top pairs
                unique_loads.sort(reverse=True)
                tried_pairs = set()
                max_pairs_to_try = 10

                for load1, i in unique_loads[:8]:
                    if load1 == 0:
                        continue
                    for load2, j in unique_loads[:8]:
                        if i == j:
                            continue
                        pair_key = (min(i, j), max(i, j))
                        if pair_key in tried_pairs:
                            continue
                        tried_pairs.add(pair_key)
                        if len(tried_pairs) > max_pairs_to_try:
                            break

                        b1 = blocks[i]
                        b2 = blocks[j]

                        # Compute needed triplets (those with count == 1)
                        ids_b1 = set(triplet_to_id[triplet] for triplet in combinations(b1, t))
                        ids_b2 = set(triplet_to_id[triplet] for triplet in combinations(b2, t))
                        all_involved = ids_b1 | ids_b2

                        needed_triplets = []
                        points_needed = set()
                        for tid in all_involved:
                            if counts[tid] == 1:
                                needed_triplets.append(tid)
                                points_needed.update(id_to_triplet[tid])

                        if not needed_triplets:
                            # Both redundant
                            # Remove j first (higher index), then i
                            for triplet in combinations(b2, t):
                                counts[triplet_to_id[triplet]] -= 1
                            blocks.pop(j)
                            for triplet in combinations(b1, t):
                                counts[triplet_to_id[triplet]] -= 1
                            blocks.pop(i)
                            improved = True
                            break

                        if len(points_needed) > k:
                            continue

                        # Enumerate blocks containing points_needed
                        found_replacement = False
                        for new_block in blocks_containing(points_needed):
                            if set(new_block) == set(b1) or set(new_block) == set(b2):
                                continue

                            # Check coverage
                            new_ids = set(triplet_to_id[triplet] for triplet in combinations(new_block, t))
                            covers = all(tid in new_ids for tid in needed_triplets)

                            if covers:
                                # Remove j first, then i
                                for triplet in combinations(b2, t):
                                    counts[triplet_to_id[triplet]] -= 1
                                blocks.pop(j)
                                for triplet in combinations(b1, t):
                                    counts[triplet_to_id[triplet]] -= 1
                                blocks.pop(i)

                                blocks.append(new_block)
                                for triplet in combinations(new_block, t):
                                    counts[triplet_to_id[triplet]] += 1

                                improved = True
                                found_replacement = True
                                break

                        if improved:
                            break
                        if time.time() > time_limit - 2:
                            break

            # 3. Try 2-for-2 exchange: replace two blocks with two different blocks
            if not improved:
                n = len(blocks)
                if n < 2:
                    pass
                else:
                    # Find pair of blocks with high combined unique load
                    unique_loads = []
                    for i in range(n):
                        b = blocks[i]
                        load = 0
                        for triplet in combinations(b, t):
                            if counts[triplet_to_id[triplet]] == 1:
                                load += 1
                        unique_loads.append((load, i))

                    unique_loads.sort(reverse=True)
                    tried_pairs = set()
                    max_pairs_to_try = 5

                    for load1, i in unique_loads[:6]:
                        if load1 == 0:
                            continue
                        for load2, j in unique_loads[:6]:
                            if i >= j:
                                continue
                            pair_key = (i, j)
                            if pair_key in tried_pairs:
                                continue
                            tried_pairs.add(pair_key)
                            if len(tried_pairs) > max_pairs_to_try:
                                break

                            b1 = blocks[i]
                            b2 = blocks[j]

                            # Get unique triplets for each block
                            ids_b1 = set(triplet_to_id[triplet] for triplet in combinations(b1, t))
                            ids_b2 = set(triplet_to_id[triplet] for triplet in combinations(b2, t))

                            unique_b1 = [tid for tid in ids_b1 if counts[tid] == 1]
                            unique_b2 = [tid for tid in ids_b2 if counts[tid] == 1]

                            if not unique_b1 and not unique_b2:
                                # Both redundant, remove both
                                for triplet in combinations(b2, t):
                                    counts[triplet_to_id[triplet]] -= 1
                                blocks.pop(j)
                                for triplet in combinations(b1, t):
                                    counts[triplet_to_id[triplet]] -= 1
                                blocks.pop(i)
                                improved = True
                                break

                            # Try to find replacement for b1 that covers its unique triplets
                            # and replacement for b2 that covers its unique triplets
                            # without overlapping too much

                            points_needed_1 = set()
                            for tid in unique_b1:
                                points_needed_1.update(id_to_triplet[tid])

                            points_needed_2 = set()
                            for tid in unique_b2:
                                points_needed_2.update(id_to_triplet[tid])

                            if len(points_needed_1) > k or len(points_needed_2) > k:
                                continue

                            # Find candidate replacements
                            candidates_1 = []
                            for new_block in blocks_containing(points_needed_1):
                                if set(new_block) == set(b1) or set(new_block) == set(b2):
                                    continue
                                new_ids = set(triplet_to_id[triplet] for triplet in combinations(new_block, t))
                                if all(tid in new_ids for tid in unique_b1):
                                    candidates_1.append(new_block)

                            candidates_2 = []
                            for new_block in blocks_containing(points_needed_2):
                                if set(new_block) == set(b1) or set(new_block) == set(b2):
                                    continue
                                new_ids = set(triplet_to_id[triplet] for triplet in combinations(new_block, t))
                                if all(tid in new_ids for tid in unique_b2):
                                    candidates_2.append(new_block)

                            # Try combinations of candidates
                            for c1 in candidates_1[:3]:
                                for c2 in candidates_2[:3]:
                                    if set(c1) == set(c2):
                                        continue
                                    # Check that c1 doesn't break b2's coverage and vice versa
                                    c1_ids = set(triplet_to_id[triplet] for triplet in combinations(c1, t))
                                    c2_ids = set(triplet_to_id[triplet] for triplet in combinations(c2, t))

                                    # Verify all unique triplets are still covered
                                    all_covered = True
                                    for tid in unique_b1:
                                        if tid not in c1_ids and tid not in c2_ids:
                                            all_covered = False
                                            break
                                    if all_covered:
                                        for tid in unique_b2:
                                            if tid not in c2_ids and tid not in c1_ids:
                                                all_covered = False
                                                break

                                    if all_covered:
                                        # Apply the exchange
                                        for triplet in combinations(b2, t):
                                            counts[triplet_to_id[triplet]] -= 1
                                        blocks.pop(j)
                                        for triplet in combinations(b1, t):
                                            counts[triplet_to_id[triplet]] -= 1
                                        blocks.pop(i)

                                        blocks.append(c1)
                                        blocks.append(c2)
                                        for triplet in combinations(c1, t):
                                            counts[triplet_to_id[triplet]] += 1
                                        for triplet in combinations(c2, t):
                                            counts[triplet_to_id[triplet]] += 1

                                        improved = True
                                        break

                                if improved:
                                    break

                            if improved:
                                break
                            if time.time() > time_limit - 2:
                                break

            # 4. Try 1-for-1 replacement
            if not improved:
                n = len(blocks)
                if n == 0:
                    break

                # Pick block with highest unique load
                best_i = -1
                best_load = -1
                for i in range(n):
                    b = blocks[i]
                    load = 0
                    for triplet in combinations(b, t):
                        if counts[triplet_to_id[triplet]] == 1:
                            load += 1
                    if load > best_load:
                        best_load = load
                        best_i = i

                if best_i == -1 or best_load == 0:
                    continue

                i = best_i
                b1 = blocks[i]
                ids_b1 = set(triplet_to_id[triplet] for triplet in combinations(b1, t))

                needed_triplets = []
                points_needed = set()
                for tid in ids_b1:
                    if counts[tid] == 1:
                        needed_triplets.append(tid)
                        points_needed.update(id_to_triplet[tid])

                if not needed_triplets:
                    continue

                if len(points_needed) > k:
                    continue

                for new_block in blocks_containing(points_needed):
                    if set(new_block) == set(b1):
                        continue
                    new_ids = set(triplet_to_id[triplet] for triplet in combinations(new_block, t))
                    covers = all(tid in new_ids for tid in needed_triplets)
                    if covers:
                        # Replace
                        for triplet in combinations(b1, t):
                            counts[triplet_to_id[triplet]] -= 1
                        blocks[i] = new_block
                        for triplet in combinations(new_block, t):
                            counts[triplet_to_id[triplet]] += 1
                        improved = True
                        break

            # 5. Try shift move: replace one point in a block
            if not improved:
                n = len(blocks)
                if n == 0:
                    break

                # Pick a random block
                i = random.randint(0, n - 1)
                b = list(blocks[i])

                # Try replacing each point with each other point
                for pos in range(k):
                    old_point = b[pos]
                    for new_point in range(v):
                        if new_point == old_point or new_point in b:
                            continue

                        # Create candidate block
                        candidate = b.copy()
                        candidate[pos] = new_point
                        candidate.sort()
                        candidate = tuple(candidate)

                        if candidate in [tuple(bl) for bl in blocks]:
                            continue

                        # Check if candidate covers all unique triplets of b
                        old_ids = set(triplet_to_id[triplet] for triplet in combinations(b, t))
                        unique_old = [tid for tid in old_ids if counts[tid] == 1]

                        if not unique_old:
                            continue

                        new_ids = set(triplet_to_id[triplet] for triplet in combinations(candidate, t))
                        covers = all(tid in new_ids for tid in unique_old)

                        if covers:
                            # Apply the change
                            for triplet in combinations(b, t):
                                counts[triplet_to_id[triplet]] -= 1
                            blocks[i] = list(candidate)
                            for triplet in combinations(candidate, t):
                                counts[triplet_to_id[triplet]] += 1
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

    # Run multiple times with different random seeds
    n_restarts = 12

    for seed in range(n_restarts):
        if time.time() > deadline - 8:
            break

        # Greedy init
        init_blocks = greedy_construct(seed)

        # Optimize
        opt_blocks = optimize(init_blocks, deadline - 8)

        # Verify validity
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in opt_blocks:
            for triplet in combinations(b, t):
                counts[triplet_to_id[triplet]] += 1

        if np.min(counts) > 0:
            if len(opt_blocks) < best_len:
                best_len = len(opt_blocks)
                best_blocks = opt_blocks

    if best_blocks is None:
        # Fallback
        best_blocks = greedy_construct(0)

    return best_blocks

if __name__ == "__main__":
    pass
