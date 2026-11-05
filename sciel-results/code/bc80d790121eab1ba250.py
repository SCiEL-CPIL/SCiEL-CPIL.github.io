import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145 # 5s safety margin

    # Precomputation
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}

    # Map triplet id to the triplet itself for reverse lookup if needed
    id_to_triplet = [None] * n_triplets
    for i, trip in enumerate(all_triplets):
        id_to_triplet[i] = trip

    # Precompute block masks for all possible blocks containing a given triplet
    # For efficiency, we'll compute masks on demand but cache them

    def get_block_mask(block):
        mask = 0
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
        return mask

    # Precompute masks for all blocks? Too many.
    # Instead, let's precompute for each point, the list of triplets containing it
    point_to_triplets = defaultdict(list)
    for tid, trip in enumerate(all_triplets):
        for p in trip:
            point_to_triplets[p].append(tid)

    def blocks_containing(points):
        points = set(points)
        if len(points) > k:
            return []
        remaining = [p for p in range(v) if p not in points]
        need = k - len(points)
        for comb in itertools.combinations(remaining, need):
            yield tuple(sorted(points.union(comb)))

    # Greedy Construction with ratio-based selection
    def greedy_construct():
        uncovered = (1 << n_triplets) - 1
        blocks = []
        covered_count = 0

        while uncovered:
            # Find an uncovered triplet to guide selection
            lsb = uncovered & -uncovered
            idx = lsb.bit_length() - 1
            start_triplet = id_to_triplet[idx]

            # Find best block containing start_triplet using ratio (gain / cost)
            best_block = None
            best_ratio = -1
            best_gain = 0
            best_mask = 0

            for block in blocks_containing(start_triplet):
                b_mask = get_block_mask(block)
                gain = (b_mask & uncovered).bit_count()
                if gain == 0:
                    continue
                # Ratio: gain per new point not in any existing block
                # Simplified: just use gain
                ratio = gain
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_gain = gain
                    best_block = block
                    best_mask = b_mask

            if best_block is None:
                # Fallback: pick any block covering start_triplet
                for block in blocks_containing(start_triplet):
                    best_block = block
                    best_mask = get_block_mask(block)
                    break

            blocks.append(best_block)
            uncovered &= ~best_mask
            covered_count += best_gain

        return blocks

    # Local Search Optimizer with improved move strategies
    def optimize(blocks):
        counts = np.zeros(n_triplets, dtype=np.int32)

        def recalc_counts(b_list):
            nonlocal counts
            counts[:] = 0
            for b in b_list:
                for triplet in itertools.combinations(b, t):
                    counts[triplet_to_id[triplet]] += 1

        recalc_counts(blocks)

        def is_valid():
            return np.min(counts) > 0

        # Precompute block masks and store them
        block_masks = [get_block_mask(b) for b in blocks]

        def update_block_masks():
            nonlocal block_masks
            block_masks = [get_block_mask(b) for b in blocks]

        # Optimization Loop
        max_iters = 5000
        no_improve_count = 0
        patience = 100

        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break

            improved = False
            n = len(blocks)

            # 1. Try Remove Redundant
            for i in range(n):
                if not improved:
                    b = blocks[i]
                    redundant = True
                    for triplet in itertools.combinations(b, t):
                        if counts[triplet_to_id[triplet]] == 1:
                            redundant = False
                            break
                    if redundant:
                        blocks.pop(i)
                        block_masks.pop(i)
                        for triplet in itertools.combinations(b, t):
                            counts[triplet_to_id[triplet]] -= 1
                        improved = True
                        break

            # 2. Try 2-for-1 swap
            if not improved:
                # Sample random pairs to avoid O(n^2) cost
                n_pairs = min(200, n * (n - 1) // 2)
                for _ in range(n_pairs):
                    if time.time() > deadline - 10:
                        break

                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i == j:
                        continue

                    b1 = blocks[i]
                    b2 = blocks[j]

                    # Compute triplets uniquely covered by b1 or b2
                    ids_b1 = set(triplet_to_id[triplet] for triplet in itertools.combinations(b1, t))
                    ids_b2 = set(triplet_to_id[triplet] for triplet in itertools.combinations(b2, t))

                    all_involved = ids_b1 | ids_b2
                    needed_triplets = [tid for tid in all_involved if counts[tid] == 1]

                    if not needed_triplets:
                        # Both redundant
                        # Remove j first if j > i
                        if j > i:
                            b_remove = blocks.pop(j)
                            block_masks.pop(j)
                            for triplet in itertools.combinations(b_remove, t):
                                counts[triplet_to_id[triplet]] -= 1
                            b_remove = blocks.pop(i)
                            block_masks.pop(i)
                            for triplet in itertools.combinations(b_remove, t):
                                counts[triplet_to_id[triplet]] -= 1
                        else:
                            b_remove = blocks.pop(i)
                            block_masks.pop(i)
                            for triplet in itertools.combinations(b_remove, t):
                                counts[triplet_to_id[triplet]] -= 1
                            b_remove = blocks.pop(j)
                            block_masks.pop(j)
                            for triplet in itertools.combinations(b_remove, t):
                                counts[triplet_to_id[triplet]] -= 1
                        improved = True
                        break

                    # Find points in needed_triplets
                    points_needed = set()
                    for tid in needed_triplets:
                        trip = id_to_triplet[tid]
                        points_needed.update(trip)

                    if len(points_needed) > k:
                        continue

                    # Enumerate blocks containing points_needed
                    found_replacement = False
                    for new_block in blocks_containing(points_needed):
                        new_set = set(new_block)
                        if new_set == set(b1) or new_set == set(b2):
                            continue

                        # Check if new_block covers all needed_triplets
                        new_ids = set(triplet_to_id[triplet] for triplet in itertools.combinations(new_block, t))
                        covers = all(tid in new_ids for tid in needed_triplets)

                        if covers:
                            # Perform swap
                            # Remove both old blocks
                            if j > i:
                                b_remove = blocks.pop(j)
                                block_masks.pop(j)
                                for triplet in itertools.combinations(b_remove, t):
                                    counts[triplet_to_id[triplet]] -= 1
                                b_remove = blocks.pop(i)
                                block_masks.pop(i)
                                for triplet in itertools.combinations(b_remove, t):
                                    counts[triplet_to_id[triplet]] -= 1
                            else:
                                b_remove = blocks.pop(i)
                                block_masks.pop(i)
                                for triplet in itertools.combinations(b_remove, t):
                                    counts[triplet_to_id[triplet]] -= 1
                                b_remove = blocks.pop(j)
                                block_masks.pop(j)
                                for triplet in itertools.combinations(b_remove, t):
                                    counts[triplet_to_id[triplet]] -= 1

                            blocks.append(new_block)
                            block_masks.append(get_block_mask(new_block))
                            for triplet in itertools.combinations(new_block, t):
                                counts[triplet_to_id[triplet]] += 1

                            improved = True
                            found_replacement = True
                            break

                    if improved:
                        break

            # 3. Try 1-for-1 swap
            if not improved:
                n_swaps = min(50, n)
                for _ in range(n_swaps):
                    if time.time() > deadline - 10:
                        break

                    i = random.randint(0, n - 1)
                    b1 = blocks[i]
                    ids_b1 = set(triplet_to_id[triplet] for triplet in itertools.combinations(b1, t))

                    needed_triplets = [tid for tid in ids_b1 if counts[tid] == 1]

                    if not needed_triplets:
                        continue

                    points_needed = set()
                    for tid in needed_triplets:
                        trip = id_to_triplet[tid]
                        points_needed.update(trip)

                    if len(points_needed) > k:
                        continue

                    for new_block in blocks_containing(points_needed):
                        if set(new_block) == set(b1):
                            continue
                        new_ids = set(triplet_to_id[triplet] for triplet in itertools.combinations(new_block, t))
                        covers = all(tid in new_ids for tid in needed_triplets)
                        if covers:
                            b_remove = blocks.pop(i)
                            block_masks.pop(i)
                            for triplet in itertools.combinations(b_remove, t):
                                counts[triplet_to_id[triplet]] -= 1
                            blocks.append(new_block)
                            block_masks.append(get_block_mask(new_block))
                            for triplet in itertools.combinations(new_block, t):
                                counts[triplet_to_id[triplet]] += 1
                            improved = True
                            break

                    if improved:
                        break

            # 4. Kick: Random perturbation to escape local minima
            if not improved:
                no_improve_count += 1
                if no_improve_count >= patience:
                    # Perturb: replace a random block with a random alternative
                    if n > 0:
                        i = random.randint(0, n - 1)
                        b1 = blocks[i]
                        # Pick a random point from b1 and try to find a different block containing it
                        point = random.choice(b1)
                        candidates = list(blocks_containing((point,)))
                        if candidates:
                            new_block = random.choice(candidates)
                            if set(new_block) != set(b1):
                                b_remove = blocks.pop(i)
                                block_masks.pop(i)
                                for triplet in itertools.combinations(b_remove, t):
                                    counts[triplet_to_id[triplet]] -= 1
                                blocks.append(new_block)
                                block_masks.append(get_block_mask(new_block))
                                for triplet in itertools.combinations(new_block, t):
                                    counts[triplet_to_id[triplet]] += 1
                                improved = True

                    no_improve_count = 0

            if improved:
                no_improve_count = 0

        return blocks

    best_blocks = None
    best_len = float('inf')

    # Run multiple times with different random seeds
    for seed in range(30):
        if time.time() > deadline - 10:
            break

        random.seed(seed)
        np.random.seed(seed)

        # Greedy init
        init_blocks = greedy_construct()

        # Optimize
        opt_blocks = optimize(init_blocks)

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
        # Fallback
        best_blocks = greedy_construct()

    return best_blocks

if __name__ == "__main__":
    pass
