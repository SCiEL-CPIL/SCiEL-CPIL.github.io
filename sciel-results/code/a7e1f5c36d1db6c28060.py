import itertools
import time
import random
import numpy as np
from collections import defaultdict
from itertools import combinations

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # Precomputation
    all_triplets = list(combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    id_to_triplet = [None] * n_triplets
    for i, trip in enumerate(all_triplets):
        id_to_triplet[i] = trip

    # Precompute mask for a block
    def get_block_mask(block):
        mask = 0
        for triplet in combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
        return mask

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

    # Greedy Construction
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = (1 << n_triplets) - 1
        blocks = []

        while uncovered:
            # Pick a random uncovered triplet
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

            # Find best block containing start_triplet
            best_block = None
            best_ratio = -1
            best_mask = 0
            best_gain = 0

            for block in blocks_containing(start_triplet):
                b_mask = get_block_mask(block)
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

        # Helper to check if a block is redundant
        def is_redundant(block):
            for triplet in combinations(block, t):
                if counts[triplet_to_id[triplet]] == 1:
                    return False
            return True

        # Helper to apply a move
        def apply_move(remove_blocks, add_blocks):
            for b in remove_blocks:
                for triplet in combinations(b, t):
                    counts[triplet_to_id[triplet]] -= 1
            for b in add_blocks:
                for triplet in combinations(b, t):
                    counts[triplet_to_id[triplet]] += 1
            for b in remove_blocks:
                blocks.remove(b)
            for b in add_blocks:
                blocks.append(b)

        no_improve_count = 0
        max_no_improve = 50
        neutral_moves = 0
        max_neutral = 100

        while time.time() < time_limit:
            improved = False
            n = len(blocks)
            if n == 0:
                break

            # 1. Try Remove Redundant
            for i in range(n):
                b = blocks[i]
                if is_redundant(b):
                    for triplet in combinations(b, t):
                        counts[triplet_to_id[triplet]] -= 1
                    blocks.pop(i)
                    improved = True
                    break

            # 2. Try 2-for-1
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

                        found_replacement = False
                        for new_block in blocks_containing(points_needed):
                            if set(new_block) == set(b1) or set(new_block) == set(b2):
                                continue

                            new_ids = set(triplet_to_id[triplet] for triplet in combinations(new_block, t))
                            covers = all(tid in new_ids for tid in needed_triplets)

                            if covers:
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

            # 3. Try 1-for-1 replacement (Shift)
            if not improved:
                n = len(blocks)
                if n == 0:
                    break

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
                        for triplet in combinations(b1, t):
                            counts[triplet_to_id[triplet]] -= 1
                        blocks[i] = new_block
                        for triplet in combinations(new_block, t):
                            counts[triplet_to_id[triplet]] += 1
                        improved = True
                        break

            # 4. Try Swap Move (Neutral or Improving)
            if not improved:
                n = len(blocks)
                if n >= 2:
                    # Pick two random blocks
                    i, j = random.sample(range(n), 2)
                    b1 = list(blocks[i])
                    b2 = list(blocks[j])
                    
                    # Try swapping a point from b1 with a point from b2
                    for p1 in b1:
                        for p2 in b2:
                            if p1 == p2:
                                continue
                            new_b1 = [x if x != p1 else p2 for x in b1]
                            new_b2 = [x if x != p2 else p1 for x in b2]
                            
                            if len(set(new_b1)) < k or len(set(new_b2)) < k:
                                continue
                            
                            new_b1.sort()
                            new_b2.sort()
                            
                            # Check if new blocks are valid and maintain coverage
                            # We need to check if the unique triplets of b1 and b2 are still covered
                            # by the new configuration
                            
                            old_ids_b1 = set(triplet_to_id[triplet] for triplet in combinations(b1, t))
                            old_ids_b2 = set(triplet_to_id[triplet] for triplet in combinations(b2, t))
                            
                            new_ids_b1 = set(triplet_to_id[triplet] for triplet in combinations(new_b1, t))
                            new_ids_b2 = set(triplet_to_id[triplet] for triplet in combinations(new_b2, t))
                            
                            # Check if all triplets that were uniquely covered by b1 or b2 are still covered
                            # A triplet is uniquely covered by b1 if count==1 and it's in b1
                            # After swap, it must be in new_b1, new_b2, or some other block
                            
                            # Simulate counts change
                            # This is expensive to do fully. 
                            # Heuristic: Check if new_b1 covers unique_b1 and new_b2 covers unique_b2
                            # This is sufficient if no cross-coverage was relied upon.
                            
                            unique_b1 = [tid for tid in old_ids_b1 if counts[tid] == 1]
                            unique_b2 = [tid for tid in old_ids_b2 if counts[tid] == 1]
                            
                            if all(tid in new_ids_b1 or tid in new_ids_b2 for tid in unique_b1) and \
                               all(tid in new_ids_b2 or tid in new_ids_b1 for tid in unique_b2):
                                # Apply swap
                                for triplet in combinations(b1, t):
                                    counts[triplet_to_id[triplet]] -= 1
                                for triplet in combinations(b2, t):
                                    counts[triplet_to_id[triplet]] -= 1
                                blocks[i] = new_b1
                                blocks[j] = new_b2
                                for triplet in combinations(new_b1, t):
                                    counts[triplet_to_id[triplet]] += 1
                                for triplet in combinations(new_b2, t):
                                    counts[triplet_to_id[triplet]] += 1
                                improved = True # Count as improvement to reset no_improve counter, even if neutral
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

    n_restarts = 10

    for seed in range(n_restarts):
        if time.time() > deadline - 5:
            break

        init_blocks = greedy_construct(seed)
        opt_blocks = optimize(init_blocks, deadline - 5)

        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in opt_blocks:
            for triplet in combinations(b, t):
                counts[triplet_to_id[triplet]] += 1

        if np.min(counts) > 0:
            if len(opt_blocks) < best_len:
                best_len = len(opt_blocks)
                best_blocks = opt_blocks

    if best_blocks is None:
        best_blocks = greedy_construct(0)

    return best_blocks

if __name__ == "__main__":
    pass
