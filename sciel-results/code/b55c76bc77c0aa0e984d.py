"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)

    # Map each t-subset to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # For each point, precompute the list of t-subsets containing it
    # This is C(v-1, t-1) per point, which for v=19,t=4 is C(18,3)=816
    point_tsubs = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubs):
        for x in ts:
            point_tsubs[x].append(i)

    # For each pair of points, precompute the t-subsets containing both
    # C(v-2, t-2) per pair, for v=19,t=4 is C(17,2)=136
    pair_tsubs = {}
    for x in range(v):
        for y in range(x + 1, v):
            common = [i for i in point_tsubs[x] if y in all_tsubs[i]]
            # More efficient: intersection of sets
            pair_tsubs[(x, y)] = set(point_tsubs[x]) & set(point_tsubs[y])

    # For each triple of points, precompute the t-subsets containing all three
    # C(v-3, t-3) per triple, for v=19,t=4 is C(16,1)=16
    triple_tsubs = {}
    for x in range(v):
        for y in range(x + 1, v):
            for z in range(y + 1, v):
                triple_tsubs[(x, y, z)] = (set(point_tsubs[x]) & set(point_tsubs[y]) & set(point_tsubs[z]))

    # uncovered: set of indices of uncovered t-subsets
    uncovered = set(range(n_tsubs))

    # Point weights: initially uniform, will adapt
    weights = [1.0] * v

    # Count how many times each point has been "needed" (in uncovered t-subsets)
    # We'll use a soft version: weight[x] = 1 + alpha * (coverage_deficit[x])
    # where coverage_deficit[x] = number of uncovered t-subsets containing x

    blocks = []

    # Greedy with adaptive weights
    while uncovered:
        if time.time() > deadline:
            break

        # Choose a starting t-subset: pick one from uncovered, biased toward
        # t-subsets with high total point-weight
        # For efficiency, pick a random uncovered t-subset but weight the choice
        # by sum of weights of its elements
        # To avoid O(n) scan each time, we'll just pick min(uncovered) for determinism
        # but occasionally randomize for diversity
        if random.random() < 0.1 and len(uncovered) > 10:
            idx = random.choice(list(uncovered))
        else:
            idx = min(uncovered)
        start_tsub = all_tsubs[idx]

        # Build block greedily: start with the t-subset, then add k-t more points
        block_set = set(start_tsub)
        block_list = list(start_tsub)

        # Update weights based on uncovered status for this round
        # Compute deficit: for each point, how many uncovered t-subsets contain it
        # This is expensive to do exactly each iteration, so we'll use a lighter approach

        while len(block_list) < k:
            if time.time() > deadline:
                break

            # Score each candidate point not in block
            best_x = -1
            best_score = -1
            for x in range(v):
                if x in block_set:
                    continue

                # Score: number of new t-subsets covered by adding x to current block
                # A t-subset is covered if all t of its elements are in the block
                # Currently block has m elements. Adding x creates new t-subsets
                # that include x and (t-1) elements from the current block
                # But we only count those that are currently uncovered

                # For t=4, k=7: when block has m elements, adding x covers
                # C(m, t-1) new t-subsets (those with x and t-1 from block)
                # But only the uncovered ones count

                if t == 1:
                    gain = 1 if idx in uncovered else 0  # not applicable for t=1
                elif t == 2:
                    # t-subsets containing x and one element from block
                    # Each such pair (x, b) where b in block
                    gain = sum(1 for b in block_list if tsub_to_idx.get(tuple(sorted((x, b)))) in uncovered)
                elif t == 3:
                    # t-subsets containing x and two elements from block
                    gain = 0
                    for i in range(len(block_list)):
                        for j in range(i + 1, len(block_list)):
                            ts = tuple(sorted((x, block_list[i], block_list[j])))
                            if tsub_to_idx.get(ts) in uncovered:
                                gain += 1
                elif t == 4:
                    # t-subsets containing x and three elements from block
                    gain = 0
                    for i in range(len(block_list)):
                        for j in range(i + 1, len(block_list)):
                            for l in range(j + 1, len(block_list)):
                                ts = tuple(sorted((x, block_list[i], block_list[j], block_list[l])))
                                idx_ts = tsub_to_idx.get(ts)
                                if idx_ts is not None and idx_ts in uncovered:
                                    gain += 1
                else:
                    # General case
                    gain = 0
                    for rest in itertools.combinations(block_list, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx_ts = tsub_to_idx.get(ts)
                        if idx_ts is not None and idx_ts in uncovered:
                            gain += 1

                # Add a small weight-based bias
                score = gain + 0.01 * weights[x]

                if score > best_score:
                    best_score = score
                    best_x = x

            if best_x == -1:
                break

            block_list.append(best_x)
            block_set.add(best_x)

        if len(block_list) < k:
            # Fill remaining with arbitrary points
            for x in range(v):
                if len(block_list) >= k:
                    break
                if x not in block_set:
                    block_list.append(x)
                    block_set.add(x)

        block_list.sort()
        blocks.append(block_list)

        # Update uncovered
        for ts in itertools.combinations(block_list, t):
            idx_ts = tsub_to_idx.get(ts)
            if idx_ts is not None:
                uncovered.discard(idx_ts)

        # Update weights: increase weight for points in frequently uncovered t-subsets
        # Light update: for each uncovered t-subset, slightly boost its elements
        # But this is O(|uncovered| * t) which could be large
        # Instead, do a periodic full update
        if len(blocks) % 10 == 0:
            # Recompute deficits
            deficit = [0] * v
            for idx_u in uncovered:
                for x in all_tsubs[idx_u]:
                    deficit[x] += 1
            # Normalize and update weights
            max_def = max(deficit) if deficit else 1
            for x in range(v):
                weights[x] = 1.0 + 0.5 * (deficit[x] / max_def if max_def > 0 else 0.0)

    # Final safety: if we didn't cover everything, try to fill gaps quickly
    if uncovered and time.time() < deadline - 5:
        # Simple greedy fill for remaining
        while uncovered and time.time() < deadline - 2:
            idx = min(uncovered)
            ts = all_tsubs[idx]
            block_list = list(ts)
            while len(block_list) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_list:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block_list, t - 1):
                        tss = tuple(sorted(rest + (x,)))
                        idx_tss = tsub_to_idx.get(tss)
                        if idx_tss is not None and idx_tss in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    for x in range(v):
                        if x not in block_list:
                            best_x = x
                            break
                block_list.append(best_x)
            block_list.sort()
            blocks.append(block_list)
            for tss in itertools.combinations(block_list, t):
                idx_tss = tsub_to_idx.get(tss)
                if idx_tss is not None:
                    uncovered.discard(idx_tss)

    return blocks
# EVOLVE-BLOCK-END
