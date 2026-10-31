"""Rewritten program for covering design C(v,k,t) with efficient greedy + local search."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 140.0

    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    # point_to_tsubsets: for each point, list of t-subset indices containing it
    point_to_tsubsets = defaultdict(list)
    for i, ts in enumerate(all_t_subsets):
        for x in ts:
            point_to_tsubsets[x].append(i)

    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered = set(range(n_t))
        blocks = []

        while uncovered:
            if time.time() > deadline - 2.0:
                return blocks, False

            # Pick a random uncovered t-subset
            start_idx = rng.choice(list(uncovered))
            start_ts = all_t_subsets[start_idx]
            block = list(start_ts)
            block_set = set(block)

            while len(block) < k:
                if time.time() > deadline - 2.0:
                    return blocks, False

                best_x = -1
                best_gain = -1
                candidates = [x for x in range(v) if x not in block_set]

                for x in candidates:
                    # Count uncovered t-subsets containing x where ts\{x} ⊆ block
                    gain = 0
                    for ts_idx in point_to_tsubsets[x]:
                        if ts_idx not in uncovered:
                            continue
                        ts = all_t_subsets[ts_idx]
                        # Check ts \ {x} ⊆ block
                        ok = True
                        for y in ts:
                            if y == x:
                                continue
                            if y not in block_set:
                                ok = False
                                break
                        if ok:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x

                if best_x == -1:
                    best_x = rng.choice(candidates)

                block.append(best_x)
                block_set.add(best_x)

            block.sort()
            blocks.append(block)

            # Update uncovered
            for ts_idx in list(uncovered):
                ts = all_t_subsets[ts_idx]
                if all(x in block_set for x in ts):
                    uncovered.discard(ts_idx)

        return blocks, True

    # Initial deterministic greedy
    blocks, complete = greedy_cover(0)
    best_blocks = blocks
    best_count = len(blocks)

    # Randomized restarts
    for seed in range(1, 200):
        if time.time() > deadline - 10.0:
            break
        b2, c2 = greedy_cover(seed)
        if c2 and len(b2) < best_count:
            best_blocks = b2
            best_count = len(b2)

    # Remove redundant blocks
    if best_count > 0 and time.time() < deadline - 5.0:
        n = len(best_blocks)
        cover_count = [0] * n_t
        block_tsubsets = []
        for b in best_blocks:
            bset = set(b)
            covered = []
            for ts_idx, ts in enumerate(all_t_subsets):
                if all(x in bset for x in ts):
                    covered.append(ts_idx)
                    cover_count[ts_idx] += 1
            block_tsubsets.append(covered)

        changed = True
        while changed and time.time() < deadline - 3.0:
            changed = False
            for i in range(len(best_blocks)):
                redundant = True
                for ts_idx in block_tsubsets[i]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for ts_idx in block_tsubsets[i]:
                        cover_count[ts_idx] -= 1
                    best_blocks.pop(i)
                    block_tsubsets.pop(i)
                    changed = True
                    break

        best_count = len(best_blocks)

    return best_blocks
# EVOLVE-BLOCK-END
