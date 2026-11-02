"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import heapq
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # safety margin for 150s limit

    def time_ok():
        return time.time() < deadline

    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute for each point, the list of (t-1)-subsets of other points
    # that with this point form a t-subset, and their indices
    # Actually, let's precompute: for each point p, list of (other_points_tuple, t_sub_idx)
    # where t_sub_idx is the index of the t-subset containing p and other_points_tuple
    point_tsub_contrib = [[] for _ in range(v)]
    for idx, sub in enumerate(all_t_subsets):
        for pos, pt in enumerate(sub):
            others = sub[:pos] + sub[pos+1:]
            point_tsub_contrib[pt].append((others, idx))

    best_blocks = None
    best_count = float('inf')

    n_restarts = 30

    for restart in range(n_restarts):
        if not time_ok():
            break

        random.seed(12345 + restart * 7919)

        covered = [False] * n_t
        covered_count = 0
        blocks = []

        # Point coverage count: how many uncovered t-subsets contain each point
        point_uncovered = [0] * v
        for idx, sub in enumerate(all_t_subsets):
            for pt in sub:
                point_uncovered[pt] += 1

        while covered_count < n_t and time_ok():
            # Select seed: pick an uncovered t-subset weighted by inverse point scarcity
            # (prefer subsets with scarce points)
            uncovered_indices = [i for i in range(n_t) if not covered[i]]
            if not uncovered_indices:
                break

            if len(uncovered_indices) <= 20:
                seed_idx = random.choice(uncovered_indices)
            else:
                # Sample a few and pick best by weight
                n_samples = min(10, len(uncovered_indices))
                best_seed = None
                best_seed_w = -1
                for _ in range(n_samples):
                    idx = random.choice(uncovered_indices)
                    w = 0.0
                    for pt in all_t_subsets[idx]:
                        w += 1.0 / max(1, point_uncovered[pt])
                    if w > best_seed_w:
                        best_seed_w = w
                        best_seed = idx
                seed_idx = best_seed

            seed_subset = all_t_subsets[seed_idx]
            block_set = set(seed_subset)

            # Extend block greedily: at each step, pick the point that maximizes
            # new coverage, with tie-breaking by point scarcity
            while len(block_set) < k and time_ok():
                block_list = sorted(block_set)
                best_x = None
                best_x_score = -1

                for x in all_points:
                    if x in block_set:
                        continue

                    # Compute gain: number of new t-subsets covered by adding x
                    gain = 0
                    # For each (t-1)-subset of current block, check if adding x covers a new t-subset
                    for rest in itertools.combinations(block_list, t - 1):
                        sub = tuple(sorted(rest + (x,)))
                        idx = t_sub_to_idx[sub]
                        if not covered[idx]:
                            gain += 1

                    # Score: gain + small bonus for scarce points
                    scarcity_bonus = 0.0
                    for pt in block_list:
                        scarcity_bonus += 1.0 / max(1, point_uncovered[pt])
                    # Add scarcity of x itself
                    scarcity_bonus += 1.0 / max(1, point_uncovered[x])

                    score = gain + 0.01 * scarcity_bonus
                    if score > best_x_score:
                        best_x_score = score
                        best_x = x

                if best_x is None:
                    break
                block_set.add(best_x)

            block = tuple(sorted(block_set))

            # Calculate gain and apply
            gain = 0
            for sub in itertools.combinations(block, t):
                idx = t_sub_to_idx[sub]
                if not covered[idx]:
                    covered[idx] = True
                    covered_count += 1
                    gain += 1
                    for pt in sub:
                        point_uncovered[pt] -= 1

            blocks.append(block)

        # Deletion phase: remove redundant blocks
        if time_ok() and covered_count == n_t:
            # Build tsub -> list of block indices
            tsub_blocks = defaultdict(list)
            for bi, b in enumerate(blocks):
                for sub in itertools.combinations(b, t):
                    idx = t_sub_to_idx[sub]
                    tsub_blocks[idx].append(bi)

            changed = True
            while changed and time_ok():
                changed = False
                for i in range(len(blocks)):
                    if not time_ok():
                        break
                    removable = True
                    for sub in itertools.combinations(blocks[i], t):
                        idx = t_sub_to_idx[sub]
                        if len(tsub_blocks[idx]) <= 1:
                            removable = False
                            break
                    if removable:
                        for sub in itertools.combinations(blocks[i], t):
                            idx = t_sub_to_idx[sub]
                            tsub_blocks[idx].remove(i)
                        # Remove block i and re-index
                        blocks.pop(i)
                        # Rebuild tsub_blocks with new indices
                        tsub_blocks = defaultdict(list)
                        for bi, b in enumerate(blocks):
                            for sub in itertools.combinations(b, t):
                                idx = t_sub_to_idx[sub]
                                tsub_blocks[idx].append(bi)
                        changed = True
                        break

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = list(blocks)

        # Early stop if we've done very well
        if best_count <= 145:
            break

    if best_blocks is None:
        return []

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
