"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import heapq
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s limit

    def time_ok():
        return time.time() < deadline

    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute: for each point, which t-subsets contain it
    point_tsubs = [[] for _ in range(v)]
    for i, sub in enumerate(all_t_subsets):
        for pt in sub:
            point_tsubs[pt].append(i)

    best_blocks = None
    best_count = float('inf')

    n_restarts = 40

    for restart in range(n_restarts):
        if not time_ok():
            break

        random.seed(12345 + restart * 7919)

        covered = [False] * n_t
        covered_count = 0
        blocks = []

        # Dynamic greedy: pick the uncovered t-subset with highest "scarcity"
        # (fewest points that could still be added to cover many uncovered subsets)
        while covered_count < n_t and time_ok():
            # Find uncovered t-subsets and score them by scarcity
            # A t-subset is "scarce" if its points appear in few uncovered subsets
            # (meaning adding those points to a block covers relatively unique ground)
            uncovered_indices = [i for i in range(n_t) if not covered[i]]
            if not uncovered_indices:
                break

            # Score each uncovered t-subset: sum of point "rarities"
            # Point rarity = 1 / (number of uncovered subsets containing it)
            # Higher score = more constrained = better seed
            point_uncovered_count = [0] * v
            for idx in uncovered_indices:
                for pt in all_t_subsets[idx]:
                    point_uncovered_count[pt] += 1

            # Pick seed: try a few candidates, pick the one that gives best block
            # Use scarcity as a guide but allow randomness
            if len(uncovered_indices) <= 50:
                candidates = uncovered_indices
            else:
                # Weighted random sampling based on scarcity
                weights = []
                for idx in uncovered_indices:
                    w = sum(1.0 / max(1, point_uncovered_count[pt]) for pt in all_t_subsets[idx])
                    weights.append(w)
                total_w = sum(weights)
                if total_w == 0:
                    candidates = [random.choice(uncovered_indices)]
                else:
                    # Sample up to 10 candidates
                    n_samples = min(10, len(uncovered_indices))
                    candidates = []
                    for _ in range(n_samples):
                        r = random.random() * total_w
                        cum = 0
                        for i, w in enumerate(weights):
                            cum += w
                            if cum >= r:
                                candidates.append(uncovered_indices[i])
                                break
                    if not candidates:
                        candidates = [random.choice(uncovered_indices)]

            best_block = None
            best_gain = -1

            for seed_idx in candidates:
                if not time_ok():
                    break
                seed_subset = all_t_subsets[seed_idx]
                block_set = set(seed_subset)

                # Extend block greedily with dynamic gain
                while len(block_set) < k and time_ok():
                    best_x = None
                    best_x_gain = -1
                    best_x_tie = -1.0

                    # Compute gain for each candidate point
                    # Gain = number of uncovered t-subsets that would be covered
                    # by adding x to the current block
                    block_tuple = tuple(sorted(block_set))
                    candidates_pts = [x for x in all_points if x not in block_set]
                    random.shuffle(candidates_pts)

                    for x in candidates_pts:
                        gain = 0
                        for rest in itertools.combinations(block_tuple, t - 1):
                            sub = tuple(sorted(rest + (x,)))
                            idx = t_sub_to_idx[sub]
                            if not covered[idx]:
                                gain += 1
                        # Add small random tiebreaker
                        tie = random.random() * 0.01
                        if gain + tie > best_x_gain:
                            best_x_gain = gain + tie
                            best_x = x
                            best_x_tie = tie

                    if best_x is None:
                        break
                    block_set.add(best_x)

                if len(block_set) < k:
                    remaining = [x for x in all_points if x not in block_set]
                    random.shuffle(remaining)
                    for x in remaining[:k - len(block_set)]:
                        block_set.add(x)

                block = tuple(sorted(block_set))

                # Calculate total gain of this block
                gain = 0
                for sub in itertools.combinations(block, t):
                    idx = t_sub_to_idx[sub]
                    if not covered[idx]:
                        gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_block = block

            if best_block is None or best_gain <= 0:
                # Fill with random block if no useful block found
                block_set = set(random.sample(all_points, k))
                best_block = tuple(sorted(block_set))
                best_gain = sum(1 for sub in itertools.combinations(best_block, t)
                                if not covered[t_sub_to_idx[sub]])

            if best_gain <= 0:
                break

            # Apply the block
            for sub in itertools.combinations(best_block, t):
                idx = t_sub_to_idx[sub]
                if not covered[idx]:
                    covered[idx] = True
                    covered_count += 1
            blocks.append(best_block)

        # Deletion phase: remove redundant blocks
        if time_ok() and covered_count == n_t:
            tsub_block_count = [0] * n_t
            for b in blocks:
                for sub in itertools.combinations(b, t):
                    idx = t_sub_to_idx[sub]
                    tsub_block_count[idx] += 1

            changed = True
            while changed and time_ok():
                changed = False
                for i in range(len(blocks)):
                    if not time_ok():
                        break
                    removable = True
                    for sub in itertools.combinations(blocks[i], t):
                        idx = t_sub_to_idx[sub]
                        if tsub_block_count[idx] <= 1:
                            removable = False
                            break
                    if removable:
                        for sub in itertools.combinations(blocks[i], t):
                            idx = t_sub_to_idx[sub]
                            tsub_block_count[idx] -= 1
                        blocks.pop(i)
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
