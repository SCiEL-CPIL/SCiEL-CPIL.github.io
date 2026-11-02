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

    # Precompute: for each k-block, the list of t-subset indices it covers
    # This is expensive to precompute all C(v,k) blocks, so we compute on the fly
    # but cache results.

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

        # Generate candidate blocks using a dynamic greedy that builds blocks
        # by starting from uncovered t-subsets and extending optimally.
        # Use a min-heap on (negative_gain, block) to always pick best current block.

        # Strategy: iteratively build blocks. For each new block, try multiple
        # seeds (uncovered t-subsets) and pick the extension that maximizes
        # uncovered coverage. Then apply and repeat.

        while covered_count < n_t and time_ok():
            # Find all uncovered t-subset indices
            uncovered_indices = [i for i in range(n_t) if not covered[i]]
            if not uncovered_indices:
                break

            # Compute point scarcity: how many uncovered t-subsets contain each point
            point_uncovered_count = [0] * v
            for idx in uncovered_indices:
                for pt in all_t_subsets[idx]:
                    point_uncovered_count[pt] += 1

            # Select seed candidates: prioritize t-subsets whose points are scarce
            # (appear in few uncovered subsets), as covering them forces useful blocks
            if len(uncovered_indices) <= 30:
                seed_candidates = uncovered_indices
            else:
                # Weight by inverse scarcity sum
                weights = []
                for idx in uncovered_indices:
                    w = 0.0
                    for pt in all_t_subsets[idx]:
                        w += 1.0 / max(1, point_uncovered_count[pt])
                    weights.append(w)
                total_w = sum(weights)
                if total_w == 0:
                    seed_candidates = [random.choice(uncovered_indices)]
                else:
                    n_samples = min(15, len(uncovered_indices))
                    seed_candidates = []
                    for _ in range(n_samples):
                        r = random.random() * total_w
                        cum = 0.0
                        for i, w in enumerate(weights):
                            cum += w
                            if cum >= r:
                                seed_candidates.append(uncovered_indices[i])
                                break
                    if not seed_candidates:
                        seed_candidates = [random.choice(uncovered_indices)]

            # For each seed, build a block and compute its gain
            best_block = None
            best_gain = -1
            best_block_tsubs = None  # list of t-subset indices this block covers

            for seed_idx in seed_candidates:
                if not time_ok():
                    break
                seed_subset = all_t_subsets[seed_idx]
                block_set = set(seed_subset)

                # Extend block greedily
                while len(block_set) < k and time_ok():
                    block_tuple = tuple(sorted(block_set))
                    candidates_pts = [x for x in all_points if x not in block_set]
                    random.shuffle(candidates_pts)

                    best_x = None
                    best_x_gain = -1

                    for x in candidates_pts:
                        gain = 0
                        for rest in itertools.combinations(block_tuple, t - 1):
                            sub = tuple(sorted(rest + (x,)))
                            idx = t_sub_to_idx[sub]
                            if not covered[idx]:
                                gain += 1
                        # Small random tiebreaker for diversity
                        tie = random.random() * 0.005
                        if gain + tie > best_x_gain:
                            best_x_gain = gain + tie
                            best_x = x

                    if best_x is None:
                        break
                    block_set.add(best_x)

                # Fill remaining spots randomly
                if len(block_set) < k:
                    remaining = [x for x in all_points if x not in block_set]
                    random.shuffle(remaining)
                    for x in remaining[:k - len(block_set)]:
                        block_set.add(x)

                block = tuple(sorted(block_set))

                # Calculate total gain and collect covered t-subset indices
                gain = 0
                block_tsubs = []
                for sub in itertools.combinations(block, t):
                    idx = t_sub_to_idx[sub]
                    block_tsubs.append(idx)
                    if not covered[idx]:
                        gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    best_block_tsubs = block_tsubs

            if best_block is None or best_gain <= 0:
                # Fallback: random block
                block_set = set(random.sample(all_points, k))
                best_block = tuple(sorted(block_set))
                best_block_tsubs = []
                best_gain = 0
                for sub in itertools.combinations(best_block, t):
                    idx = t_sub_to_idx[sub]
                    best_block_tsubs.append(idx)
                    if not covered[idx]:
                        best_gain += 1

            if best_gain <= 0:
                break

            # Apply the block
            for idx in best_block_tsubs:
                if not covered[idx]:
                    covered[idx] = True
                    covered_count += 1
            blocks.append(best_block)

        # Deletion phase: remove redundant blocks (iterate until stable)
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

            # Second pass: try replacing pairs of blocks with a single better block
            if time_ok():
                for i in range(len(blocks)):
                    if not time_ok():
                        break
                    for j in range(i + 1, len(blocks)):
                        if not time_ok():
                            break
                        # Check if blocks[i] and blocks[j] can be merged/replaced
                        # by finding a single block that covers all t-subsets
                        # uniquely covered by either block[i] or block[j]
                        unique_to_i = []
                        unique_to_j = []
                        for sub in itertools.combinations(blocks[i], t):
                            idx = t_sub_to_idx[sub]
                            if tsub_block_count[idx] == 1:
                                unique_to_i.append(idx)
                        for sub in itertools.combinations(blocks[j], t):
                            idx = t_sub_to_idx[sub]
                            if tsub_block_count[idx] == 1:
                                unique_to_j.append(idx)

                        if not unique_to_i and not unique_to_j:
                            # Both blocks are fully redundant together
                            for sub in itertools.combinations(blocks[i], t):
                                idx = t_sub_to_idx[sub]
                                tsub_block_count[idx] -= 1
                            for sub in itertools.combinations(blocks[j], t):
                                idx = t_sub_to_idx[sub]
                                tsub_block_count[idx] -= 1
                            blocks.pop(j)
                            blocks.pop(i)
                            changed = True
                            break

                        if unique_to_i or unique_to_j:
                            # Try to find a replacement block
                            all_unique = set(unique_to_i + unique_to_j)
                            # Collect points that appear in these unique t-subsets
                            required_points = set()
                            for idx in all_unique:
                                for pt in all_t_subsets[idx]:
                                    required_points.add(pt)

                            if len(required_points) <= k:
                                # Can form a block from required points
                                rep_block = tuple(sorted(required_points))
                                if len(rep_block) < k:
                                    extra = [x for x in all_points if x not in required_points]
                                    random.shuffle(extra)
                                    rep_block = tuple(sorted(set(list(rep_block) + extra[:k - len(rep_block)])))

                                # Verify this block covers all unique t-subsets
                                valid = True
                                for idx in all_unique:
                                    sub = all_t_subsets[idx]
                                    if not all(pt in rep_block for pt in sub):
                                        valid = False
                                        break

                                if valid:
                                    # Remove both old blocks and add replacement
                                    for sub in itertools.combinations(blocks[i], t):
                                        idx = t_sub_to_idx[sub]
                                        tsub_block_count[idx] -= 1
                                    for sub in itertools.combinations(blocks[j], t):
                                        idx = t_sub_to_idx[sub]
                                        tsub_block_count[idx] -= 1
                                    for sub in itertools.combinations(rep_block, t):
                                        idx = t_sub_to_idx[sub]
                                        tsub_block_count[idx] += 1
                                    blocks.pop(j)
                                    blocks.pop(i)
                                    blocks.append(rep_block)
                                    changed = True
                                    break

                if changed:
                    continue

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
