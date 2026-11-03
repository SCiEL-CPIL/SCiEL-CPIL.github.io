"""Randomized local search for covering design C(24,7,3) with integer-indexed subsets."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0  # leave 5s buffer for safety

    # Precompute all t-subsets and map to integer indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    subset_to_idx = {frozenset(s): i for i, s in enumerate(all_tsubsets)}

    # For each point, precompute which t-subsets contain it
    # point_subs[p] = list of subset indices containing point p
    point_subs = [[] for _ in range(v)]
    for idx, s in enumerate(all_tsubsets):
        for p in s:
            point_subs[p].append(idx)

    # Precompute for each pair of points, the t-subsets containing both
    # This helps with fast gain computation when adding a point to a block
    # pair_subs[(a,b)] = list of subset indices containing both a and b
    pair_subs = {}
    for idx, s in enumerate(all_tsubsets):
        for a, b in itertools.combinations(s, 2):
            key = (min(a,b), max(a,b))
            if key not in pair_subs:
                pair_subs[key] = []
            pair_subs[key].append(idx)

    def block_to_indices(block):
        """Convert a block (tuple of k points) to the set of t-subset indices it covers."""
        covered = set()
        for combo in itertools.combinations(block, t):
            fs = frozenset(combo)
            idx = subset_to_idx[fs]
            covered.add(idx)
        return covered

    def greedy_cover():
        """Fast greedy construction to get a valid starting covering."""
        uncovered = set(range(n_tsubsets))
        blocks = []
        seen_blocks = set()

        while uncovered:
            # Pick a random uncovered t-subset as seed
            seed_idx = rng.choice(list(uncovered))
            seed = all_tsubsets[seed_idx]
            block_list = list(seed)
            block_set = set(block_list)

            # Greedily add points to maximize coverage of uncovered subsets
            while len(block_list) < k:
                best_x = None
                best_gain = -1
                cands = [x for x in range(v) if x not in block_set]
                rng.shuffle(cands)

                for x in cands:
                    gain = 0
                    # For each (t-1)-subset of current block, check if adding x covers an uncovered subset
                    for rest in itertools.combinations(block_list, t - 1):
                        fs = frozenset(list(rest) + [x])
                        idx = subset_to_idx[fs]
                        if idx in uncovered:
                            gain += 1
                    if gain > best_gain or (gain == best_gain and best_x is not None and rng.random() < 0.3):
                        best_gain = gain
                        best_x = x

                if best_x is None:
                    break
                block_list.append(best_x)
                block_set.add(best_x)

            block = tuple(sorted(block_list))
            if block not in seen_blocks:
                seen_blocks.add(block)
                blocks.append(block)
                for combo in itertools.combinations(block, t):
                    fs = frozenset(combo)
                    idx = subset_to_idx[fs]
                    uncovered.discard(idx)

        return blocks

    # Build initial valid covering
    blocks = greedy_cover()

    # Initialize counts and block_covered
    counts = [0] * n_tsubsets
    block_covered = []
    for b in blocks:
        covered = block_to_indices(b)
        block_covered.append(covered)
        for idx in covered:
            counts[idx] += 1

    best_blocks = [list(b) for b in blocks]
    best_count = len(blocks)

    # Local search loop
    no_improve = 0
    max_no_improve = 200

    while time.time() - start_time < time_limit:
        # 1. Try to remove redundant blocks
        removed_any = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            redundant = True
            for idx in block_covered[i]:
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for idx in block_covered[i]:
                    counts[idx] -= 1
                blocks.pop(i)
                block_covered.pop(i)
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
                    no_improve = 0
                removed_any = True
                break

        if removed_any:
            continue

        # 2. Try to swap a block with a better one
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            old_covered = block_covered[i]

            # Identify critical subsets (count == 1) in old_block
            critical = [idx for idx in old_covered if counts[idx] == 1]

            # Collect points that must be in the new block
            points_needed = set()
            for idx in critical:
                points_needed.update(all_tsubsets[idx])

            # If we can't fit all critical points in one block, skip
            if len(points_needed) > k:
                no_improve += 1
                continue

            # Build new block containing points_needed
            new_block_set = set(points_needed)
            cands = [x for x in range(v) if x not in new_block_set]
            rng.shuffle(cands)

            # Greedy fill to maximize coverage of low-count subsets
            for x in cands:
                if len(new_block_set) >= k:
                    break
                gain = 0
                # For each (t-1)-subset of new_block_set, check coverage
                for rest in itertools.combinations(list(new_block_set), t - 1):
                    fs = frozenset(list(rest) + [x])
                    idx = subset_to_idx[fs]
                    if counts[idx] == 0:
                        gain += 10
                    elif counts[idx] == 1:
                        gain += 1
                if gain > 0:
                    new_block_set.add(x)
                elif len(new_block_set) < k:
                    new_block_set.add(x)

            if len(new_block_set) == k:
                new_block = tuple(sorted(new_block_set))

                if new_block == old_block:
                    no_improve += 1
                    continue

                new_covered = block_to_indices(new_block)

                # Check if swap is valid: no subset loses its last cover
                valid = True
                for idx in old_covered:
                    if idx not in new_covered and counts[idx] <= 1:
                        valid = False
                        break

                if valid:
                    # Apply swap
                    for idx in old_covered:
                        counts[idx] -= 1
                    for idx in new_covered:
                        counts[idx] += 1

                    blocks[i] = new_block
                    block_covered[i] = new_covered
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

        # 3. Perturbation if stuck
        if no_improve > max_no_improve:
            # Reset to best and try a different random seed
            blocks = [tuple(b) for b in best_blocks]
            block_covered = [block_to_indices(b) for b in blocks]
            counts = [0] * n_tsubsets
            for covered in block_covered:
                for idx in covered:
                    counts[idx] += 1
            no_improve = 0

    return best_blocks
# EVOLVE-BLOCK-END
