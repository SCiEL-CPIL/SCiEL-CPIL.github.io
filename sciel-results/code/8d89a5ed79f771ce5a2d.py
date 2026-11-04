"""Combined program for C(24,7,3) covering design: strong-greedy with bitsets + multi-restart + refinement."""
import itertools
import time
import random
from collections import defaultdict
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin before 120s limit

    # Precompute all t-subsets and index them
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubsets)}

    # For each point, precompute which t-subsets contain it
    point_to_tsubs = defaultdict(list)
    for i, ts in enumerate(all_tsubsets):
        for p in ts:
            point_to_tsubs[p].append(i)

    # For each pair of points, precompute which t-subsets contain both
    # This is useful for fast gain calculation
    pair_to_tsubs = defaultdict(list)
    if t >= 2:
        for i, ts in enumerate(all_tsubsets):
            for a, b in itertools.combinations(ts, 2):
                pair_to_tsubs[(a, b)].append(i)

    best_blocks = None
    best_len = float('inf')

    # Try multiple restarts with different random seeds
    trial = 0
    while time.time() < deadline - 5:  # Leave 5s for final refinement
        trial += 1
        blocks = _strong_greedy_cover(v, k, t, all_tsubsets, tsub_index, point_to_tsubs, deadline, trial)
        if blocks is None:
            continue

        # Phase 1: Remove redundant blocks
        blocks = _remove_redundant(blocks, v, k, t, deadline)

        # Phase 2: Local search refinement
        blocks = _local_search(blocks, v, k, t, all_tsubsets, tsub_index, point_to_tsubs, deadline, trial)

        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks

        # Early termination if we hit the known best
        if best_len <= 76:
            break

    if best_blocks is None:
        # Fallback: return empty or a trivial cover
        return []

    return [list(b) for b in best_blocks]


def _strong_greedy_cover(v, k, t, all_tsubsets, tsub_index, point_to_tsubs, deadline, seed_val):
    """
    Strong greedy: at each step, pick the k-block that covers the most uncovered t-subsets.
    Uses bitsets for fast gain computation.
    """
    rng = random.Random(seed_val)
    n_tsub = len(all_tsubsets)
    
    # Use a numpy bitset to track uncovered t-subsets
    uncovered_bits = np.ones(n_tsub, dtype=bool)
    n_uncovered = n_tsub

    blocks = []
    points = list(range(v))

    while n_uncovered > 0:
        if time.time() > deadline:
            return None

        # Find the k-block with maximum gain
        # We'll use a heuristic: build the block greedily but with a better scoring
        # Actually, for strong greedy, we want to pick the best k-subset overall.
        # That's expensive (C(24,7) ~ 346k blocks). Instead, we do a two-phase:
        # Phase 1: pick a seed point that appears in many uncovered t-subsets
        # Phase 2: greedily add points to maximize gain

        # Count how many uncovered t-subsets contain each point
        point_scores = np.zeros(v, dtype=np.int32)
        for p in range(v):
            for idx in point_to_tsubs[p]:
                if uncovered_bits[idx]:
                    point_scores[p] += 1

        # Pick the point with highest score as seed
        best_seed = int(np.argmax(point_scores))
        if point_scores[best_seed] == 0:
            # No uncovered t-subsets contain any point? Shouldn't happen if n_uncovered > 0
            break

        block_set = {best_seed}
        block = [best_seed]

        # Greedily add k-1 more points
        while len(block) < k:
            if time.time() > deadline:
                return None

            # For each candidate x not in block, compute gain
            best_x = None
            best_gain = -1
            best_tiebreak = []

            # Pre-compute (t-1) subsets of current block for fast gain calculation
            t_minus_1_subsets = list(itertools.combinations(sorted(block_set), t - 1)) if t >= 2 else [()]

            for x in points:
                if x in block_set:
                    continue

                gain = 0
                for sub in t_minus_1_subsets:
                    # sub is a tuple of t-1 elements from current block
                    # Check if {sub...} U {x} is a t-subset and is uncovered
                    new_subset = tuple(sorted(sub + (x,)))
                    if new_subset in tsub_index:
                        idx = tsub_index[new_subset]
                        if uncovered_bits[idx]:
                            gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    best_tiebreak = [x]
                elif gain == best_gain and gain > 0:
                    best_tiebreak.append(x)

            if best_x is None:
                available = [x for x in points if x not in block_set]
                if not available:
                    return None
                best_x = rng.choice(available)
                best_gain = 0

            block.append(best_x)
            block_set.add(best_x)

        block.sort()
        blocks.append(tuple(block))

        # Update uncovered bits
        for ts in itertools.combinations(block, t):
            if ts in tsub_index:
                idx = tsub_index[ts]
                if uncovered_bits[idx]:
                    uncovered_bits[idx] = False
                    n_uncovered -= 1

    return blocks


def _remove_redundant(blocks, v, k, t, deadline):
    """
    Iteratively removes blocks that are redundant (every t-subset in them is covered by others).
    """
    if not blocks:
        return blocks

    # Count how many times each t-subset is covered
    cover_count = defaultdict(int)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            cover_count[sub] += 1

    improved = list(blocks)
    changed = True

    while changed and time.time() < deadline - 1:
        changed = False
        for i in range(len(improved) - 1, -1, -1):
            if time.time() > deadline - 1:
                break
            block = improved[i]
            is_redundant = True
            for sub in itertools.combinations(block, t):
                if cover_count[sub] <= 1:
                    is_redundant = False
                    break

            if is_redundant:
                # Remove block
                for sub in itertools.combinations(block, t):
                    cover_count[sub] -= 1
                improved.pop(i)
                changed = True

    return improved


def _local_search(blocks, v, k, t, all_tsubsets, tsub_index, point_to_tsubs, deadline, seed_val):
    """
    Local search: try to replace blocks to enable removal of other blocks.
    Strategy: for each block, try replacing it with a different block that covers
    all its t-subsets and potentially helps make another block redundant.
    """
    if not blocks or time.time() > deadline - 2:
        return blocks

    rng = random.Random(seed_val + 1000)
    n_blocks = len(blocks)

    # Build coverage counts
    cover_count = defaultdict(int)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            cover_count[sub] += 1

    improved = list(blocks)
    any_change = False

    # Try to find a block that can be removed by replacing another block
    # We'll do a few passes
    for _ in range(3):
        if time.time() > deadline - 2:
            break

        # Find blocks that are "critical" (have at least one t-subset with count 1)
        critical_blocks = []
        for i, block in enumerate(improved):
            is_critical = False
            for sub in itertools.combinations(block, t):
                if cover_count[sub] == 1:
                    is_critical = True
                    break
            if is_critical:
                critical_blocks.append(i)

        if not critical_blocks:
            break

        # Try to replace a non-critical block to cover some critical t-subsets
        non_critical = [i for i in range(len(improved)) if i not in critical_blocks]
        if not non_critical:
            break

        # For each non-critical block, try to replace it with a better one
        for idx in non_critical:
            if time.time() > deadline - 2:
                break

            old_block = improved[idx]
            old_tsubs = set(itertools.combinations(old_block, t))

            # Find t-subsets in old_block that are covered only by this block
            must_cover = set()
            for sub in old_tsubs:
                if cover_count[sub] == 1:
                    must_cover.add(sub)

            if not must_cover:
                # This block is fully redundant already? Should have been removed.
                continue

            # Try to find a new block that covers all must_cover t-subsets
            # and possibly some critical t-subsets from other blocks
            best_new_block = None
            best_score = -1

            # Sample some candidate blocks
            for _ in range(20):
                # Randomly sample a k-block that contains at least one t-subset from must_cover
                if not must_cover:
                    break
                sample_tsub = rng.choice(list(must_cover))
                sample_set = set(sample_tsub)
                
                # Build a block around this t-subset
                candidates = list(range(v))
                for p in sample_set:
                    candidates.remove(p)
                
                remaining = k - len(sample_set)
                chosen = rng.sample(candidates, min(remaining, len(candidates)))
                new_block_set = sample_set | set(chosen)
                if len(new_block_set) < k:
                    # Fill with random points
                    still_missing = k - len(new_block_set)
                    for p in range(v):
                        if p not in new_block_set and still_missing > 0:
                            new_block_set.add(p)
                            still_missing -= 1
                
                new_block = tuple(sorted(new_block_set))
                new_tsubs = set(itertools.combinations(new_block, t))

                # Check if new_block covers all must_cover
                if not must_cover.issubset(new_tsubs):
                    continue

                # Score: how many critical t-subsets from other blocks does it cover?
                score = 0
                for sub in new_tsubs:
                    if sub not in old_tsubs and cover_count[sub] == 1:
                        score += 1

                if score > best_score:
                    best_score = score
                    best_new_block = new_block

            if best_new_block is not None and best_score > 0:
                # Replace old_block with best_new_block
                # Update cover counts
                for sub in old_tsubs:
                    cover_count[sub] -= 1
                for sub in itertools.combinations(best_new_block, t):
                    cover_count[sub] += 1

                improved[idx] = best_new_block
                any_change = True

                # Try to remove redundant blocks again
                improved = _remove_redundant(improved, v, k, t, deadline)
                # Rebuild cover_count
                cover_count = defaultdict(int)
                for block in improved:
                    for sub in itertools.combinations(block, t):
                        cover_count[sub] += 1

                if time.time() > deadline - 2:
                    break

    return improved
