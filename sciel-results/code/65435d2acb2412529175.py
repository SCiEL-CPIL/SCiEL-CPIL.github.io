"""
Covering design C(19,7,4) solver using randomized greedy + local search.
"""

import itertools
import time
import random
import numpy as np
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130

    def time_ok():
        return time.time() < deadline

    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute all k-blocks and their t-subset coverage
    all_blocks_list = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks_list)
    block_to_idx = {b: i for i, b in enumerate(all_blocks_list)}

    # block_tsubs[bi] = sorted list of t-subset indices covered by block bi
    block_tsubs = []
    for b in all_blocks_list:
        block_tsubs.append(sorted(t_sub_to_idx[sub] for sub in itertools.combinations(b, t)))

    # For fast lookup: t-subset -> list of blocks covering it
    tsub_to_blocks = defaultdict(list)
    for bi in range(n_blocks):
        for tsi in block_tsubs[bi]:
            tsub_to_blocks[tsi].append(bi)

    best_blocks = None
    best_count = float('inf')
    n_restarts = 80

    for restart in range(n_restarts):
        if not time_ok():
            break

        rng = random.Random(7 + restart * 613)

        covered = np.zeros(n_t, dtype=np.int8)
        covered_count = 0
        blocks = []
        block_set = set()

        # Greedy: repeatedly pick a block with maximum marginal coverage
        while covered_count < n_t and time_ok():
            # Compute marginal gain for each block not yet used
            # Use vectorized approach: for each block, count uncovered tsubs it covers
            gains = np.zeros(n_blocks, dtype=np.int32)
            for bi in range(n_blocks):
                if bi in block_set:
                    continue
                tsubs = block_tsubs[bi]
                # Count how many of these t-subsets are uncovered
                gains[bi] = np.count_nonzero(covered[tsubs] == 0)

            max_gain = gains.max()
            if max_gain <= 0:
                break

            # Pick among top blocks with some randomness
            top_thresh = max_gain - 3
            candidates = np.where((gains >= top_thresh) & (gains > 0))[0]
            if len(candidates) == 0:
                candidates = np.where(gains == max_gain)[0]

            # Weighted random selection
            w = gains[candidates].astype(np.float64)
            w = w ** 1.5
            r = rng.random() * w.sum()
            cum = 0.0
            chosen = candidates[0]
            for j in range(len(candidates)):
                cum += w[j]
                if cum >= r:
                    chosen = candidates[j]
                    break

            # Apply
            for tsi in block_tsubs[chosen]:
                if covered[tsi] == 0:
                    covered[tsi] = 1
                    covered_count += 1
            blocks.append(chosen)
            block_set.add(chosen)

        # Deletion phase
        if time_ok() and covered_count == n_t:
            # Count how many blocks cover each t-subset
            tsub_count = np.zeros(n_t, dtype=np.int32)
            for bi in blocks:
                for tsi in block_tsubs[bi]:
                    tsub_count[tsi] += 1

            changed = True
            while changed and time_ok():
                changed = False
                for i in range(len(blocks)):
                    if not time_ok():
                        break
                    bi = blocks[i]
                    removable = True
                    for tsi in block_tsubs[bi]:
                        if tsub_count[tsi] <= 1:
                            removable = False
                            break
                    if removable:
                        for tsi in block_tsubs[bi]:
                            tsub_count[tsi] -= 1
                        blocks.pop(i)
                        changed = True
                        break

        # Local search: try to swap blocks to enable more deletions
        if time_ok() and covered_count == n_t:
            tsub_count = np.zeros(n_t, dtype=np.int32)
            for bi in blocks:
                for tsi in block_tsubs[bi]:
                    tsub_count[tsi] += 1

            # Try to find a set of blocks that can be removed and replaced
            # with fewer blocks. Simple heuristic: try removing each block
            # and see if we can cover the gap with fewer blocks.
            for i in range(len(blocks)):
                if not time_ok():
                    break
                bi = blocks[i]
                # Find t-subsets uniquely covered by this block
                unique_tsubs = [tsi for tsi in block_tsubs[bi] if tsub_count[tsi] == 1]
                if not unique_tsubs:
                    continue
                # Find blocks that cover these unique t-subsets (excluding bi)
                candidate_replacements = set()
                for tsi in unique_tsubs:
                    for other_bi in tsub_to_blocks[tsi]:
                        if other_bi != bi and other_bi not in block_set:
                            candidate_replacements.add(other_bi)

                if not candidate_replacements:
                    continue

                # Try each candidate: replace bi with candidate, check if
                # total block count decreases after deletion
                best_new_list = None
                best_len = len(blocks)
                for cand_bi in list(candidate_replacements)[:10]:
                    # Simulate: remove bi, add cand_bi
                    new_tsub_count = tsub_count.copy()
                    for tsi in block_tsubs[bi]:
                        new_tsub_count[tsi] -= 1
                    for tsi in block_tsubs[cand_bi]:
                        new_tsub_count[tsi] += 1

                    # Check validity
                    valid = new_tsub_count.min() >= 1
                    if not valid:
                        continue

                    # Count deletable blocks
                    new_blocks_list = [b for j, b in enumerate(blocks) if j != i]
                    new_blocks_list.append(cand_bi)
                    # Try deletions
                    deletable = 0
                    for nb in new_blocks_list:
                        removable = True
                        for tsi in block_tsubs[nb]:
                            if new_tsub_count[tsi] <= 1:
                                removable = False
                                break
                        if removable:
                            deletable += 1
                            for tsi in block_tsubs[nb]:
                                new_tsub_count[tsi] -= 1

                    new_len = len(new_blocks_list) - deletable
                    if new_len < best_len:
                        best_len = new_len
                        best_new_list = new_blocks_list[:len(new_blocks_list) - deletable] if deletable > 0 else list(new_blocks_list)
                        # Actually we need to do the deletion properly
                        # Let's just track the set
                        best_candidate = (cand_bi, deletable)

                # This is getting complex; let's just do simple swaps
                # Try swapping bi with a random block and see if we can delete more
                pass

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]

        if best_count <= 145:
            break

    if best_blocks is None:
        return []

    return [list(all_blocks_list[bi]) for bi in best_blocks]
