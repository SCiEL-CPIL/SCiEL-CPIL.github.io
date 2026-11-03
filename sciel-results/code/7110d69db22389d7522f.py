import itertools
import time
import random
import numpy as np
from collections import defaultdict
import heapq

def construct(v, k, t):
    deadline = time.time() + 115.0

    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)

    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # block_tsubs[i] = list of t-subset indices covered by block i
    block_tsubs = [None] * n_blocks
    # tsub_blocks[j] = list of block indices covering t-subset j
    tsub_blocks = [[] for _ in range(n_tsubsets)]

    for bi, block in enumerate(all_blocks):
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs[bi] = tsubs
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)

    # Precompute set versions of block_tsubs for faster intersection in local search
    block_tsub_sets = [set(tsubs) for tsubs in block_tsubs]

    # Precompute tsub_blocks as sets for faster intersection
    tsub_block_sets = [set(lst) for lst in tsub_blocks]

    def greedy_cover(seed):
        """Classic greedy set cover using a max-heap."""
        rng = random.Random(seed)
        uncovered = [True] * n_tsubsets
        gain = [0] * n_blocks
        for bi in range(n_blocks):
            gain[bi] = len(block_tsubs[bi])

        heap = [(-gain[bi], bi) for bi in range(n_blocks)]
        heapq.heapify(heap)

        selected = []

        while True:
            if time.time() > deadline - 5:
                break

            best_bi = -1
            max_gain = 0
            while heap:
                neg_g, bi = heapq.heappop(heap)
                g = -neg_g
                if g == gain[bi] and g > 0:
                    best_bi = bi
                    max_gain = g
                    break
            else:
                break

            if max_gain <= 0:
                break

            selected.append(best_bi)

            for ts_idx in block_tsubs[best_bi]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    for bj in tsub_blocks[ts_idx]:
                        gain[bj] -= 1
                        heapq.heappush(heap, (-gain[bj], bj))

            gain[best_bi] = -1
            heapq.heappush(heap, (-gain[best_bi], best_bi))

        return selected

    def primal_dual_cover(seed):
        """
        Primal-dual approximation for minimum weight set cover.
        Dual variables y[ts] for each t-subset.
        At each step, raise y for all uncovered t-subsets uniformly until
        some block's accumulated dual weight equals its cost (1 here).
        Select that block, mark its t-subsets covered.
        This exploits submodularity: blocks covering many uncovered
        t-subsets get selected earlier.
        """
        rng = random.Random(seed)
        uncovered = [True] * n_tsubsets
        y = [0.0] * n_tsubsets  # dual variables
        block_cost = [1.0] * n_blocks  # all blocks cost 1
        selected = []
        n_uncovered = n_tsubsets

        # For efficiency, maintain for each block the sum of y over its uncovered t-subsets
        # marginal[bi] = sum of y[ts] for ts in block_tsubs[bi] where uncovered[ts]
        marginal = [0.0] * n_blocks

        # Initially y=0, so marginal=0. We raise y uniformly.
        # The rate at which marginal increases for block bi is:
        #   number of uncovered t-subsets in block bi
        # When marginal[bi] reaches block_cost[bi], we select bi.

        while n_uncovered > 0 and time.time() < deadline - 5:
            # Compute current marginal gains (number of uncovered t-subsets per block)
            # and find the block that will hit cost=1 first.
            # marginal[bi] + rate[bi] * dt = 1
            # dt = (1 - marginal[bi]) / rate[bi]
            # We want the smallest dt.

            best_dt = float('inf')
            best_bi = -1

            for bi in range(n_blocks):
                if marginal[bi] >= block_cost[bi]:
                    best_dt = 0.0
                    best_bi = bi
                    break
                rate = 0
                for ts_idx in block_tsubs[bi]:
                    if uncovered[ts_idx]:
                        rate += 1
                if rate == 0:
                    continue
                dt = (block_cost[bi] - marginal[bi]) / rate
                if dt < best_dt:
                    best_dt = dt
                    best_bi = bi

            if best_bi == -1:
                break

            # Raise y by best_dt for all uncovered t-subsets
            for ts_idx in range(n_tsubsets):
                if uncovered[ts_idx]:
                    y[ts_idx] += best_dt

            # Update marginal for all blocks
            for bi in range(n_blocks):
                if marginal[bi] >= block_cost[bi]:
                    continue
                rate = 0
                for ts_idx in block_tsubs[bi]:
                    if uncovered[ts_idx]:
                        rate += 1
                marginal[bi] += rate * best_dt

            # Select best_bi
            selected.append(best_bi)

            # Mark its t-subsets as covered
            for ts_idx in block_tsubs[best_bi]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    n_uncovered -= 1

            # Update marginal: blocks that shared ts_idx with best_bi lose that contribution
            for ts_idx in block_tsubs[best_bi]:
                if not uncovered[ts_idx]:  # was just covered
                    for bj in tsub_blocks[ts_idx]:
                        if bj != best_bi:
                            # This ts_idx no longer contributes to bj's rate
                            # We need to reduce marginal[bj] by y[ts_idx]
                            # But actually marginal tracks sum of y, not rate.
                            # Let me re-think: marginal[bi] = sum of y[ts] for uncovered ts in bi
                            # When ts becomes covered, we subtract y[ts] from marginal[bj]
                            marginal[bj] -= y[ts_idx]

            # Mark best_bi as used
            marginal[best_bi] = block_cost[best_bi]

        return selected

    def local_search(blocks, seed):
        rng = random.Random(seed)
        block_set = set(blocks)
        cover_count = [0] * n_tsubsets

        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1

        improved = True
        while improved and time.time() < deadline - 2:
            improved = False

            # Phase 1: Remove redundant blocks
            to_remove = []
            for bi in block_set:
                redundant = True
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(bi)

            if to_remove:
                for bi in to_remove:
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                improved = True
                continue

            # Phase 2: Try swaps
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            for bi in current_blocks:
                if time.time() > deadline - 2:
                    break
                if bi not in block_set:
                    continue

                unique_tsubs = []
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)

                if not unique_tsubs:
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break

                min_ts = min(unique_tsubs, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_block_sets[min_ts])

                for ts_idx in unique_tsubs[1:]:
                    if not candidates:
                        break
                    candidates &= tsub_block_sets[ts_idx]

                candidates.discard(bi)

                if not candidates:
                    continue

                cand_list = list(candidates)
                rng.shuffle(cand_list)

                swapped = False
                for bj in cand_list[:30]:
                    if bj in block_set:
                        continue

                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1

                    block_set.add(bj)
                    for ts_idx in block_tsubs[bj]:
                        cover_count[ts_idx] += 1

                    improved = True
                    swapped = True
                    break

                if swapped:
                    break

        return list(block_set)

    def two_block_swap_search(blocks, seed):
        rng = random.Random(seed)
        block_set = set(blocks)
        cover_count = [0] * n_tsubsets

        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1

        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            n_checked = 0
            for i, bi in enumerate(current_blocks):
                if time.time() > deadline - 2:
                    break
                if bi not in block_set:
                    continue

                if len(block_set) < 2:
                    break

                others = [b for b in block_set if b != bi]
                if not others:
                    continue
                bj = rng.choice(others)

                n_checked += 1
                if n_checked > 8000:
                    break

                req_bi = [ts for ts in block_tsubs[bi] if cover_count[ts] == 1]
                req_bj = [ts for ts in block_tsubs[bj] if cover_count[ts] == 1]

                required = set(req_bi)
                required.update(req_bj)

                inter = block_tsub_sets[bi] & block_tsub_sets[bj]
                for ts in inter:
                    if cover_count[ts] == 2:
                        required.add(ts)

                if not required:
                    continue

                min_ts = min(required, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_block_sets[min_ts])

                for ts in required:
                    if ts == min_ts:
                        continue
                    if not candidates:
                        break
                    candidates &= tsub_block_sets[ts]

                candidates.discard(bi)
                candidates.discard(bj)

                if not candidates:
                    continue

                cand_list = list(candidates)
                rng.shuffle(cand_list)

                swapped = False
                for bk in cand_list[:15]:
                    if bk in block_set:
                        continue

                    block_set.remove(bi)
                    for ts in block_tsubs[bi]:
                        cover_count[ts] -= 1

                    block_set.remove(bj)
                    for ts in block_tsubs[bj]:
                        cover_count[ts] -= 1

                    block_set.add(bk)
                    for ts in block_tsubs[bk]:
                        cover_count[ts] += 1

                    improved = True
                    swapped = True
                    break

                if swapped:
                    break

        return list(block_set)

    def final_polish(blocks, seed):
        """
        Aggressive final polishing with more iterations and tighter search.
        """
        rng = random.Random(seed)
        block_set = set(blocks)
        cover_count = [0] * n_tsubsets

        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1

        # Try multiple passes with increasing aggressiveness
        for pass_num in range(5):
            if time.time() > deadline - 2:
                break

            # Remove redundant
            to_remove = []
            for bi in block_set:
                redundant = True
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(bi)
            if to_remove:
                for bi in to_remove:
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                continue

            # Try 1-for-1 swaps with more candidates
            improved = False
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            for bi in current_blocks:
                if time.time() > deadline - 2:
                    break
                if bi not in block_set:
                    continue
                unique_tsubs = [ts for ts in block_tsubs[bi] if cover_count[ts] == 1]
                if not unique_tsubs:
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break
                min_ts = min(unique_tsubs, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_block_sets[min_ts])
                for ts_idx in unique_tsubs[1:]:
                    if not candidates:
                        break
                    candidates &= tsub_block_sets[ts_idx]
                candidates.discard(bi)
                if not candidates:
                    continue
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                # Try more candidates in later passes
                limit = 20 + pass_num * 10
                for bj in cand_list[:limit]:
                    if bj in block_set:
                        continue
                    block_set.remove(bi)
                    for ts in block_tsubs[bi]:
                        cover_count[ts] -= 1
                    block_set.add(bj)
                    for ts in block_tsubs[bj]:
                        cover_count[ts] += 1
                    improved = True
                    break
                if improved:
                    break

            if not improved and time.time() < deadline - 2:
                # Try 2-for-1 swaps
                current_blocks = list(block_set)
                rng.shuffle(current_blocks)
                n_checked = 0
                max_checks = 5000 + pass_num * 2000
                for bi in current_blocks:
                    if time.time() > deadline - 2:
                        break
                    if bi not in block_set:
                        continue
                    if len(block_set) < 2:
                        break
                    others = [b for b in block_set if b != bi]
                    if not others:
                        continue
                    bj = rng.choice(others)
                    n_checked += 1
                    if n_checked > max_checks:
                        break
                    req_bi = [ts for ts in block_tsubs[bi] if cover_count[ts] == 1]
                    req_bj = [ts for ts in block_tsubs[bj] if cover_count[ts] == 1]
                    required = set(req_bi)
                    required.update(req_bj)
                    inter = block_tsub_sets[bi] & block_tsub_sets[bj]
                    for ts in inter:
                        if cover_count[ts] == 2:
                            required.add(ts)
                    if not required:
                        continue
                    min_ts = min(required, key=lambda x: len(tsub_blocks[x]))
                    candidates = set(tsub_block_sets[min_ts])
                    for ts in required:
                        if ts == min_ts:
                            continue
                        if not candidates:
                            break
                        candidates &= tsub_block_sets[ts]
                    candidates.discard(bi)
                    candidates.discard(bj)
                    if not candidates:
                        continue
                    cand_list = list(candidates)
                    rng.shuffle(cand_list)
                    limit = 10 + pass_num * 5
                    for bk in cand_list[:limit]:
                        if bk in block_set:
                            continue
                        block_set.remove(bi)
                        for ts in block_tsubs[bi]:
                            cover_count[ts] -= 1
                        block_set.remove(bj)
                        for ts in block_tsubs[bj]:
                            cover_count[ts] -= 1
                        block_set.add(bk)
                        for ts in block_tsubs[bk]:
                            cover_count[ts] += 1
                        improved = True
                        break
                    if improved:
                        break
                if not improved:
                    break

        return list(block_set)

    best_blocks = None
    best_len = float('inf')

    # Run multiple initial constructions: mix of greedy and primal-dual
    num_trials = 35
    for i in range(num_trials):
        if time.time() > deadline - 25:
            break

        # Alternate between greedy and primal-dual for diversity
        if i % 2 == 0:
            blocks = greedy_cover(i)
        else:
            blocks = primal_dual_cover(i)

        if not blocks:
            continue

        # Local search (single swaps)
        blocks = local_search(blocks, i)

        # Two-block swap search
        if time.time() < deadline - 12:
            blocks = two_block_swap_search(blocks, i + 1000)

        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks

    # Final polish on the best solution
    if best_blocks is not None and time.time() < deadline - 8:
        best_blocks = final_polish(best_blocks, 77777)

    # Fallback if something went wrong
    if best_blocks is None:
        best_blocks = greedy_cover(0)

    return [all_blocks[i] for i in best_blocks]
