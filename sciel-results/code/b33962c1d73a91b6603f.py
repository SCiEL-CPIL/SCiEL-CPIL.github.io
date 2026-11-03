"""Covering design C(19,7,4) via heap-based greedy with multi-restart and local search."""
import itertools
import random
import time
import numpy as np
import heapq
from collections import defaultdict

DEADLINE_MARGIN = 8.0  # seconds before the 150s limit

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsubset_index = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Precompute all k-subsets (blocks)
    all_ksubsets = list(itertools.combinations(range(v), k))
    n_ksubsets = len(all_ksubsets)

    # For each block, compute the list of t-subset indices it covers
    block_tsubs = [[] for _ in range(n_ksubsets)]
    # For each t-subset, compute the list of block indices that cover it
    tsub_blocks = [[] for _ in range(n_tsubsets)]

    for bi, block in enumerate(all_ksubsets):
        tsubs = [tsubset_index[ts] for ts in itertools.combinations(block, t)]
        block_tsubs[bi] = tsubs
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)

    # Precompute sets for faster intersection
    tsub_block_sets = [set(lst) for lst in tsub_blocks]

    def greedy_cover(seed, time_budget):
        """
        Greedy set cover using a max-heap with lazy updates.
        """
        rng = random.Random(seed)
        uncovered = [True] * n_tsubsets
        gain = [len(block_tsubs[bi]) for bi in range(n_ksubsets)]

        heap = [(-gain[bi], bi) for bi in range(n_ksubsets)]
        heapq.heapify(heap)

        selected = []

        while time.time() < start_time + time_budget:
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

        return selected, uncovered

    def local_search(blocks, seed, time_budget):
        """
        Local search: Remove redundant blocks and perform swaps to improve coverage.
        """
        rng = random.Random(seed)
        block_set = set(blocks)

        # cover_count[ts_idx] = number of blocks in block_set covering ts_idx
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1

        improved = True
        while improved and time.time() < start_time + time_budget:
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
                if time.time() > start_time + time_budget:
                    break
                if bi not in block_set:
                    continue

                # Find unique t-subsets for bi
                unique_tsubs = []
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)

                if not unique_tsubs:
                    # This block is redundant
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break

                # Find candidate blocks that cover ALL unique_tsubs
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

                    # Perform swap
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

        # Check if all t-subsets are covered
        for ts_idx in range(n_tsubsets):
            if cover_count[ts_idx] == 0:
                return None  # Invalid

        return list(block_set)

    # Multi-restart strategy
    best_blocks = None
    best_len = float('inf')

    # Allocate time budget: 70% for greedy+local search, 30% for final refinement
    greedy_budget = 80.0
    local_budget = 50.0
    final_budget = 5.0

    num_restarts = 0
    while time.time() < start_time + greedy_budget:
        seed = num_restarts
        blocks, uncovered = greedy_cover(seed, greedy_budget)

        if not uncovered:
            # Try local search to improve
            refined = local_search(blocks, seed, local_budget)
            if refined is not None:
                if len(refined) < best_len:
                    best_blocks = refined
                    best_len = len(refined)
        else:
            # Greedy didn't finish, try to complete with remaining time
            # For now, skip this restart
            pass

        num_restarts += 1
        if time.time() > start_time + greedy_budget:
            break

    if best_blocks is None:
        # Fallback: use the last greedy result even if incomplete
        blocks, uncovered = greedy_cover(0, greedy_budget)
        if uncovered:
            # Try to fill in missing t-subsets
            remaining_uncovered = [i for i, u in enumerate(uncovered) if u]
            for ts_idx in remaining_uncovered:
                for bi in tsub_blocks[ts_idx]:
                    if bi not in blocks:
                        blocks.append(bi)
                        break
            best_blocks = blocks
            best_len = len(blocks)
        else:
            best_blocks = blocks
            best_len = len(blocks)

    # Final refinement pass
    if time.time() < start_time + final_budget:
        refined = local_search(best_blocks, 999, final_budget)
        if refined is not None and len(refined) < best_len:
            best_blocks = refined
            best_len = len(refined)

    return [list(all_ksubsets[bi]) for bi in best_blocks]
