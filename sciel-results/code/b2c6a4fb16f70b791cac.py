import itertools
import time
import random
import numpy as np
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {frozenset(c): i for i, c in enumerate(all_tsubs)}

    # Precompute all k-blocks and their t-subset coverage
    all_kblocks = list(itertools.combinations(range(v), k))
    n_kblocks = len(all_kblocks)

    # block_tsubs[bi] = list of t-subset indices covered by block bi
    block_tsubs = [None] * n_kblocks
    # tsub_blocks[j] = list of block indices that cover t-subset j
    tsub_blocks = [[] for _ in range(n_tsubs)]

    for bi, block in enumerate(all_kblocks):
        tsubs = [tsub_to_idx[frozenset(c)] for c in itertools.combinations(block, t)]
        block_tsubs[bi] = tsubs
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)

    # Precompute sets for faster operations
    block_tsub_sets = [set(tsubs) for tsubs in block_tsubs]
    tsub_block_sets = [set(lst) for lst in tsub_blocks]
    
    # Bitset representation for fast intersection
    # Each t-subset has a bitset of blocks that cover it
    n_bits = (n_kblocks + 63) // 64
    tsub_block_bitsets = [None] * n_tsubs
    for j in range(n_tsubs):
        bits = [0] * n_bits
        for bi in tsub_blocks[j]:
            bits[bi // 64] |= (1 << (bi % 64))
        tsub_block_bitsets[j] = bits
    
    def _intersect_bitsets(bitsets):
        """Intersect multiple bitsets, return list of set bits."""
        if not bitsets:
            return []
        # Start with the smallest bitset (fewest bits)
        min_idx = min(range(len(bitsets)), key=lambda i: sum(bin(x).count('1') for x in bitsets[i]))
        result = list(bitsets[min_idx])
        for other_idx in range(len(bitsets)):
            if other_idx == min_idx:
                continue
            other = bitsets[other_idx]
            for i in range(len(result)):
                result[i] &= other[i]
        # Extract set bits
        bits_list = []
        for word_idx, word in enumerate(result):
            while word:
                bit = (word & -word).bit_length() - 1
                bits_list.append(word_idx * 64 + bit)
                word &= word - 1
        return bits_list

    def _primal_dual_greedy(rng, dl):
        """
        Primal-dual approximation: maintain dual variables y[j] for each t-subset j.
        Select blocks where marginal cost (1) is balanced by accumulated dual weight
        on uncovered t-subsets it covers.
        """
        uncovered = np.ones(n_tsubs, dtype=bool)
        dual = np.zeros(n_tsubs, dtype=np.float64)  # dual variables
        selected = []
        selected_set = set()

        # Initial gains: number of uncovered t-subsets covered by each block
        gain = np.array([len(block_tsubs[bi]) for bi in range(n_kblocks)], dtype=np.int64)

        # Max-heap: (-gain, bi)
        heap = [(-gain[bi], bi) for bi in range(n_kblocks)]
        heapq.heapify(heap)

        while True:
            if time.time() > dl:
                break

            # Pop until we find a valid (non-stale) entry with positive gain
            best_bi = -1
            max_gain = 0
            while heap:
                neg_g, bi = heapq.heappop(heap)
                g = -neg_g
                if bi in selected_set:
                    continue
                if g == gain[bi] and g > 0:
                    best_bi = bi
                    max_gain = g
                    break

            if max_gain <= 0:
                break

            selected.append(best_bi)
            selected_set.add(best_bi)

            # Update: for each newly covered t-subset, increase dual and update gains
            for ts_idx in block_tsubs[best_bi]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    # Increase dual variable for this t-subset
                    dual[ts_idx] = 1.0
                    # Decrement gain for all blocks covering this t-subset
                    for bj in tsub_blocks[ts_idx]:
                        if bj not in selected_set:
                            gain[bj] -= 1
                            heapq.heappush(heap, (-gain[bj], bj))

            # Mark selected block
            gain[best_bi] = -1
            heapq.heappush(heap, (-gain[best_bi], best_bi))

        return [list(all_kblocks[bi]) for bi in selected]

    def _local_search(blocks, dl):
        """
        Local search: remove redundant blocks, then try 1-for-1 swaps
        using bitset intersection for faster candidate generation.
        """
        # Convert to block indices
        block_to_idx = {tuple(b): i for i, b in enumerate(all_kblocks)}
        block_indices = [block_to_idx[tuple(b)] for b in blocks]
        block_set = set(block_indices)

        # Compute cover counts
        cover_count = np.zeros(n_tsubs, dtype=np.int64)
        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1

        improved = True
        while improved and time.time() < dl:
            improved = False

            # Phase 1: Remove redundant blocks
            to_remove = []
            for bi in list(block_set):
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

            # Phase 2: Try 1-for-1 swaps using bitset intersection
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            for bi in current_blocks:
                if time.time() > dl:
                    break
                if bi not in block_set:
                    continue

                # Find unique t-subsets for bi
                unique_tsubs = []
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)

                if not unique_tsubs:
                    # Block is redundant
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break

                # Find candidate blocks that cover all unique_tsubs using bitsets
                candidate_bitsets = [tsub_block_bitsets[ts_idx] for ts_idx in unique_tsubs]
                candidates = _intersect_bitsets(candidate_bitsets)
                
                # Remove bi from candidates
                if bi in candidates:
                    candidates.remove(bi)

                if not candidates:
                    continue

                cand_list = candidates[:]
                rng.shuffle(cand_list)

                swapped = False
                for bj in cand_list[:20]:
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

        return [list(all_kblocks[bi]) for bi in block_set]

    def _two_block_swap(blocks, dl):
        """
        Try to replace two blocks with one that covers all their unique t-subsets.
        Uses bitset intersection for faster candidate generation.
        """
        block_to_idx = {tuple(b): i for i, b in enumerate(all_kblocks)}
        block_indices = [block_to_idx[tuple(b)] for b in blocks]
        block_set = set(block_indices)

        cover_count = np.zeros(n_tsubs, dtype=np.int64)
        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1

        improved = True
        while improved and time.time() < dl:
            improved = False
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            for i, bi in enumerate(current_blocks):
                if time.time() > dl:
                    break
                if bi not in block_set:
                    continue

                if len(block_set) < 2:
                    break

                others = [b for b in block_set if b != bi]
                if not others:
                    break

                # Pick a random partner
                bj = rng.choice(others)

                # Find unique t-subsets for bi and bj
                unique_bi = [ts for ts in block_tsubs[bi] if cover_count[ts] == 1]
                unique_bj = [ts for ts in block_tsubs[bj] if cover_count[ts] == 1]

                if not unique_bi or not unique_bj:
                    continue

                all_unique = set(unique_bi) | set(unique_bj)

                # Find candidate blocks that cover all unique t-subsets using bitsets
                candidate_bitsets = [tsub_block_bitsets[ts_idx] for ts_idx in all_unique]
                candidates = _intersect_bitsets(candidate_bitsets)
                
                # Remove bi and bj from candidates
                if bi in candidates:
                    candidates.remove(bi)
                if bj in candidates:
                    candidates.remove(bj)

                if not candidates:
                    continue

                cand_list = candidates[:]
                rng.shuffle(cand_list)

                swapped = False
                for bk in cand_list[:10]:
                    if bk in block_set:
                        continue

                    # Perform 2-for-1 swap
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    block_set.remove(bj)
                    for ts_idx in block_tsubs[bj]:
                        cover_count[ts_idx] -= 1

                    block_set.add(bk)
                    for ts_idx in block_tsubs[bk]:
                        cover_count[ts_idx] += 1

                    improved = True
                    swapped = True
                    break

                if swapped:
                    break

        return [list(all_kblocks[bi]) for bi in block_set]

    best_blocks = None
    best_len = float('inf')

    # Multiple restarts with primal-dual greedy
    for seed in range(10):
        if time.time() > deadline - 30:
            break
        rng = random.Random(seed * 7919 + 13)
        blocks = _primal_dual_greedy(rng, deadline - 30)
        if blocks and len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)

    if best_blocks and time.time() < deadline - 10:
        best_blocks = _local_search(best_blocks, deadline - 5)
        if len(best_blocks) < best_len:
            best_len = len(best_blocks)

    if best_blocks and time.time() < deadline - 5:
        best_blocks = _two_block_swap(best_blocks, deadline)
        if len(best_blocks) < best_len:
            best_len = len(best_blocks)

    return best_blocks if best_blocks else []
