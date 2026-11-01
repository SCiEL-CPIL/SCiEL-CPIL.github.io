"""Covering design C(24,7,3): bitset-accelerated greedy + block-replacement local search."""
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute all t-subsets (triples)
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute all k-subsets (7-blocks)
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # For each k-subset, precompute the bitmask of t-subsets it covers
    # and the list of t-subset indices
    ksub_bitmasks = np.zeros(n_ksubs, dtype=np.uint64)
    ksub_tsub_idx = []  # list of lists of t-subset indices
    for ki, ks in enumerate(all_ksubs):
        tsubs = list(itertools.combinations(ks, t))
        idxs = [tsub_to_idx[ts] for ts in tsubs]
        ksub_tsub_idx.append(idxs)
        mask = 0
        for idx in idxs:
            mask |= (1 << idx)
        ksub_bitmasks[ki] = mask

    # For fast lookup: for each t-subset, which k-subsets cover it
    tsub_to_ksubs = [[] for _ in range(n_tsubs)]
    for ki, ks in enumerate(all_ksubs):
        for ts in itertools.combinations(ks, t):
            tsub_to_ksubs[tsub_to_idx[ts]].append(ki)

    best_blocks = None
    best_count = float('inf')

    def _greedy_construct(rng_seed, deadline):
        """Fast greedy using bitmasks for coverage tracking."""
        local_rng = random.Random(rng_seed)
        uncovered_mask = (1 << n_tsubs) - 1
        blocks = []

        while uncovered_mask:
            if time.time() > deadline:
                return None

            # Pick a random uncovered triple
            # Find an uncovered t-subset index
            # Use a simple approach: iterate through set bits
            uncovered_list = []
            m = uncovered_mask
            while m:
                low_bit = m & (-m)
                idx = low_bit.bit_length() - 1
                uncovered_list.append(idx)
                m ^= low_bit

            if not uncovered_list:
                break

            start_idx = local_rng.choice(uncovered_list)
            start_triple = all_tsubs[start_idx]
            block = list(start_triple)
            block_set = set(block)

            # Mark covered
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                uncovered_mask &= ~(1 << idx)

            while len(block) < k:
                if time.time() > deadline:
                    return None

                best_x = -1
                best_gain = -1

                for x in range(v):
                    if x in block_set:
                        continue

                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        idx = tsub_to_idx.get(candidate)
                        if idx is not None and (uncovered_mask >> idx) & 1:
                            gain += 1

                    if gain > best_gain:
                        best_gain = gain
                        best_x = x

                if best_x == -1:
                    break

                block.append(best_x)
                block_set.add(best_x)

                for rest in itertools.combinations(block[:-1], t - 1):
                    candidate = tuple(sorted(rest + (best_x,)))
                    idx = tsub_to_idx.get(candidate)
                    if idx is not None:
                        uncovered_mask &= ~(1 << idx)

            if len(block) < k:
                return None

            block.sort()
            blocks.append(block)

        return blocks

    def _remove_redundant(blocks, deadline):
        """Remove blocks whose t-subsets are all covered by other blocks."""
        if not blocks:
            return blocks

        n = len(blocks)
        # Coverage count for each t-subset
        cov_count = np.zeros(n_tsubs, dtype=np.int32)
        for blk in blocks:
            for ts in itertools.combinations(blk, t):
                idx = tsub_to_idx[ts]
                cov_count[idx] += 1

        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            new_blocks = []
            for blk in blocks:
                redundant = True
                for ts in itertools.combinations(blk, t):
                    idx = tsub_to_idx[ts]
                    if cov_count[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for ts in itertools.combinations(blk, t):
                        idx = tsub_to_idx[ts]
                        cov_count[idx] -= 1
                    changed = True
                else:
                    new_blocks.append(blk)
            blocks = new_blocks

        return blocks

    def _local_search(blocks, deadline):
        """Try to replace each block with a better one that creates redundancy."""
        if not blocks:
            return blocks

        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False

            n = len(blocks)
            # Build coverage map
            cov = {}
            for i, blk in enumerate(blocks):
                for tsub in itertools.combinations(blk, t):
                    if tsub not in cov:
                        cov[tsub] = []
                    cov[tsub].append(i)

            for i in range(n):
                if time.time() > deadline:
                    return blocks

                old_block = blocks[i]
                old_tsubs = list(itertools.combinations(old_block, t))

                # Find t-subsets uniquely covered by this block
                unique_tsubs = []
                for tsub in old_tsubs:
                    covering = cov.get(tsub, [])
                    if len(covering) == 1:
                        unique_tsubs.append(tsub)

                if not unique_tsubs:
                    # Block is redundant, remove it
                    for tsub in old_tsubs:
                        if tsub in cov:
                            cov[tsub].remove(i)
                    blocks.pop(i)
                    improved = True
                    break

                # Find the minimal set of points that must be in the new block
                required_points = set()
                for tsub in unique_tsubs:
                    required_points.update(tsub)

                if len(required_points) > k:
                    continue

                # Find k-subsets that contain all required_points
                candidates = []
                for ki, ks in enumerate(all_ksubs):
                    if all(p in ks for p in required_points):
                        candidates.append(ki)

                if not candidates:
                    continue

                best_candidate = None
                best_redundancy_gain = -1

                for ki in candidates:
                    new_block = all_ksubs[ki]
                    new_tsub_set = set(itertools.combinations(new_block, t))

                    # Check: does new_block cover all unique_tsubs?
                    covers_all = True
                    for tsub in unique_tsubs:
                        if tsub not in new_tsub_set:
                            covers_all = False
                            break
                    if not covers_all:
                        continue

                    # Count how many blocks become redundant after this swap
                    redundancy_gain = 0
                    for j in range(n):
                        if j == i:
                            continue
                        blk_j = blocks[j]
                        redundant = True
                        for tsub in itertools.combinations(blk_j, t):
                            # Is tsub covered by some block other than j?
                            # Check blocks \ {j, i} and new_block
                            covered = False
                            if tsub in new_tsub_set:
                                covered = True
                            else:
                                for m in range(n):
                                    if m == j or m == i:
                                        continue
                                    if tsub in itertools.combinations(blocks[m], t):
                                        covered = True
                                        break
                            if not covered:
                                redundant = False
                                break
                        if redundant:
                            redundancy_gain += 1

                    if redundancy_gain > best_redundancy_gain:
                        best_redundancy_gain = redundancy_gain
                        best_candidate = ki

                if best_candidate is not None and best_redundancy_gain > 0:
                    new_block = list(all_ksubs[best_candidate])
                    new_block.sort()
                    blocks[i] = new_block
                    improved = True
                    break

        return blocks

    # Main search loop
    for restart in range(300):
        if time.time() > deadline:
            break

        blocks = _greedy_construct(42 + restart, deadline)

        if blocks is not None:
            # Remove redundant blocks
            blocks = _remove_redundant(blocks, deadline)

            if len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)

            # Try local search
            if best_blocks is not None and len(best_blocks) <= 85:
                improved = _local_search(list(best_blocks), deadline)
                improved = _remove_redundant(improved, deadline)
                if len(improved) < best_count:
                    best_blocks = improved
                    best_count = len(improved)

    # Fallback: simple greedy
    if best_blocks is None:
        uncovered = set(all_tsubs)
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            block_set = set(block)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    break
                block.append(best_x)
                block_set.add(best_x)
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        best_blocks = blocks

    return best_blocks
