"""Covering design C(24,7,3): greedy construction with NumPy-accelerated local search."""
import itertools
import time
import random
import numpy as np
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute for each point, the list of t-subsets it participates in
    # point_tsubs[x] = list of t-subset indices containing x
    point_tsubs = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubs):
        for x in ts:
            point_tsubs[x].append(i)

    # Greedy construction with multiple randomized restarts
    best_blocks = None
    best_count = float('inf')

    rng = random.Random(12345)

    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline - 60.0:  # leave time for local search
            break

        # Randomized greedy: start with a random uncovered t-subset
        uncovered_mask = np.ones(n_tsubs, dtype=np.int32)
        uncovered_count = n_tsubs
        blocks = []

        while uncovered_count > 0:
            if time.time() > deadline - 60.0:
                break

            # Pick a random uncovered t-subset to start
            uncovered_indices = np.where(uncovered_mask == 1)[0]
            start_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
            start_triple = all_tsubs[start_idx]

            block = list(start_triple)
            block_set = set(block)

            # Greedily add points to maximize coverage of uncovered triples
            while len(block) < k:
                best_x = -1
                best_gain = -1

                for x in range(v):
                    if x in block_set:
                        continue

                    # Compute gain: how many uncovered triples would be covered
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        cand = tuple(sorted(rest + (x,)))
                        if cand in tsub_to_idx:
                            idx = tsub_to_idx[cand]
                            if uncovered_mask[idx] == 1:
                                gain += 1

                    if gain > best_gain:
                        best_gain = gain
                        best_x = x

                if best_x == -1 or best_gain == 0:
                    break

                block.append(best_x)
                block_set.add(best_x)

            if len(block) < k:
                break

            block.sort()
            blocks.append(block)

            # Update uncovered mask
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                uncovered_mask[idx] = 0

            uncovered_count = np.sum(uncovered_mask == 1)

        if uncovered_count == 0 and len(blocks) < best_count:
            best_blocks = [list(b) for b in blocks]
            best_count = len(blocks)

            # If we have a good solution, run local search
            if best_count <= 80:
                improved = _local_search_numpy(best_blocks, v, k, t, all_tsubs, tsub_to_idx, point_tsubs, deadline)
                if improved is not None:
                    # Verify
                    cov = np.zeros(n_tsubs, dtype=np.int32)
                    for blk in improved:
                        for ts in itertools.combinations(blk, t):
                            cov[tsub_to_idx[ts]] += 1
                    if np.all(cov >= 1) and len(improved) < best_count:
                        best_blocks = improved
                        best_count = len(improved)

    # Fallback: deterministic greedy
    if best_blocks is None:
        uncovered_mask = np.ones(n_tsubs, dtype=np.int32)
        blocks = []
        while np.any(uncovered_mask == 1):
            # Pick the first uncovered triple
            start_idx = np.argmax(uncovered_mask)
            start_triple = all_tsubs[start_idx]
            block = list(start_triple)
            block_set = set(block)

            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        cand = tuple(sorted(rest + (x,)))
                        if cand in tsub_to_idx:
                            idx = tsub_to_idx[cand]
                            if uncovered_mask[idx] == 1:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1 or best_gain == 0:
                    break
                block.append(best_x)
                block_set.add(best_x)

            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                uncovered_mask[tsub_to_idx[ts]] = 0

        best_blocks = blocks

    # Final local search pass if time remains
    if time.time() < deadline - 10.0:
        improved = _local_search_numpy(best_blocks, v, k, t, all_tsubs, tsub_to_idx, point_tsubs, deadline)
        if improved is not None:
            cov = np.zeros(n_tsubs, dtype=np.int32)
            for blk in improved:
                for ts in itertools.combinations(blk, t):
                    cov[tsub_to_idx[ts]] += 1
            if np.all(cov >= 1) and len(improved) < len(best_blocks):
                best_blocks = improved

    return best_blocks


def _local_search_numpy(blocks, v, k, t, all_tsubs, tsub_to_idx, point_tsubs, deadline):
    """Local search using NumPy for coverage counting."""
    n_tsubs = len(all_tsubs)
    blocks = [list(b) for b in blocks]
    n = len(blocks)

    def _build_cov_matrix(blks):
        cov = np.zeros((len(blks), n_tsubs), dtype=np.int8)
        for i, blk in enumerate(blks):
            for ts in itertools.combinations(blk, t):
                cov[i, tsub_to_idx[ts]] = 1
        return cov

    def _cov_counts(cov):
        return np.sum(cov, axis=0)

    # Pass 1: Remove redundant blocks
    for _pass in range(20):
        if time.time() > deadline - 5.0:
            return blocks
        if not blocks:
            break

        cov = _build_cov_matrix(blocks)
        counts = _cov_counts(cov)
        n = len(blocks)
        removed = False

        for i in range(n):
            if time.time() > deadline - 5.0:
                return blocks

            # Check if block i is redundant
            redundant = True
            for ts in itertools.combinations(blocks[i], t):
                idx = tsub_to_idx[ts]
                if counts[idx] <= 1:
                    redundant = False
                    break

            if redundant:
                blocks.pop(i)
                n -= 1
                removed = True
                # Rebuild
                cov = _build_cov_matrix(blocks)
                counts = _cov_counts(cov)

        if not removed:
            break

    # Pass 2: Try to replace blocks with better ones
    if time.time() > deadline - 5.0:
        return blocks

    for _pass in range(10):
        if time.time() > deadline - 5.0:
            return blocks

        cov = _build_cov_matrix(blocks)
        counts = _cov_counts(cov)
        n = len(blocks)
        improved = False

        for i in range(n):
            if time.time() > deadline - 5.0:
                return blocks

            old_block = blocks[i]
            # Find triples uniquely covered by this block
            unique_triples = []
            for ts in itertools.combinations(old_block, t):
                idx = tsub_to_idx[ts]
                if counts[idx] == 1:
                    unique_triples.append(ts)

            if not unique_triples:
                # Redundant, remove
                blocks.pop(i)
                n -= 1
                improved = True
                cov = _build_cov_matrix(blocks)
                counts = _cov_counts(cov)
                continue

            # Try to find a replacement block that covers all unique_triples
            # and has minimal overlap with existing blocks
            best_new_block = None
            best_score = float('inf')

            # Try starting from each unique triple
            for start_triple in unique_triples[:3]:  # Limit tries
                new_block = list(start_triple)
                new_block_set = set(new_block)

                while len(new_block) < k:
                    best_x = -1
                    best_gain = -1
                    for x in range(v):
                        if x in new_block_set:
                            continue
                        gain = 0
                        for rest in itertools.combinations(new_block, t - 1):
                            cand = tuple(sorted(rest + (x,)))
                            if cand in tsub_to_idx:
                                idx = tsub_to_idx[cand]
                                # Prefer covering unique triples
                                if idx in [tsub_to_idx[ts] for ts in unique_triples]:
                                    gain += 10
                                elif counts[idx] == 0:
                                    gain += 5
                                else:
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x == -1 or best_gain == 0:
                        break
                    new_block.append(best_x)
                    new_block_set.add(best_x)

                if len(new_block) < k:
                    continue

                new_block.sort()
                # Check if it covers all unique triples
                new_tsubs = set(itertools.combinations(new_block, t))
                if all(ts in new_tsubs for ts in unique_triples):
                    # Score: number of new triples covered (higher is better)
                    new_covered = sum(1 for ts in itertools.combinations(new_block, t)
                                      if tsub_to_idx[ts] not in set(tsub_to_idx[ts2] for ts2 in old_block))
                    if new_covered < best_score:
                        best_score = new_covered
                        best_new_block = new_block

            if best_new_block is not None:
                # Check for duplicates
                if not any(tuple(b) == tuple(best_new_block) for b in blocks if b != old_block):
                    blocks[i] = best_new_block
                    improved = True
                    cov = _build_cov_matrix(blocks)
                    counts = _cov_counts(cov)

        if not improved:
            break

    return blocks
