"""Structural change: formulate covering design as minimum set cover and solve
via a fast randomized greedy with vectorized numpy scoring, then apply a
multi-pass local search (redundancy removal + point-swap + block-replacement).
Key differences from prior attempts:
  - No precomputation of the full 75582x11628 incidence matrix (too large/slow).
  - Instead, maintain a bitset of covered 5-subsets and score candidate blocks
    on the fly using precomputed per-block t-subset index arrays.
  - Local search uses numpy-vectorized redundancy detection and targeted swaps.
  - Deadline guard with safety margin.
"""
import time
import itertools
import random
import numpy as np

_DEADLINE = 145.0
_START = time.time()


def _time_left():
    return _DEADLINE - time.time()


def _build_tsub_index(v, t):
    """Map every t-subset of {0..v-1} to a unique index."""
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    return all_tsubs, tsub_index, len(all_tsubs)


def _block_tsub_indices(block, tsub_index, t):
    """Return list of t-subset indices covered by a block."""
    return [tsub_index[ts] for ts in itertools.combinations(block, t)]


def _greedy_cover(v, k, t, all_ksubs, tsub_index, n_tsubs, time_limit):
    """Randomized greedy set cover with on-the-fly scoring.

    Returns list of block indices (into all_ksubs) that form a valid cover.
    """
    # Precompute for each k-subset the list of t-subset indices it covers
    # We do this lazily: build a dict block_idx -> array of t-sub indices
    # To avoid storing 75582 arrays upfront, we compute on demand and cache.

    covered = np.zeros(n_tsubs, dtype=np.int8)
    uncovered_count = n_tsubs
    selected = []
    cache = {}

    start = time.time()
    while uncovered_count > 0 and (time.time() - start) < time_limit:
        # For efficiency, we sample a subset of candidate blocks each iteration
        # when the problem is large, or iterate all when small
        n_candidates = len(all_ksubs)
        if n_candidates > 5000:
            # Sample 2000 random candidates + always include best-so-far
            sample_size = min(2000, n_candidates)
            candidates = random.sample(range(n_candidates), sample_size)
        else:
            candidates = list(range(n_candidates))

        best_gain = -1
        best_idx = -1
        for bi in candidates:
            if bi in cache:
                tsubs = cache[bi]
            else:
                tsubs = np.array(_block_tsub_indices(all_ksubs[bi], tsub_index, t), dtype=np.int32)
                cache[bi] = tsubs
            gain = int((covered[tsubs] == 0).sum())
            if gain > best_gain:
                best_gain = gain
                best_idx = bi

        if best_gain <= 0:
            # No improvement; try a full scan as fallback
            for bi in range(n_candidates):
                if bi in cache:
                    tsubs = cache[bi]
                else:
                    tsubs = np.array(_block_tsub_indices(all_ksubs[bi], tsub_index, t), dtype=np.int32)
                    cache[bi] = tsubs
                gain = int((covered[tsubs] == 0).sum())
                if gain > best_gain:
                    best_gain = gain
                    best_idx = bi
            if best_gain <= 0:
                break

        selected.append(best_idx)
        tsubs = cache[best_idx]
        newly = covered[tsubs] == 0
        uncovered_count -= int(newly.sum())
        covered[tsubs] = 1

    return selected


def _validate_cover(blocks, v, t, tsub_index):
    """Check all t-subsets are covered. Returns set of uncovered t-sub indices."""
    covered = set()
    for block in blocks:
        for ts in itertools.combinations(block, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                covered.add(idx)
    all_idx = set(range(len(tsub_index)))
    return all_idx - covered


def _local_search(blocks, v, k, t, tsub_index, n_tsubs, time_limit):
    """Local search to reduce block count.

    Phase 1: Remove redundant blocks (those whose coverage is fully contained
             in the union of other blocks' coverage).
    Phase 2: For each block, try replacing one point with another to see if
             coverage is preserved (enabling future redundancy removal).
    Phase 3: Try replacing a block with a different k-subset that covers all
             its unique t-subsets.
    """
    blocks = [sorted(b) for b in blocks]
    n_blocks = len(blocks)

    # Build coverage structures
    # block_tsubs[i] = set of t-sub indices covered by block i
    # tsub_blocks[j] = set of block indices that cover t-sub j
    block_tsubs = []
    tsub_blocks = [set() for _ in range(n_tsubs)]
    for bi, block in enumerate(blocks):
        tset = set()
        for ts in itertools.combinations(block, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                tset.add(idx)
                tsub_blocks[idx].add(bi)
        block_tsubs.append(tset)

    start = time.time()
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False

        # Phase 1: Remove redundant blocks
        removed_any = False
        for bi in range(len(blocks)):
            if (time.time() - start) > time_limit:
                break
            redundant = True
            for ts_idx in block_tsubs[bi]:
                if len(tsub_blocks[ts_idx]) <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block bi
                for ts_idx in block_tsubs[bi]:
                    tsub_blocks[ts_idx].discard(bi)
                block_tsubs.pop(bi)
                blocks.pop(bi)
                improved = True
                removed_any = True
                break

        if removed_any:
            continue

        # Phase 2: Point swap - try replacing one point in a block with another
        for bi in range(len(blocks)):
            if (time.time() - start) > time_limit:
                break
            block = blocks[bi]
            found_swap = False
            for pi in range(k):
                if found_swap:
                    break
                old_p = block[pi]
                for new_p in range(v):
                    if new_p == old_p or new_p in block:
                        continue
                    new_block = sorted(block[:pi] + [new_p] + block[pi + 1:])
                    # Compute new coverage
                    new_tset = set()
                    for ts in itertools.combinations(new_block, t):
                        idx = tsub_index.get(ts)
                        if idx is not None:
                            new_tset.add(idx)
                    # Check: every t-sub in old block_tsubs[bi] must be in new_tset
                    # or covered by some other block
                    ok = True
                    for ts_idx in block_tsubs[bi]:
                        if ts_idx not in new_tset:
                            others = tsub_blocks[ts_idx]
                            others.discard(bi)
                            if not others:
                                ok = False
                                break
                    if ok:
                        # Perform swap
                        for ts_idx in block_tsubs[bi]:
                            tsub_blocks[ts_idx].discard(bi)
                        for ts_idx in new_tset:
                            tsub_blocks[ts_idx].add(bi)
                        block_tsubs[bi] = new_tset
                        blocks[bi] = new_block
                        improved = True
                        found_swap = True
                        break

        # Phase 3: Block replacement - for each block, try to find a k-subset
        # that covers all t-subsets uniquely covered by this block
        # This is expensive; do it sparsely
        if (time.time() - start) > time_limit:
            break
        # Only do this for blocks with few unique t-subsets (cheap to satisfy)
        for bi in range(len(blocks)):
            if (time.time() - start) > time_limit:
                break
            unique_tsubs = set()
            for ts_idx in block_tsubs[bi]:
                if len(tsub_blocks[ts_idx]) == 1:
                    unique_tsubs.add(ts_idx)
            if len(unique_tsubs) > 10:
                continue  # Too many to satisfy easily

            # Try to find a k-subset containing all points of these t-subsets
            required_points = set()
            for ts_idx in unique_tsubs:
                # Get the actual t-subset
                ts = next(iter([k2 for k2, v2 in tsub_index.items() if v2 == ts_idx]))
                required_points.update(ts)

            if len(required_points) > k:
                continue

            # Try adding random points to fill to size k
            remaining_points = set(range(v)) - required_points
            random.shuffle(list(remaining_points))
            for _ in range(50):
                extra = list(remaining_points)[:k - len(required_points)]
                candidate = sorted(required_points | set(extra))
                if len(candidate) != k:
                    continue
                # Check if candidate covers all unique t-subsets
                cand_tset = set()
                for ts in itertools.combinations(candidate, t):
                    idx = tsub_index.get(ts)
                    if idx is not None:
                        cand_tset.add(idx)
                if unique_tsubs <= cand_tset:
                    # Replace block
                    for ts_idx in block_tsubs[bi]:
                        tsub_blocks[ts_idx].discard(bi)
                    for ts_idx in cand_tset:
                        tsub_blocks[ts_idx].add(bi)
                    block_tsubs[bi] = cand_tset
                    blocks[bi] = candidate
                    improved = True
                    break

    return blocks


def construct(v, k, t):
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 145.0

    rng = random.Random(42)
    all_tsubs, tsub_index, n_tsubs = _build_tsub_index(v, t)
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # Phase 1: Greedy cover
    greedy_time = min(40, _time_left() * 0.35)
    selected_indices = _greedy_cover(v, k, t, all_ksubs, tsub_index, n_tsubs, greedy_time)
    blocks = [list(all_ksubs[i]) for i in selected_indices]

    # Validate
    uncovered = _validate_cover(blocks, v, t, tsub_index)
    if uncovered:
        # Add blocks to cover remaining t-subsets
        for ts_idx in sorted(uncovered):
            ts = next(iter([k2 for k2, v2 in tsub_index.items() if v2 == ts_idx]))
            block = list(ts)
            for p in range(v):
                if p not in block and len(block) < k:
                    block.append(p)
            block.sort()
            if tuple(block) not in set(tuple(b) for b in blocks):
                blocks.append(block)
        # Re-validate
        uncovered = _validate_cover(blocks, v, t, tsub_index)
        if uncovered:
            # Emergency: add one block per uncovered t-subset
            for ts_idx in sorted(uncovered):
                ts = next(iter([k2 for k2, v2 in tsub_index.items() if v2 == ts_idx]))
                block = list(ts) + [p for p in range(v) if p not in ts][:k - t]
                block.sort()
                if tuple(block) not in set(tuple(b) for b in blocks):
                    blocks.append(block)

    # Phase 2: Local search
    remaining = _time_left()
    if remaining > 10:
        ls_time = min(remaining - 5, 100)
        blocks = _local_search(blocks, v, k, t, tsub_index, n_tsubs, ls_time)

    # Final validation and deduplication
    uncovered = _validate_cover(blocks, v, t, tsub_index)
    if uncovered:
        for ts_idx in sorted(uncovered):
            ts = next(iter([k2 for k2, v2 in tsub_index.items() if v2 == ts_idx]))
            block = list(ts)
            for p in range(v):
                if p not in block and len(block) < k:
                    block.append(p)
            block.sort()
            blocks.append(block)

    # Deduplicate
    seen = set()
    unique_blocks = []
    for b in blocks:
        bt = tuple(sorted(b))
        if bt not in seen:
            seen.add(bt)
            unique_blocks.append(list(bt))

    return unique_blocks
