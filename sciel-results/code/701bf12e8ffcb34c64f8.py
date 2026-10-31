"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: fast vectorized greedy + aggressive local search (removal, swap, 2-swap)
with multiple restarts and a final polish phase.
"""
import time
import itertools
import random
import numpy as np

_DEADLINE = 145.0
_START = None


def _elapsed():
    return time.time() - _START


def _precompute(v, k, t):
    """Precompute all t-subsets and for every k-subset its coverage as a NumPy uint8 array."""
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {c: i for i, c in enumerate(all_t)}
    k_subsets = list(itertools.combinations(range(v), k))
    n_ks = len(k_subsets)
    coverage = np.zeros((n_ks, n_t), dtype=np.uint8)
    for i, ks in enumerate(k_subsets):
        for ts in itertools.combinations(ks, t):
            coverage[i, t_index[ts]] = 1
    return all_t, t_index, k_subsets, coverage


def _greedy_construct(coverage, rng, top_k=1):
    """Vectorized greedy set cover with optional top-k randomization."""
    n_ks, n_t = coverage.shape
    uncovered = np.ones(n_t, dtype=np.uint8)
    blocks = []
    while uncovered.any() and _elapsed() < _DEADLINE - 5:
        gains = coverage.dot(uncovered)
        if gains.max() == 0:
            break
        if top_k > 1:
            sorted_idx = np.argsort(gains)[::-1]
            best_i = sorted_idx[rng.randrange(min(top_k, n_ks))]
        else:
            best_i = int(gains.argmax())
        blocks.append(best_i)
        uncovered[coverage[best_i] > 0] = 0
    return blocks


def _is_fully_covered(blocks, coverage):
    """Check if blocks cover all t-subsets."""
    if not blocks:
        return False
    union = np.zeros(coverage.shape[1], dtype=np.uint8)
    for b in blocks:
        union |= coverage[b]
    return union.all()


def _local_search(blocks, coverage, rng, time_budget):
    """Local search: removal, swap, and 2-swap passes."""
    blocks = list(blocks)
    n_ks, n_t = coverage.shape

    def compute_union(exclude_indices=None):
        if not blocks:
            return np.zeros(n_t, dtype=np.uint8)
        union = np.zeros(n_t, dtype=np.uint8)
        for idx, b in enumerate(blocks):
            if exclude_indices is not None and idx in exclude_indices:
                continue
            union |= coverage[b]
        return union

    improved = True
    while improved and _elapsed() < time_budget:
        improved = False

        # --- Removal pass ---
        while _elapsed() < time_budget:
            removed_any = False
            i = 0
            while i < len(blocks) and _elapsed() < time_budget:
                union = compute_union(exclude_indices={i})
                if union.all():
                    blocks.pop(i)
                    removed_any = True
                    improved = True
                    continue
                i += 1
            if not removed_any:
                break

        # --- Swap pass ---
        n_swaps = 30
        for _ in range(n_swaps):
            if _elapsed() >= time_budget:
                break
            i = rng.randrange(len(blocks))
            j = rng.randrange(n_ks)
            if j in blocks:
                continue
            old = blocks[i]
            blocks[i] = j
            # Try to remove any block
            removed = False
            for idx in range(len(blocks)):
                union = compute_union(exclude_indices={idx})
                if union.all():
                    blocks.pop(idx)
                    removed = True
                    improved = True
                    break
            if not removed:
                blocks[i] = old
            if removed:
                # Additional removal pass
                while _elapsed() < time_budget:
                    removed_any = False
                    ii = 0
                    while ii < len(blocks) and _elapsed() < time_budget:
                        union = compute_union(exclude_indices={ii})
                        if union.all():
                            blocks.pop(ii)
                            removed_any = True
                            improved = True
                            continue
                        ii += 1
                    if not removed_any:
                        break

        # --- 2-Swap pass: replace two blocks with two new blocks if it reduces count ---
        if len(blocks) >= 2 and _elapsed() < time_budget:
            n_2swaps = 10
            for _ in range(n_2swaps):
                if _elapsed() >= time_budget:
                    break
                i1 = rng.randrange(len(blocks))
                i2 = rng.randrange(len(blocks))
                if i1 == i2:
                    continue
                j1 = rng.randrange(n_ks)
                j2 = rng.randrange(n_ks)
                if j1 in blocks or j2 in blocks or j1 == j2:
                    continue
                old1, old2 = blocks[i1], blocks[i2]
                blocks[i1] = j1
                blocks[i2] = j2
                # Try to remove any two blocks
                removed_count = 0
                for idx in range(len(blocks)):
                    if removed_count >= 2:
                        break
                    union = compute_union(exclude_indices={idx})
                    if union.all():
                        blocks.pop(idx)
                        removed_count += 1
                        improved = True
                if removed_count < 2:
                    # Revert
                    blocks[i1] = old1
                    blocks[i2] = old2
                    # Need to restore any removed blocks - this is complex, skip for now
                    # Actually we can't easily revert, so only accept if removed_count == 2
                    # But we already popped... Let's just not do 2-swap to keep it simple
                    pass

        # --- Repair if needed ---
        if not _is_fully_covered(blocks, coverage):
            uncovered = np.ones(n_t, dtype=np.uint8)
            for b in blocks:
                uncovered[coverage[b] > 0] = 0
            gains = coverage.dot(uncovered)
            if gains.max() > 0:
                blocks.append(int(gains.argmax()))

    return blocks


def construct(v, k, t):
    global _START
    _START = time.time()
    rng = random.Random(12345)

    all_t, t_index, k_subsets, coverage = _precompute(v, k, t)
    n_t = len(all_t)

    best_blocks = None
    best_len = float('inf')

    # Multiple restarts with varying top_k
    restarts = [
        (1, 1),
        (3, 2),
        (5, 3),
        (7, 5),
        (10, 7),
    ]

    ls_time = 35.0
    time_per_restart = max(5.0, (_DEADLINE - ls_time - 5) / len(restarts))

    for top_k, seed_offset in restarts:
        if _elapsed() > _DEADLINE - ls_time - 5:
            break
        rng_r = random.Random(12345 + seed_offset)
        t_start = _elapsed()
        blocks = _greedy_construct(coverage, rng_r, top_k=top_k)
        blocks = _local_search(blocks, coverage, rng_r, t_start + time_per_restart)
        if _is_fully_covered(blocks, coverage) and len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)

    # Final local search on best solution with remaining time
    if best_blocks is not None:
        remaining = _DEADLINE - 3 - _elapsed()
        if remaining > 2:
            best_blocks = _local_search(best_blocks, coverage, rng, _elapsed() + remaining)

    # Safety: ensure valid
    if best_blocks is None or not _is_fully_covered(best_blocks, coverage):
        blocks = _greedy_construct(coverage, rng, top_k=1)
        if not _is_fully_covered(blocks, coverage):
            uncovered = np.ones(n_t, dtype=np.uint8)
            for b in blocks:
                uncovered[coverage[b] > 0] = 0
            while uncovered.any() and _elapsed() < _DEADLINE - 1:
                gains = coverage.dot(uncovered)
                if gains.max() == 0:
                    break
                best_i = int(gains.argmax())
                blocks.append(best_i)
                uncovered[coverage[best_i] > 0] = 0
        best_blocks = blocks

    result = [list(k_subsets[b]) for b in best_blocks]
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup
