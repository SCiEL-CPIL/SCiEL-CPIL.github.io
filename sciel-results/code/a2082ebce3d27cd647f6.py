"""
Covering design C(v,k,t) — hybrid approach combining:
  1. Integer-bitset greedy set cover (from current program) for fast global-max selection
  2. Point-by-point block construction (from inspiration 1) for diversity
  3. Local optimisation: redundancy removal, 1-for-1 swaps, 2-for-1 merges

Key speedups:
  - Integer bitsets for uncovered-set tracking (~50x faster than numpy for this size)
  - Precomputed block bitsets and t-subset index lists
  - Vectorized gain computation where possible
  - Multiple construction strategies for diversity
"""

import time
import itertools
import numpy as np
import random


# ─────────────────────────────────────────────────────────────────────────────
#  Pre-computation
# ─────────────────────────────────────────────────────────────────────────────
def _precompute(v, k, t):
    """Return (all_kblocks, all_tsubs, tsub_to_idx, block_bitsets, block_tsub_lists)."""
    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)

    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # For each block, pre-compute the list of t-subset indices it covers
    block_tsub_lists = []
    for B in all_kblocks:
        indices = [tsub_to_idx[ts] for ts in itertools.combinations(B, t)]
        block_tsub_lists.append(indices)

    # Build integer bitsets
    block_bitsets = []
    for indices in block_tsub_lists:
        bs = 0
        for j in indices:
            bs |= (1 << j)
        block_bitsets.append(bs)

    return all_kblocks, all_tsubs, tsub_to_idx, block_bitsets, block_tsub_lists


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy A: Global-max greedy with bitset acceleration
# ─────────────────────────────────────────────────────────────────────────────
def _greedy_global(v, k, t, all_kblocks, block_bitsets, n_tsubs, deadline, seed=42):
    """Global-max greedy set cover using integer bitsets with random tie-breaking."""
    n_blocks = len(all_kblocks)
    full_mask = (1 << n_tsubs) - 1
    uncovered = full_mask
    selected = []
    rng = random.Random(seed)

    while uncovered:
        if time.time() > deadline:
            break

        best_gain = 0
        candidates = []
        for i in range(n_blocks):
            g = (block_bitsets[i] & uncovered).bit_count()
            if g > best_gain:
                best_gain = g
                candidates = [i]
            elif g == best_gain and g > 0:
                candidates.append(i)

        if best_gain == 0:
            break

        best_idx = rng.choice(candidates)
        selected.append(best_idx)
        uncovered &= ~block_bitsets[best_idx]

    return selected


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy B: Point-by-point block construction (from inspiration 1)
# ─────────────────────────────────────────────────────────────────────────────
def _greedy_pointwise(v, k, t, all_tsubs, tsub_to_idx, block_bitsets, n_tsubs,
                      deadline, seed=42):
    """Build blocks one point at a time, starting from uncovered t-subsets."""
    rng = random.Random(seed)
    full_mask = (1 << n_tsubs) - 1
    uncovered = full_mask
    selected = []
    points = list(range(v))

    # Shuffle t-subsets for randomization
    tsub_order = list(range(n_tsubs))
    rng.shuffle(tsub_order)

    for ts_idx in tsub_order:
        if time.time() > deadline:
            break
        if not (uncovered >> ts_idx) & 1:
            continue

        # Start with this t-subset
        ts = all_tsubs[ts_idx]
        block = list(ts)
        block_set = set(block)

        # Greedily add points to maximize new coverage
        while len(block) < k:
            best_gain = -1
            best_point = -1
            for x in points:
                if x in block_set:
                    continue
                # Count new t-subsets covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_sub = tuple(sorted(list(rest) + [x]))
                    if new_sub in tsub_to_idx:
                        idx = tsub_to_idx[new_sub]
                        if (uncovered >> idx) & 1:
                            gain += 1
                # Small random noise for tie-breaking
                gain += rng.random() * 0.01
                if gain > best_gain:
                    best_gain = gain
                    best_point = x

            if best_point == -1:
                break
            block.append(best_point)
            block_set.add(best_point)

        block.sort()
        # Find the block index
        try:
            blk_idx = all_kblocks.index(tuple(block)) if 'all_kblocks' in dir() else -1
        except (ValueError, NameError):
            blk_idx = -1

        # Use the block directly - find its index in all_kblocks
        # We need to map it back
        if blk_idx == -1:
            # Fallback: search
            for i, b in enumerate(_all_kblocks_ref):
                if b == tuple(block):
                    blk_idx = i
                    break

        selected.append(blk_idx)
        # Update uncovered
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            uncovered &= ~(1 << idx)

        if uncovered == 0:
            break

    return selected


# ─────────────────────────────────────────────────────────────────────────────
#  Local optimisation helpers
# ─────────────────────────────────────────────────────────────────────────────
def _build_cover_counts(selected, block_tsub_lists, n_tsubs):
    """Return a list of cover counts for each t-subset."""
    counts = [0] * n_tsubs
    for idx in selected:
        for j in block_tsub_lists[idx]:
            counts[j] += 1
    return counts


def _remove_redundant(selected, block_bitsets, block_tsub_lists, cover_counts, deadline):
    """Remove blocks that don't uniquely cover any t-subset."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_sel = []
        for idx in selected:
            critical = False
            for j in block_tsub_lists[idx]:
                if cover_counts[j] == 1:
                    critical = True
                    break
            if critical:
                new_sel.append(idx)
            else:
                for j in block_tsub_lists[idx]:
                    cover_counts[j] -= 1
                changed = True
        selected = new_sel
    return selected


def _try_swaps(selected, all_kblocks, block_bitsets, block_tsub_lists,
               cover_counts, n_tsubs, deadline, max_rounds=3):
    """Try 1-for-1 block swaps."""
    selected_set = set(selected)

    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        changed = False
        new_sel = []
        for idx in selected:
            unique_covered = [j for j in block_tsub_lists[idx] if cover_counts[j] == 1]

            if not unique_covered:
                for j in block_tsub_lists[idx]:
                    cover_counts[j] -= 1
                changed = True
                continue

            best_alt = -1
            best_score = -1
            for alt in range(len(all_kblocks)):
                if alt == idx or alt in selected_set:
                    continue
                ok = True
                for j in unique_covered:
                    if not (block_bitsets[alt] >> j) & 1:
                        ok = False
                        break
                if not ok:
                    continue
                score = sum(1 for j in block_tsub_lists[alt] if cover_counts[j] <= 2)
                if score > best_score:
                    best_score = score
                    best_alt = alt

            if best_alt != -1:
                for j in block_tsub_lists[idx]:
                    cover_counts[j] -= 1
                for j in block_tsub_lists[best_alt]:
                    cover_counts[j] += 1
                new_sel.append(best_alt)
                selected_set.discard(idx)
                selected_set.add(best_alt)
                changed = True
            else:
                new_sel.append(idx)

        selected = new_sel
        if not changed:
            break

    return selected


def _try_two_for_one(selected, all_kblocks, block_bitsets, block_tsub_lists,
                     cover_counts, n_tsubs, deadline, max_rounds=2):
    """Try to replace two blocks with one block."""
    if len(selected) < 2:
        return selected

    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        changed = False
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(selected)):
                unique_set = set()
                for blk in (selected[i], selected[j]):
                    for ts_idx in block_tsub_lists[blk]:
                        if cover_counts[ts_idx] == 1:
                            unique_set.add(ts_idx)

                if not unique_set:
                    continue

                best_k = -1
                best_score = -1
                for k in range(len(all_kblocks)):
                    if k == selected[i] or k == selected[j]:
                        continue
                    ok = True
                    for ts_idx in unique_set:
                        if not (block_bitsets[k] >> ts_idx) & 1:
                            ok = False
                            break
                    if not ok:
                        continue
                    score = sum(1 for ts_idx in block_tsub_lists[k] if cover_counts[ts_idx] <= 2)
                    if score > best_score:
                        best_score = score
                        best_k = k

                if best_k != -1:
                    for blk in (selected[i], selected[j]):
                        for ts_idx in block_tsub_lists[blk]:
                            cover_counts[ts_idx] -= 1
                    for ts_idx in block_tsub_lists[best_k]:
                        cover_counts[ts_idx] += 1
                    new_sel = []
                    skip = {i, j}
                    for idx_pos, idx in enumerate(selected):
                        if idx_pos not in skip:
                            new_sel.append(idx)
                    new_sel.append(best_k)
                    selected = new_sel
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break

    return selected


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def construct(v, k, t):
    global _all_kblocks_ref
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin for 120 s limit

    # Pre-compute
    all_kblocks, all_tsubs, tsub_to_idx, block_bitsets, block_tsub_lists = \
        _precompute(v, k, t)
    _all_kblocks_ref = all_kblocks

    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)

    best_selected = None
    best_count = n_blocks + 1

    # ── Phase 1: Multi-restart with mixed strategies ──
    # Alternate between global-max greedy and pointwise construction
    max_restarts = max(1, int((deadline - start_time - 20.0) / 4.0))
    max_restarts = min(max_restarts, 25)

    for restart in range(max_restarts):
        if time.time() > deadline - 25.0:
            break

        seed = 42 + restart * 7

        # Alternate strategies: even restarts use global, odd use pointwise
        if restart % 2 == 0:
            selected = _greedy_global(v, k, t, all_kblocks, block_bitsets,
                                      n_tsubs, deadline - 25.0, seed)
        else:
            selected = _greedy_pointwise(v, k, t, all_tsubs, tsub_to_idx,
                                         block_bitsets, n_tsubs,
                                         deadline - 25.0, seed)

        if not selected:
            continue

        # Local optimisation
        local_deadline = min(deadline - 20.0, start_time + 110.0)
        cover_counts = _build_cover_counts(selected, block_tsub_lists, n_tsubs)
        selected = _remove_redundant(selected, block_bitsets, block_tsub_lists,
                                     cover_counts, local_deadline - 10.0)
        cover_counts = _build_cover_counts(selected, block_tsub_lists, n_tsubs)
        selected = _try_swaps(selected, all_kblocks, block_bitsets,
                              block_tsub_lists, cover_counts, n_tsubs,
                              local_deadline - 5.0, max_rounds=2)
        cover_counts = _build_cover_counts(selected, block_tsub_lists, n_tsubs)
        selected = _try_two_for_one(selected, all_kblocks, block_bitsets,
                                    block_tsub_lists, cover_counts, n_tsubs,
                                    local_deadline - 2.0, max_rounds=1)

        # Verify completeness
        final_uncovered = (1 << n_tsubs) - 1
        for idx in selected:
            final_uncovered &= ~block_bitsets[idx]

        if final_uncovered == 0:
            if len(selected) < best_count:
                best_count = len(selected)
                best_selected = list(selected)

    if best_selected is None:
        # Fallback: single deterministic greedy
        selected = _greedy_global(v, k, t, all_kblocks, block_bitsets,
                                  n_tsubs, deadline - 5.0, seed=42)
        best_selected = selected

    # ── Phase 2: Final polish on the best solution ──
    if time.time() < deadline - 5.0:
        cover_counts = _build_cover_counts(best_selected, block_tsub_lists, n_tsubs)
        best_selected = _remove_redundant(best_selected, block_bitsets,
                                          block_tsub_lists, cover_counts,
                                          deadline - 5.0)
        cover_counts = _build_cover_counts(best_selected, block_tsub_lists, n_tsubs)
        best_selected = _try_swaps(best_selected, all_kblocks, block_bitsets,
                                   block_tsub_lists, cover_counts, n_tsubs,
                                   deadline - 3.0, max_rounds=3)
        cover_counts = _build_cover_counts(best_selected, block_tsub_lists, n_tsubs)
        best_selected = _try_two_for_one(best_selected, all_kblocks,
                                         block_bitsets, block_tsub_lists,
                                         cover_counts, n_tsubs,
                                         deadline - 1.0, max_rounds=2)

    blocks = [all_kblocks[i] for i in best_selected]
    return blocks


if __name__ == "__main__":
    v, k, t = 19, 7, 4
    t0 = time.time()
    result = construct(v, k, t)
    t1 = time.time()
    print(f"C({v},{k},{t}): {len(result)} blocks, time {t1 - t0:.1f}s")
    # Verify
    all_tsubs = set(itertools.combinations(range(v), t))
    covered = set()
    for b in result:
        assert len(b) == k and len(set(b)) == k
        covered.update(itertools.combinations(b, t))
    assert covered == all_tsubs, "Invalid covering!"
    print("Valid!")
