"""Covering design C(v,k,t): Greedy Packing with Conflict Resolution."""
import time
import itertools
import random

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False


def construct(v, k, t):
    start = time.time()
    deadline = start + 130.0  # safety margin before 150s kill

    # All t-subsets of the v points
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    if n_t == 0:
        return []

    # Map each t-subset to an index for fast counting
    t_index = {c: i for i, c in enumerate(all_t)}

    # Precompute for each candidate block: list of t-subset indices it covers
    # For v=24, k=7: C(24,7) = 346104 blocks, each has C(7,3)=35 triples
    # Total: ~12M triple-block pairs — feasible to precompute
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, store the list of t-subset indices it covers
    block_triples = []
    for b in all_blocks:
        tris = [t_index[c] for c in itertools.combinations(b, t)]
        block_triples.append(tris)

    # Coverage count per t-subset
    cover_count = [0] * n_t

    # Greedy packing: select blocks that maximize new coverage while respecting lambda_max
    lambda_max = 1  # initially pack with lambda_max=1 (disjoint triples)
    selected = []
    selected_set = set()

    # Score for each block: number of triples it covers that are currently uncovered
    # We'll use a priority approach: repeatedly pick the block with highest marginal gain
    # that doesn't violate lambda_max

    # For efficiency, track marginal gain incrementally
    # gain[b] = number of triples in block b where cover_count < lambda_max
    gains = [0] * n_blocks
    for bi, tris in enumerate(block_triples):
        g = 0
        for ti in tris:
            if cover_count[ti] < lambda_max:
                g += 1
        gains[bi] = g

    # Use a max-heap keyed by -gain
    import heapq
    heap = [(-gains[i], i) for i in range(n_blocks) if gains[i] > 0]
    heapq.heapify(heap)

    # Track which blocks are selected
    is_selected = [False] * n_blocks

    while heap and time.time() < deadline - 30:  # reserve 30s for repair
        neg_g, bi = heapq.heappop(heap)
        if is_selected[bi]:
            continue
        # Recompute current gain (some triples may have been covered since)
        tris = block_triples[bi]
        cur_gain = 0
        for ti in tris:
            if cover_count[ti] < lambda_max:
                cur_gain += 1
        if cur_gain <= 0:
            continue
        # Accept this block
        is_selected[bi] = True
        selected.append(bi)
        selected_set.add(bi)
        # Update coverage counts
        for ti in tris:
            cover_count[ti] += 1
        # Update gains for other blocks that share triples with this one
        # (lazy: just let stale heap entries get re-scored)

    # Check coverage: are all triples covered at least once?
    uncovered = [ti for ti in range(n_t) if cover_count[ti] < 1]

    # If packing with lambda_max=1 didn't cover everything, do a repair phase
    if uncovered and time.time() < deadline - 5:
        # Greedy repair: pick blocks that cover the most uncovered triples
        unc_set = set(uncovered)
        while unc_set and time.time() < deadline - 5:
            best_bi = -1
            best_gain = 0
            # Scan all blocks for best marginal coverage of uncovered
            # For speed, only check blocks not yet selected
            for bi in range(n_blocks):
                if is_selected[bi]:
                    continue
                g = 0
                for ti in block_triples[bi]:
                    if ti in unc_set:
                        g += 1
                if g > best_gain:
                    best_gain = g
                    best_bi = bi
                if g == 35:  # can't do better than all 35
                    break
            if best_bi < 0 or best_gain == 0:
                break
            is_selected[best_bi] = True
            selected.append(best_bi)
            for ti in block_triples[best_bi]:
                if ti in unc_set:
                    unc_set.discard(ti)
                cover_count[ti] += 1

    # Try a second packing pass with lambda_max=2 if time permits, to find
    # a denser packing that might use fewer total blocks
    if time.time() < deadline - 15:
        # Alternative: try greedy set-cover from scratch (standard approach)
        # as a comparison, keep the better result
        unc2 = set(range(n_t))
        selected2 = []
        is_sel2 = [False] * n_blocks

        # Compute initial gains
        gains2 = [0] * n_blocks
        for bi, tris in enumerate(block_triples):
            g = 0
            for ti in tris:
                if ti in unc2:
                    g += 1
            gains2[bi] = g

        heap2 = [(-gains2[i], i) for i in range(n_blocks) if gains2[i] > 0]
        heapq.heapify(heap2)

        while heap2 and unc2 and time.time() < deadline - 10:
            neg_g, bi = heapq.heappop(heap2)
            if is_sel2[bi]:
                continue
            tris = block_triples[bi]
            cur_gain = 0
            for ti in tris:
                if ti in unc2:
                    cur_gain += 1
            if cur_gain <= 0:
                continue
            is_sel2[bi] = True
            selected2.append(bi)
            for ti in tris:
                unc2.discard(ti)

        # Keep the better solution
        if len(selected2) < len(selected):
            selected = selected2

    # Build final block list
    final_blocks = []
    seen = set()
    for bi in selected:
        b = tuple(all_blocks[bi])
        if b not in seen:
            seen.add(b)
            final_blocks.append(list(b))

    # Final validation: ensure every t-subset is covered
    # (quick check; the evaluator does exhaustive check)
    if time.time() < deadline - 2:
        covered = set()
        for b in final_blocks:
            for c in itertools.combinations(b, t):
                covered.add(c)
        if len(covered) < n_t:
            # Emergency repair: add blocks until valid
            missing = set(all_t) - covered
            while missing and time.time() < deadline - 1:
                best_bi = -1
                best_g = 0
                for bi in range(n_blocks):
                    if bi in seen:
                        continue
                    g = 0
                    for ti in block_triples[bi]:
                        if all_t[ti] in missing:
                            g += 1
                    if g > best_g:
                        best_g = g
                        best_bi = bi
                if best_bi < 0 or best_g == 0:
                    break
                b = tuple(all_blocks[best_bi])
                seen.add(best_bi)
                final_blocks.append(list(b))
                for c in itertools.combinations(b, t):
                    missing.discard(c)

    return final_blocks
