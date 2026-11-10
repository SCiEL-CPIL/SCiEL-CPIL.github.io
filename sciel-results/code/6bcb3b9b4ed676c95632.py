# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112  # safety margin within 150s limit

    def now():
        return time.time()

    def remaining():
        return deadline - now()

    # ---- Precompute all t-subsets and block coverage bitmasks ----
    all_tsubsets = list(itertools.combinations(range(v), t))
    nts = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # For each (t-1)-subset, list the indices of t-subsets it extends (by one point)
    ext = defaultdict(list)
    for i, ts in enumerate(all_tsubsets):
        for j in range(t):
            key = ts[:j] + ts[j+1:]
            ext[key].append(i)

    # Precompute coverage bitmask for every k-block (as frozenset)
    block_mask = {}
    for b in itertools.combinations(range(v), k):
        m = 0
        for ts in itertools.combinations(b, t):
            m |= 1 << tsubset_to_idx[ts]
        block_mask[frozenset(b)] = m

    def block_mask_of(block):
        return block_mask[frozenset(block)]

    # ---- Greedy construction (vectorized) ----
    def greedy_build(seed):
        rng = random.Random(seed)
        covered = 0
        blocks = []
        covered_set = set()
        all_points = list(range(v))

        while True:
            if remaining() < 2:
                break
            # pick a random uncovered t-subset
            while True:
                if remaining() < 2:
                    return None
                ts = rng.choice(all_tsubsets)
                if ts not in covered_set:
                    break

            # build block starting from ts
            block = list(ts)
            block_set = set(block)
            remaining_points = [x for x in all_points if x not in block_set]

            while len(block) < k:
                if remaining() < 2:
                    return None
                best_x = None
                best_gain = -1
                # vectorize gain over candidate points
                for x in remaining_points:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts2 = tuple(sorted(rest + (x,)))
                        if ts2 not in covered_set:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
                block_set.add(best_x)
                remaining_points.remove(best_x)

            block.sort()
            m = block_mask_of(block)
            new_bits = m & ~covered
            if new_bits:
                blocks.append(block)
                covered |= m
                covered_set = set()
                # rebuild covered_set lazily is expensive; use bit test instead
                # We'll track covered via bits; covered_set replaced by bit check.
                # To keep it simple, maintain covered as int and drop covered_set.
                break
            else:
                # block adds nothing; skip (shouldn't happen often)
                pass

        # The above loop got tangled; redo cleanly below.
        return None

    # Cleaner greedy using bit coverage only
    def greedy_build2(seed):
        rng = random.Random(seed)
        covered = 0
        blocks = []
        all_points = list(range(v))

        while True:
            if remaining() < 2:
                break
            # find an uncovered t-subset (random scan)
            while True:
                if remaining() < 2:
                    return None
                ts = rng.choice(all_tsubsets)
                idx = tsubset_to_idx[ts]
                if not (covered >> idx) & 1:
                    break

            block = list(ts)
            block_set = set(block)
            remaining_points = [x for x in all_points if x not in block_set]

            while len(block) < k:
                if remaining() < 2:
                    return None
                best_x = None
                best_gain = -1
                for x in remaining_points:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts2 = tuple(sorted(rest + (x,)))
                        idx2 = tsubset_to_idx[ts2]
                        if not (covered >> idx2) & 1:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
                block_set.add(best_x)
                remaining_points.remove(best_x)

            block.sort()
            m = block_mask_of(block)
            blocks.append(block)
            covered |= m
            if covered == (1 << nts) - 1:
                break

        if covered == (1 << nts) - 1:
            return blocks
        return None

    # ---- Local search: remove redundant + swap ----
    def local_search(blocks, max_time):
        start_ls = now()
        # coverage count per t-subset via bitmasks is hard for counts; use counter
        def cover_counts(blks):
            cc = defaultdict(int)
            for b in blks:
                for ts in itertools.combinations(b, t):
                    cc[ts] += 1
            return cc

        cc = cover_counts(blocks)
        changed = True
        while changed and (now() - start_ls) < max_time:
            changed = False
            new_blocks = []
            for i, block in enumerate(blocks):
                critical = False
                for ts in itertools.combinations(block, t):
                    if cc[ts] == 1:
                        critical = True
                        break
                if not critical:
                    for ts in itertools.combinations(block, t):
                        cc[ts] -= 1
                    changed = True
                else:
                    new_blocks.append(block)
            blocks = new_blocks

        # swap pass: remove one block, re-add best block
        improved = True
        while improved and (now() - start_ls) < max_time:
            improved = False
            for i in range(len(blocks)):
                if now() - start_ls > max_time:
                    break
                saved = blocks.pop(i)
                cc = cover_counts(blocks)
                # find uncovered
                uncovered = [ts for ts, c in cc.items() if c == 0]
                if not uncovered:
                    blocks.insert(i, saved)
                    improved = True
                    continue
                # find best block to add
                best_block = None
                best_gain = -1
                # candidate blocks: extend each uncovered subset
                candidates = set()
                for ts in uncovered[:50]:
                    block = list(ts)
                    block_set = set(block)
                    rem = [x for x in range(v) if x not in block_set]
                    while len(block) < k:
                        best_x = None
                        bg = -1
                        for x in rem:
                            g = 0
                            for rest in itertools.combinations(block, t - 1):
                                ts2 = tuple(sorted(rest + (x,)))
                                if ts2 in uncovered:
                                    g += 1
                            if g > bg:
                                bg = g
                                best_x = x
                        if best_x is None:
                            break
                        block.append(best_x)
                        block_set.add(best_x)
                        rem.remove(best_x)
                    block.sort()
                    candidates.add(frozenset(block))
                for c in candidates:
                    m = block_mask[c]
                    gain = sum(1 for ts in uncovered if (m >> tsubset_to_idx[ts]) & 1)
                    if gain > best_gain:
                        best_gain = gain
                        best_block = list(c)
                if best_block is not None:
                    blocks.append(best_block)
                    improved = True

        return blocks

    # ---- Validation ----
    def validate_covering(blocks):
        covered = 0
        for b in blocks:
            covered |= block_mask_of(b)
        return covered == (1 << nts) - 1

    # ---- Main search: multiple greedy restarts + local search ----
    best_result = None
    best_len = float('inf')

    # Phase 1: several greedy restarts, local search each
    n_restarts = 12
    for seed in range(n_restarts):
        if remaining() < 8:
            break
        t0 = now()
        result = greedy_build2(seed)
        if result is None:
            continue
        if not validate_covering(result):
            continue
        # local search with time budget
        budget = min(4, max(0.5, remaining() - 6))
        result = local_search(result, budget)
        if validate_covering(result) and len(result) < best_len:
            best_len = len(result)
            best_result = result

    # Phase 2: refine the best further
    if best_result is not None and remaining() > 6:
        budget = max(0.5, remaining() - 4)
        refined = local_search(list(best_result), budget)
        if validate_covering(refined) and len(refined) < best_len:
            best_len = len(refined)
            best_result = refined

    if best_result is not None:
        return best_result

    # Fallback: return a single greedy result even if not fully covered is invalid;
    # but we must return something. Return empty list would be invalid.
    # Do a last-ditch greedy.
    return greedy_build2(0) or []

# EVOLVE-BLOCK-END
