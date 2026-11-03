import itertools
import time
import random
import numpy as np
import heapq

def construct(v, k, t):
    deadline = time.time() + 115.0

    # ---- Precomputation ----
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)

    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # block_tsubs[bi] = list of t-subset indices covered by block bi
    block_tsubs = [None] * n_blocks
    # tsub_blocks[ts] = list of block indices covering t-subset ts
    tsub_blocks = [[] for _ in range(n_tsubsets)]

    for bi, block in enumerate(all_blocks):
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs[bi] = tsubs
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)

    # Incidence matrix: M[bi, ts] = 1 if block bi covers t-subset ts
    M = np.zeros((n_blocks, n_tsubsets), dtype=np.int8)
    for bi in range(n_blocks):
        for ts_idx in block_tsubs[bi]:
            M[bi, ts_idx] = 1

    # tsub_block_sets for fast intersection in local search
    tsub_block_sets = [set(lst) for lst in tsub_blocks]

    # ---- Vectorized greedy cover ----
    def greedy_cover(seed):
        rng = random.Random(seed)
        # uncovered_mask: 1 = uncovered, 0 = covered
        uncovered_mask = np.ones(n_tsubsets, dtype=np.int8)
        # gain[bi] = number of uncovered t-subsets in block bi
        gain = M.dot(uncovered_mask)  # int8 vector, values 0..C(k,t)
        # Track which blocks have been selected (to exclude)
        selected_mask = np.zeros(n_blocks, dtype=np.int8)

        selected = []

        while True:
            if time.time() > deadline - 8:
                break

            # Find best block: max gain among unselected
            # Mask out selected blocks
            effective_gain = gain.copy()
            effective_gain[selected_mask.astype(bool)] = -1
            max_gain = int(effective_gain.max())
            if max_gain <= 0:
                break

            # Random tie-breaking among blocks with max gain
            candidates = np.where(effective_gain == max_gain)[0]
            bi = int(candidates[rng.randint(0, len(candidates) - 1)])

            selected.append(bi)
            selected_mask[bi] = 1

            # Update: mark t-subsets covered by bi as covered
            new_covered = block_tsubs[bi]
            uncovered_mask[new_covered] = 0

            # Update gain: for all blocks sharing t-subsets with bi, decrement
            # gain[bj] -= (number of newly-covered t-subsets in block bj)
            # = M[:, new_covered].sum(axis=1)
            decrement = M[:, new_covered].sum(axis=1)
            gain -= decrement

        return selected

    # ---- Local search: redundancy removal + 1-for-1 swaps ----
    def local_search(blocks, seed):
        rng = random.Random(seed)
        block_set = set(blocks)
        cover_count = np.zeros(n_tsubsets, dtype=np.int16)

        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1

        improved = True
        stall_count = 0
        max_stalls = 3

        while improved and time.time() < deadline - 3:
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
                stall_count = 0
                continue

            # Phase 2: Try 1-for-1 swaps
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            for bi in current_blocks:
                if time.time() > deadline - 3:
                    break
                if bi not in block_set:
                    continue

                unique_tsubs = [ts_idx for ts_idx in block_tsubs[bi] if cover_count[ts_idx] == 1]

                if not unique_tsubs:
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break

                # Find candidates covering all unique_tsubs
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
                    stall_count = 0
                    break

            if not improved:
                stall_count += 1
                if stall_count >= max_stalls:
                    break

        return list(block_set)

    # ---- 2-for-1 swap search ----
    def two_block_swap_search(blocks, seed):
        rng = random.Random(seed)
        block_set = set(blocks)
        cover_count = np.zeros(n_tsubsets, dtype=np.int16)

        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1

        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            n_checked = 0
            for i, bi in enumerate(current_blocks):
                if time.time() > deadline - 3:
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
                if n_checked > 2000:
                    break

                # Required t-subsets: unique to bi or bj, plus intersection with count==2
                req_bi = [ts for ts in block_tsubs[bi] if cover_count[ts] == 1]
                req_bj = [ts for ts in block_tsubs[bj] if cover_count[ts] == 1]

                set_bi = set(block_tsubs[bi])
                set_bj = set(block_tsubs[bj])
                inter = set_bi & set_bj

                required = set(req_bi)
                required.update(req_bj)
                for ts in inter:
                    if cover_count[ts] == 2:
                        required.add(ts)

                if not required:
                    continue

                # Find candidates covering all required
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

    # ---- 3-for-2 swap search (new) ----
    def three_for_two_swap_search(blocks, seed):
        """Try to replace 3 blocks with 2 blocks, reducing count by 1."""
        rng = random.Random(seed)
        block_set = set(blocks)
        if len(block_set) < 3:
            return list(block_set)

        cover_count = np.zeros(n_tsubsets, dtype=np.int16)
        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1

        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            n_checked = 0
            for i in range(len(current_blocks)):
                if time.time() > deadline - 3:
                    break
                bi = current_blocks[i]
                if bi not in block_set:
                    continue
                if len(block_set) < 3:
                    break

                others = [b for b in current_blocks[i+1:] if b in block_set]
                if len(others) < 2:
                    continue

                # Pick two random partners
                j1 = rng.randint(0, len(others) - 1)
                bj = others[j1]
                others2 = others[:j1] + others[j1+1:]
                if not others2:
                    continue
                j2 = rng.randint(0, len(others2) - 1)
                bk = others2[j2]

                n_checked += 1
                if n_checked > 1500:
                    break

                # Required t-subsets: those with cover_count == 1 (unique to one of the three)
                # or cover_count == 2 where the covering set is subset of {bi, bj, bk}
                # Simplification: required = t-subsets with cover_count <= 2 that are in
                # the union of the three blocks, minus those still covered by others
                set_bi = set(block_tsubs[bi])
                set_bj = set(block_tsubs[bj])
                set_bk = set(block_tsubs[bk])
                union = set_bi | set_bj | set_bk

                required = set()
                for ts in union:
                    cc = cover_count[ts]
                    if cc == 1:
                        required.add(ts)
                    elif cc == 2:
                        # Check if both covering blocks are in {bi, bj, bk}
                        # This is expensive; approximate by checking if ts is in at least 2 of the 3
                        in_bi = 1 if ts in set_bi else 0
                        in_bj = 1 if ts in set_bj else 0
                        in_bk = 1 if ts in set_bk else 0
                        if in_bi + in_bj + in_bk >= 2:
                            required.add(ts)

                if not required:
                    continue

                # We need to find 2 blocks that together cover all of `required`
                # Strategy: enumerate pairs of candidate blocks from the intersection
                # of tsub_blocks for a rare required t-subset
                min_ts = min(required, key=lambda x: len(tsub_blocks[x]))
                base_candidates = tsub_block_sets[min_ts]

                # For each pair of candidates, check if they cover all required
                # Limit to a small number of pairs
                cand_list = list(base_candidates)
                cand_list = [c for c in cand_list if c not in (bi, bj, bk)]
                rng.shuffle(cand_list)
                cand_list = cand_list[:40]  # Limit candidates

                found = False
                for idx1 in range(len(cand_list)):
                    c1 = cand_list[idx1]
                    # Check c1 covers how many required
                    set_c1 = set(block_tsubs[c1])
                    remaining_after_c1 = required - set_c1
                    if not remaining_after_c1:
                        # c1 alone covers all; pair with any other block
                        c2 = c1  # degenerate: use c1 twice? No, need 2 distinct
                        # Actually if c1 covers all, we just need one more block (any)
                        # But we need exactly 2 blocks total. Use c1 and any valid block.
                        # For simplicity, skip this degenerate case
                        continue

                    for idx2 in range(idx1 + 1, len(cand_list)):
                        c2 = cand_list[idx2]
                        set_c2 = set(block_tsubs[c2])
                        # Check if c1 and c2 together cover all required
                        if remaining_after_c1.issubset(set_c2):
                            # Valid pair found
                            # Perform the swap
                            block_set.remove(bi)
                            for ts in block_tsubs[bi]:
                                cover_count[ts] -= 1
                            block_set.remove(bj)
                            for ts in block_tsubs[bj]:
                                cover_count[ts] -= 1
                            block_set.remove(bk)
                            for ts in block_tsubs[bk]:
                                cover_count[ts] -= 1

                            block_set.add(c1)
                            for ts in block_tsubs[c1]:
                                cover_count[ts] += 1
                            block_set.add(c2)
                            for ts in block_tsubs[c2]:
                                cover_count[ts] += 1

                            improved = True
                            found = True
                            break
                    if found:
                        break

                if found:
                    break

        return list(block_set)

    # ---- Main search loop ----
    best_blocks = None
    best_len = float('inf')

    num_trials = 60
    for i in range(num_trials):
        if time.time() > deadline - 20:
            break

        blocks = greedy_cover(i)
        if not blocks:
            continue

        blocks = local_search(blocks, i)

        if time.time() < deadline - 10:
            blocks = two_block_swap_search(blocks, i + 1000)

        if time.time() < deadline - 5:
            blocks = three_for_two_swap_search(blocks, i + 2000)

        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)

    # Final polish on best solution
    if best_blocks is not None and time.time() < deadline - 5:
        best_blocks = local_search(best_blocks, 99999)
        if time.time() < deadline - 3:
            best_blocks = two_block_swap_search(best_blocks, 88888)
        if time.time() < deadline - 2:
            best_blocks = three_for_two_swap_search(best_blocks, 77777)

    if best_blocks is None:
        best_blocks = greedy_cover(0)

    return [all_blocks[i] for i in best_blocks]
