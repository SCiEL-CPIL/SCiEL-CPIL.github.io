"""Covering design C(19,7,4): vectorized greedy + fast count-based local search."""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 142.0
    t0 = time.time()

    # --- Precomputation -------------------------------------------------------
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    c_k_t = 1
    for i in range(t):
        c_k_t = c_k_t * (k - i) // (i + 1)

    # For each block, the flat list of t-subset indices it covers.
    block_tsubs = np.empty((n_blocks, c_k_t), dtype=np.int32)
    for bi, block in enumerate(all_blocks):
        block_tsubs[bi] = [tsub_idx[ts] for ts in itertools.combinations(block, t)]

    # Inverse index: for each t-subset, the blocks covering it.
    tsub_block_counts = np.zeros(n_tsubsets, dtype=np.int32)
    np.add.at(tsub_block_counts, block_tsubs.ravel(), 1)
    tsub_blocks = [None] * n_tsubsets
    for i in range(n_tsubsets):
        tsub_blocks[i] = np.empty(tsub_block_counts[i], dtype=np.int32)
    pos = np.zeros(n_tsubsets, dtype=np.int32)
    for bi in range(n_blocks):
        for ts in block_tsubs[bi]:
            p = pos[ts]
            tsub_blocks[ts][p] = bi
            pos[ts] = p + 1

    # Flat arrays for vectorized gain updates.
    block_ids_flat = np.repeat(np.arange(n_blocks, dtype=np.int32), c_k_t)
    tsub_flat = block_tsubs.ravel()

    # --- Greedy (vectorized) --------------------------------------------------
    def greedy_cover(seed=0, diversify=False):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        blocks_used = []
        used = np.zeros(n_blocks, dtype=bool)
        gain = np.zeros(n_blocks, dtype=np.int32)
        mask = uncovered[tsub_flat]
        if mask.any():
            np.add.at(gain, block_ids_flat[mask], 1)
        while uncovered.any():
            if time.time() > deadline:
                break
            max_gain = gain.max()
            if max_gain <= 0:
                break
            cands = np.where((gain == max_gain) & (~used))[0]
            if cands.size == 0:
                cands = np.where(gain == max_gain)[0]
            if diversify and cands.size > 1:
                # temperature-style: prefer but allow lower-gain picks
                if rng.random() < 0.15:
                    chosen = cands[rng.randrange(cands.size)]
                else:
                    chosen = cands[rng.randrange(cands.size)]
            else:
                chosen = cands[rng.randrange(cands.size)]
            blocks_used.append(all_blocks[chosen])
            used[chosen] = True
            chosen_tsubs = block_tsubs[chosen]
            newly = chosen_tsubs[uncovered[chosen_tsubs]]
            if newly.size:
                uncovered[newly] = False
                all_aff = np.concatenate([tsub_blocks[x] for x in newly])
                np.add.at(gain, all_aff, -1)
            gain[chosen] = -1
        return blocks_used

    # --- Fast count-based redundancy removal (cached) -------------------------
    def remove_redundant(blocks, tsub_lists=None, cover_count=None):
        n = len(blocks)
        if n == 0:
            return blocks, [], np.zeros(n_tsubsets, dtype=np.int32)
        if tsub_lists is None:
            tsub_lists = [block_tsubs[i].tolist() for i, _ in enumerate(blocks)]
        if cover_count is None:
            cover_count = np.zeros(n_tsubsets, dtype=np.int32)
            for tl in tsub_lists:
                for idx in tl:
                    cover_count[idx] += 1
        kept = np.ones(n, dtype=bool)
        # cached unique-count per block (number of tsubsets it uniquely covers)
        uniq_count = np.empty(n, dtype=np.int32)
        for i in range(n):
            uniq_count[i] = np.sum(cover_count[np.array(tsub_lists[i], dtype=np.int32)] == 1)
        # Repeatedly drop blocks that are fully redundant (uniq_count == 0)
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if not kept[i]:
                    continue
                if uniq_count[i] == 0:
                    kept[i] = False
                    for idx in tsub_lists[i]:
                        cover_count[idx] -= 1
                        if cover_count[idx] == 1:
                            # some block now uniquely covers idx; find which kept block
                            # cheap: increment uniq_count lazily handled below
                            pass
                    # refresh uniq_count for affected blocks lazily
                    for j in range(n):
                        if not kept[j]:
                            continue
                        arr = tsub_lists[j]
                        # recompute cheaply only if it touched removed idx
                        uniq_count[j] = np.sum(cover_count[np.array(arr, dtype=np.int32)] == 1)
                    changed = True
        new_blocks = [blocks[i] for i in range(n) if kept[i]]
        new_tsub = [tsub_lists[i] for i in range(n) if kept[i]]
        return new_blocks, new_tsub, cover_count

    # --- Local search: 1-for-1 swaps + 2-for-1 merges -------------------------
    def local_search(blocks, seed=0, max_iterations=200):
        if not blocks:
            return blocks
        rng = random.Random(seed)
        reduced = list(blocks)
        tsub_lists = [block_tsubs[all_blocks.index(b)].tolist() if b in all_blocks else
                      [tsub_idx[ts] for ts in itertools.combinations(b, t)] for b in reduced]
        # build a fast lookup block->index in all_blocks
        bmap = {b: i for i, b in enumerate(all_blocks)}
        tsub_lists = [block_tsubs[bmap[b]].tolist() for b in reduced]
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        for tl in tsub_lists:
            for idx in tl:
                cover_count[idx] += 1
        used_blocks = set(reduced)

        for it in range(max_iterations):
            if time.time() > deadline - 2:
                break
            improved = False
            n = len(reduced)
            if n <= 1:
                break
            order = list(range(n))
            rng.shuffle(order)

            # 1-for-1: remove redundant, then swap to reduce unique burden
            for i in order:
                if time.time() > deadline - 2:
                    break
                if i >= len(reduced):
                    continue
                tl = tsub_lists[i]
                arr = np.array(tl, dtype=np.int32)
                unique_tsubs = arr[cover_count[arr] == 1]
                if unique_tsubs.size == 0:
                    # redundant: drop
                    for idx in tl:
                        cover_count[idx] -= 1
                    reduced.pop(i)
                    tsub_lists.pop(i)
                    used_blocks.discard(reduced and None)
                    improved = True
                    continue
                # find candidate replacement covering all unique_tsubs
                cand = set()
                for ts in unique_tsubs:
                    for bi in tsub_blocks[ts]:
                        if all_blocks[bi] not in used_blocks:
                            cand.add(bi)
                if not cand:
                    continue
                cand_list = list(cand)
                rng.shuffle(cand_list)
                found = False
                for ci in cand_list[:40]:
                    cset = set(block_tsubs[ci].tolist())
                    if not all(ts in cset for ts in unique_tsubs):
                        continue
                    ok = True
                    for idx in tl:
                        if cover_count[idx] == 1 and idx not in cset:
                            ok = False
                            break
                    if ok:
                        for idx in tl:
                            cover_count[idx] -= 1
                        for idx in block_tsubs[ci]:
                            cover_count[idx] += 1
                        used_blocks.discard(reduced[i])
                        reduced[i] = all_blocks[ci]
                        used_blocks.add(reduced[i])
                        tsub_lists[i] = block_tsubs[ci].tolist()
                        found = True
                        improved = True
                        break
                if found:
                    continue

            if not improved:
                # 2-for-1 merges
                if len(reduced) > 3 and time.time() < deadline - 5:
                    n = len(reduced)
                    found_pair = False
                    for i in range(n):
                        if time.time() > deadline - 2 or found_pair:
                            break
                        ai = np.array(tsub_lists[i], dtype=np.int32)
                        ui = ai[cover_count[ai] == 1]
                        if ui.size == 0:
                            continue
                        for j in range(i + 1, n):
                            if time.time() > deadline - 2 or found_pair:
                                break
                            aj = np.array(tsub_lists[j], dtype=np.int32)
                            uj = aj[cover_count[aj] == 1]
                            if uj.size == 0:
                                continue
                            allu = set(ui.tolist()) | set(uj.tolist())
                            cand = set()
                            for ts in allu:
                                for bi in tsub_blocks[ts]:
                                    if all_blocks[bi] not in used_blocks:
                                        cand.add(bi)
                            if not cand:
                                continue
                            for ci in list(cand)[:25]:
                                if time.time() > deadline - 2:
                                    break
                                cset = set(block_tsubs[ci].tolist())
                                if not all(ts in cset for ts in allu):
                                    continue
                                ok = True
                                for idx in list(ai) + list(aj):
                                    if cover_count[idx] == 1 and idx not in cset:
                                        ok = False
                                        break
                                if ok:
                                    for idx in tsub_lists[i]:
                                        cover_count[idx] -= 1
                                    for idx in tsub_lists[j]:
                                        cover_count[idx] -= 1
                                    for idx in block_tsubs[ci]:
                                        cover_count[idx] += 1
                                    used_blocks.discard(reduced[i])
                                    used_blocks.discard(reduced[j])
                                    reduced[i] = all_blocks[ci]
                                    used_blocks.add(reduced[i])
                                    tsub_lists[i] = block_tsubs[ci].tolist()
                                    reduced.pop(j)
                                    tsub_lists.pop(j)
                                    found_pair = True
                                    improved = True
                                    break
                            if found_pair:
                                break
                    if found_pair:
                        continue

            if not improved:
                # add a block that covers many singly-covered tsubsets, then prune
                if len(reduced) > 2 and time.time() < deadline - 5:
                    single = np.where(cover_count == 1)[0]
                    if single.size:
                        best_gain = 0
                        best_block = -1
                        for bi in range(n_blocks):
                            if all_blocks[bi] in used_blocks:
                                continue
                            cnt = np.sum(cover_count[block_tsubs[bi]] == 1)
                            if cnt > best_gain:
                                best_gain = cnt
                                best_block = bi
                        if best_gain >= 2:
                            trial = reduced + [all_blocks[best_block]]
                            trial, t2, c2 = remove_redundant(trial)
                            if len(trial) < len(reduced):
                                reduced = trial
                                tsub_lists = t2
                                cover_count = c2
                                used_blocks = set(reduced)
                                improved = True
            if not improved:
                break
        return reduced

    # --- Main loop ------------------------------------------------------------
    best_blocks = None
    best_len = float('inf')
    num_trials = 80
    for seed in range(num_trials):
        if time.time() > deadline - 12:
            break
        div = (seed % 4 == 0)
        blocks = greedy_cover(seed, diversify=div)
        if not blocks:
            continue
        blocks = local_search(blocks, seed, max_iterations=150)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            if time.time() < deadline - 25:
                deep = local_search(best_blocks, seed + 1000, max_iterations=300)
                if len(deep) < best_len:
                    best_len = len(deep)
                    best_blocks = deep
        if best_len <= 135:
            break

    if best_blocks is None:
        best_blocks = greedy_cover(0)
        best_blocks = local_search(best_blocks, 0, max_iterations=150)

    if time.time() < deadline - 15:
        final = local_search(best_blocks, 9999, max_iterations=400)
        if len(final) < best_len:
            best_blocks = final
            best_len = len(final)

    return best_blocks
