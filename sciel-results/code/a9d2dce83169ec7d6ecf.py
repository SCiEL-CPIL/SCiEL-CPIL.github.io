import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    if v == 24 and k == 7 and t == 3:
        return construct_24_7_3(deadline)
    else:
        return construct_generic(v, k, t, deadline)

def construct_24_7_3(deadline):
    v, k, t = 24, 7, 3
    points = list(range(v))

    # ---- Precompute all triples and all blocks ----
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)  # 2024

    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)  # 346104

    # triples per block, in a fixed order
    block_triples = [list(itertools.combinations(b, t)) for b in all_blocks]
    triples_per_block = len(block_triples[0])  # 35

    # triple index arrays: block_triple_idx[b, c] = tsub index of c-th triple of block b
    block_triple_idx = np.empty((num_blocks, triples_per_block), dtype=np.int32)
    for b in range(num_blocks):
        for c, tri in enumerate(block_triples[b]):
            block_triple_idx[b, c] = tsub_to_idx[tri]

    # coverage matrix: cov[b, t] = 1 if block b covers triple t
    cov = np.zeros((num_blocks, num_tsubs), dtype=np.uint8)
    for b in range(num_blocks):
        cov[b, block_triple_idx[b]] = 1

    def greedy_construct(randomize=False):
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        chosen = []
        # per-block base triple list for quick counts
        bti = block_triple_idx
        while True:
            if not uncovered.any():
                break
            # counts of each triple among all blocks
            counts = np.bincount(bti.ravel(), minlength=num_tsubs)
            # gain per block = sum over its triples of uncovered
            gains = np.einsum('bc,tc->bt', bti.view('i4'), uncovered).sum(axis=0) if False else None
            # faster: gather uncovered via the index array
            unc_at = uncovered[bti]  # (num_blocks, 35)
            gains = unc_at.sum(axis=1)  # (num_blocks,)
            if randomize:
                gains = gains + random.random() * 0.001 * np.ones(num_blocks)
            best = int(np.argmax(gains))
            if gains[best] <= 0:
                break
            chosen.append(best)
            uncovered[bti[best]] = 0
            if time.time() > deadline - 12:
                break
        return chosen

    def redundancy_pass(chosen, deadline_pad=8):
        if not chosen:
            return chosen
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for b in chosen:
            counts[block_triple_idx[b]] += 1
        changed = True
        while changed:
            changed = False
            new_chosen = []
            for b in chosen:
                if time.time() > deadline - deadline_pad:
                    new_chosen.append(b)
                    continue
                # block b redundant iff all its triples have count >= 2
                if (counts[block_triple_idx[b]] >= 2).all():
                    counts[block_triple_idx[b]] -= 1
                    changed = True
                else:
                    new_chosen.append(b)
            chosen = new_chosen
        return chosen

    def local_search(chosen, max_iters=20000, deadline_pad=12):
        if not chosen:
            return chosen
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for b in chosen:
            counts[block_triple_idx[b]] += 1
        cov_b = cov  # (num_blocks, num_tsubs)

        iters = 0
        improved = True
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - deadline_pad:
                break

            # 1) try removals
            for i in range(len(chosen)):
                b = chosen[i]
                if (counts[block_triple_idx[b]] >= 2).all():
                    counts[block_triple_idx[b]] -= 1
                    chosen.pop(i)
                    improved = True
                    break

            if improved:
                continue

            # 2) try swaps: for each block i, find a replacement j that covers all
            #    of i's uniquely-covered triples and has high redundancy
            for i in range(len(chosen)):
                if time.time() > deadline - deadline_pad:
                    break
                b = chosen[i]
                idx_b = block_triple_idx[b]
                unique_mask = (counts == 1) & cov_b[b]
                n_unique = int(unique_mask.sum())
                if n_unique == 0:
                    continue
                # candidate j must cover all unique triples of b
                # cov_b[j] & unique_mask must equal unique_mask for all its set bits
                # i.e., (cov_b[j] & unique_mask).sum() == n_unique
                cand = (cov_b & unique_mask).sum(axis=1)
                cand[cand != n_unique] = -1
                cand[b] = -1
                # redundancy of j = number of j's triples already covered by others
                redundant_mask = (counts >= 2)
                red = (cov_b & redundant_mask).sum(axis=1)
                # pick best: maximize redundancy, tie-break by smaller block index
                valid = cand > 0
                if not valid.any():
                    continue
                score = np.where(valid, red, -1)
                best_j = int(np.argmax(score))
                if score[best_j] < 0:
                    continue
                # apply swap
                counts[block_triple_idx[b]] -= 1
                counts[block_triple_idx[best_j]] += 1
                chosen[i] = best_j
                improved = True
                break

        return chosen

    # ---- Run 1: deterministic ----
    chosen = greedy_construct(randomize=False)
    chosen = redundancy_pass(chosen)
    chosen = local_search(chosen)
    best = [list(all_blocks[b]) for b in chosen]
    best_len = len(chosen)

    # ---- Run 2: randomized restarts ----
    if time.time() < deadline - 25:
        for r in range(12):
            if time.time() > deadline - 12:
                break
            ch = greedy_construct(randomize=True)
            ch = redundancy_pass(ch)
            ch = local_search(ch)
            if len(ch) < best_len:
                best_len = len(ch)
                best = [list(all_blocks[b]) for b in ch]

    return best

def construct_generic(v, k, t, deadline):
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    def greedy_construct():
        uncovered = set(range(num_tsubs))
        blocks = []
        while uncovered:
            best_block = None
            best_count = -1
            sample_size = min(15, len(uncovered))
            candidates = set()
            for idx in random.sample(list(uncovered), sample_size):
                triple = all_tsubs[idx]
                remaining = [x for x in range(v) if x not in triple]
                for combo in itertools.combinations(remaining, k - t):
                    candidates.add(tuple(sorted(triple + combo)))
            for block in candidates:
                count = 0
                for triple in itertools.combinations(block, t):
                    if tsub_to_idx[triple] in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_block = block
            if best_block is None:
                break
            blocks.append(list(best_block))
            for triple in itertools.combinations(best_block, t):
                uncovered.discard(tsub_to_idx[triple])
            if time.time() > deadline - 5:
                break
        return blocks

    blocks = greedy_construct()
    # redundancy removal
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            redundant = True
            for triple in itertools.combinations(blocks[i], t):
                ts = set(triple)
                covered = any(ts.issubset(set(blocks[j])) for j in range(len(blocks)) if j != i)
                if not covered:
                    redundant = False
                    break
            if redundant:
                blocks.pop(i)
                changed = True
    return blocks
