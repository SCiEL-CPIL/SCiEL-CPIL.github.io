"""Covering design C(v,k,t): fast global greedy (multiple randomized restarts) + local-search refinement."""
import itertools
import random
import time


def _build_structures(v, k, t):
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    idx_of = {s: i for i, s in enumerate(all_subsets)}
    subs_containing = [[] for _ in range(v)]
    for i, s in enumerate(all_subsets):
        for x in s:
            subs_containing[x].append(i)
    all_kblocks = list(itertools.combinations(range(v), k))
    # precompute t-subset-id set for each k-block
    block_ids = []
    for comb in all_kblocks:
        ids = set()
        for s in itertools.combinations(comb, t):
            ids.add(idx_of[s])
        block_ids.append(ids)
    return all_subsets, n_subsets, idx_of, subs_containing, all_kblocks, block_ids


def _greedy(v, k, t, order, subs_containing, n_subsets, deadline):
    covered = bytearray(n_subsets)
    blocks = []
    remaining = n_subsets
    while remaining > 0 and time.time() < deadline:
        best_block = None
        best_gain = -1
        for comb in order:
            seen = set()
            for x in comb:
                for sid in subs_containing[x]:
                    seen.add(sid)
            gain = 0
            for sid in seen:
                if not covered[sid]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = comb
                if gain == 0:
                    break
        if best_block is None or best_gain <= 0:
            break
        for x in best_block:
            for sid in subs_containing[x]:
                if not covered[sid]:
                    covered[sid] = 1
                    remaining -= 1
        blocks.append(list(best_block))
    return blocks


def _refine(v, k, t, blocks, all_subsets, idx_of, n_subsets, subs_containing, all_kblocks, block_ids, deadline):
    # map block tuple -> its id-set
    def ids_of(bl):
        ids = set()
        for s in itertools.combinations(bl, t):
            ids.add(idx_of[s])
        return ids

    id_sets = [ids_of(b) for b in blocks]

    def full_coverage(exclude=None):
        cov = bytearray(n_subsets)
        for i, ids in enumerate(id_sets):
            if i == exclude:
                continue
            for sid in ids:
                cov[sid] = 1
        return cov

    changed = True
    while changed and time.time() < deadline:
        changed = False
        n = len(id_sets)
        # deletion
        for i in range(n):
            if time.time() > deadline:
                return blocks
            cov = full_coverage(exclude=i)
            if all(cov[sid] for sid in range(n_subsets)):
                del blocks[i]
                del id_sets[i]
                changed = True
                break
        if changed:
            continue
        # replacement: for each i, find block covering all t-subsets uncovered by others
        n = len(id_sets)
        for i in range(n):
            if time.time() > deadline:
                return blocks
            cov = full_coverage(exclude=i)
            uncovered = [sid for sid in range(n_subsets) if not cov[sid]]
            if not uncovered:
                del blocks[i]
                del id_sets[i]
                changed = True
                break
            un_set = set(uncovered)
            best_gain = 0
            best_new = None
            for bi, ids in enumerate(block_ids):
                gain = len(ids & un_set)
                if gain > best_gain:
                    best_gain = gain
                    best_new = all_kblocks[bi]
                    if gain == len(un_set):
                        break
            if best_new is not None and best_gain == len(un_set):
                blocks[i] = list(best_new)
                id_sets[i] = ids_of(best_new)
                changed = True
                break
    return blocks


def construct(v, k, t):
    start = time.time()
    deadline = start + 110.0
    (all_subsets, n_subsets, idx_of, subs_containing, all_kblocks, block_ids) = _build_structures(v, k, t)

    best = None
    best_len = None
    n_restarts = 30
    for seed in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        rng = random.Random(seed)
        order = all_kblocks[:]
        rng.shuffle(order)
        blocks = _greedy(v, k, t, order, subs_containing, n_subsets, deadline - 15.0)
        if blocks and (best_len is None or len(blocks) < best_len):
            if time.time() < deadline - 8.0:
                blocks = _refine(v, k, t, blocks, all_subsets, idx_of, n_subsets,
                                 subs_containing, all_kblocks, block_ids, deadline - 5.0)
            best = blocks
            best_len = len(blocks)
    return best
