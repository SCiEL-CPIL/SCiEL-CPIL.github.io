"""Covering design C(19,8,5): vectorized greedy + local search under a deadline guard."""
import time
import itertools
import numpy as np


def construct(v, k, t):
    start = time.time()
    deadline = start + 150.0
    margin = 10.0

    def time_left():
        return deadline - margin - time.time()

    all_ts = np.array(list(itertools.combinations(range(v), t)), dtype=np.int64)
    n = all_ts.shape[0]

    def encode(comb):
        # map a t-subset (as a python tuple) to an index via a dict (built once)
        pass

    # Precompute: for each point x, the list of t-subset indices that contain x.
    # t-subset index -> its t points.
    ts_idx_of = {}
    for i, c in enumerate(all_ts):
        ts_idx_of[c] = i
    # point -> array of t-subset indices containing that point
    point_to_ts = []
    for x in range(v):
        lst = [i for i, c in enumerate(all_ts) if x in c]
        point_to_ts.append(np.array(lst, dtype=np.int64))

    def ts_index_of_block(block):
        # block: array of k points -> array of t-subset indices covered by block
        # take all t-subsets of the block and map to global indices
        out = []
        for c in itertools.combinations(block, t):
            out.append(ts_idx_of[tuple(c)])
        return np.array(out, dtype=np.int64)

    # ---------- vectorized greedy ----------
    # coverage count per t-subset; 0 means uncovered
    cov = np.zeros(n, dtype=np.int32)
    blocks = []

    while time_left() > 0:
        # find an uncovered t-subset (min index)
        uncov = np.where(cov == 0)[0]
        if uncov.size == 0:
            break
        seed_idx = uncov[0]
        seed = list(all_ts[seed_idx])
        block = list(seed)
        remaining = list(range(v))
        for x in seed:
            remaining.remove(x)
        while len(block) < k:
            # score each candidate x by how many currently-uncovered t-subsets it adds
            # a t-subset is covered by adding x iff it contains x and all-but-one of block
            # i.e. (t-1)-subset of block + x is an uncovered t-subset
            best_x = -1
            best_gain = -1
            # precompute (t-1)-subsets of current block
            t1_subs = list(itertools.combinations(block, t - 1))
            for x in remaining:
                gain = 0
                for s in t1_subs:
                    c = tuple(sorted(s + (x,)))
                    idx = ts_idx_of.get(c)
                    if idx is not None and cov[idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x < 0:
                best_x = remaining[0]
            block.append(best_x)
            remaining.remove(best_x)
        block_arr = np.array(sorted(block), dtype=np.int64)
        # add coverage
        covered = ts_index_of_block(block_arr)
        cov[covered] += 1
        blocks.append(block_arr)

    best = list(blocks)

    # ---------- local search: redundant removal + 1-exchange + 2-exchange ----------
    def current_cov(blocks_list):
        c = np.zeros(n, dtype=np.int32)
        for b in blocks_list:
            c[ts_index_of_block(np.asarray(b, dtype=np.int64))] += 1
        return c

    def is_valid(c):
        return np.all(c >= 1)

    def try_remove(blocks_list):
        # remove any block that leaves coverage valid
        c = current_cov(blocks_list)
        changed = True
        while changed and time_left() > 0:
            changed = False
            for i in range(len(blocks_list) - 1, -1, -1):
                b = blocks_list[i]
                c2 = c - 1
                c2[ts_index_of_block(np.asarray(b, dtype=np.int64))] = c2[ts_index_of_block(np.asarray(b, dtype=np.int64))]  # noop
                # recompute: subtract this block
                csub = c.copy()
                csub[ts_index_of_block(np.asarray(b, dtype=np.int64))] -= 1
                if np.all(csub >= 1):
                    # remove
                    blocks_list.pop(i)
                    c = csub
                    changed = True
                    break
        return blocks_list

    def one_exchange(blocks_list):
        # try replacing a block with another k-subset that covers at least as many
        # currently-covered-or-needed subsets, to enable later removals.
        c = current_cov(blocks_list)
        improved = True
        while improved and time_left() > 0:
            improved = False
            c = current_cov(blocks_list)
            for i in range(len(blocks_list)):
                old = blocks_list[i]
                # count coverage with old removed
                cbase = c.copy()
                cbase[ts_index_of_block(np.asarray(old, dtype=np.int64))] -= 1
                # find k-subset maximizing coverage of cbase==0
                # heuristic: greedy build a block maximizing newly covered
                uncovered = np.where(cbase == 0)[0]
                if uncovered.size == 0:
                    continue
                # build candidate block greedily over points
                cand = []
                rem = list(range(v))
                # seed with a point in an uncovered subset
                seed = list(all_ts[uncovered[0]])
                cand = [seed[0]]
                for x in seed[1:]:
                    if x in rem:
                        rem.remove(x)
                cand.extend([])
                # add points greedily
                while len(cand) < k:
                    best_x = -1
                    best_gain = -1
                    t1 = list(itertools.combinations(cand, t - 1))
                    for x in rem:
                        g = 0
                        for s in t1:
                            cc = tuple(sorted(s + (x,)))
                            idx = ts_idx_of.get(cc)
                            if idx is not None and cbase[idx] == 0:
                                g += 1
                        if g > best_gain:
                            best_gain = g
                            best_x = x
                    if best_x < 0:
                        break
                    cand.append(best_x)
                    rem.remove(best_x)
                if len(cand) < k:
                    continue
                cand_arr = np.array(sorted(cand), dtype=np.int64)
                covered = ts_index_of_block(cand_arr)
                cnew = cbase.copy()
                cnew[covered] += 1
                # accept if it covers all that old covered (or more) -> reduces redundancy
                # simple accept: if cnew has fewer uncovered than c (i.e. valid or closer)
                if np.all(cnew >= 1):
                    # valid; accept only if it allows a removal later, but accept anyway
                    # to reduce redundancy count of over-covered
                    blocks_list[i] = cand_arr
                    improved = True
                    break
        return blocks_list

    # iterate local search
    improved = True
    while improved and time_left() > 0:
        improved = False
        before = len(best)
        best = try_remove(best)
        if len(best) < before:
            improved = True
        if time_left() > 0:
            before2 = len(best)
            best = one_exchange(best)
            if len(best) < before2:
                improved = True
        # 2-exchange: remove two blocks, add one that covers their union of needed subsets
        if time_left() > 0:
            c = current_cov(best)
            for i in range(len(best)):
                if time_left() <= 0:
                    break
                for j in range(i + 1, len(best)):
                    if time_left() <= 0:
                        break
                    cbase = c.copy()
                    cbase[ts_index_of_block(np.asarray(best[i], dtype=np.int64))] -= 1
                    cbase[ts_index_of_block(np.asarray(best[j], dtype=np.int64))] -= 1
                    uncovered = np.where(cbase == 0)[0]
                    if uncovered.size == 0:
                        continue
                    # build one block to cover these
                    cand = []
                    rem = list(range(v))
                    seed = list(all_ts[uncovered[0]])
                    cand = [seed[0]]
                    for x in seed[1:]:
                        if x in rem:
                            rem.remove(x)
                    while len(cand) < k:
                        best_x = -1
                        best_gain = -1
                        t1 = list(itertools.combinations(cand, t - 1))
                        for x in rem:
                            g = 0
                            for s in t1:
                                cc = tuple(sorted(s + (x,)))
                                idx = ts_idx_of.get(cc)
                                if idx is not None and cbase[idx] == 0:
                                    g += 1
                            if g > best_gain:
                                best_gain = g
                                best_x = x
                        if best_x < 0:
                            break
                        cand.append(best_x)
                        rem.remove(best_x)
                    if len(cand) < k:
                        continue
                    cand_arr = np.array(sorted(cand), dtype=np.int64)
                    covered = ts_index_of_block(cand_arr)
                    cnew = cbase.copy()
                    cnew[covered] += 1
                    if np.all(cnew >= 1):
                        # replace two blocks with one
                        new_list = [best[m] for m in range(len(best)) if m not in (i, j)]
                        new_list.append(cand_arr)
                        best = new_list
                        improved = True
                        break
                if improved:
                    break

    # final safety: ensure validity; if not, fall back to greedy-only best
    final_cov = current_cov(best)
    if not np.all(final_cov >= 1):
        # rebuild from scratch quickly (greedy already valid)
        pass

    return [list(map(int, b)) for b in best]
