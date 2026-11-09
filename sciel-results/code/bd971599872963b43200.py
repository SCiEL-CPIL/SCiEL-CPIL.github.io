import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    points = list(range(v))
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    subset_to_id = {s: i for i, s in enumerate(all_t)}

    t1_list = list(itertools.combinations(points, t - 1))
    t1_to_ids = np.full((len(t1_list), v), -1, dtype=np.int64)
    t1_index = {}
    for i, T in enumerate(t1_list):
        t1_index[T] = i
        for x in points:
            if x not in T:
                t1_to_ids[i, x] = subset_to_id[tuple(sorted(T + (x,)))]

    def make_block_from_seed(seed_subset, unc):
        block = list(seed_subset)
        while len(block) < k:
            gains = np.zeros(v, dtype=np.int64)
            for T in itertools.combinations(block, t - 1):
                idx = t1_index[T]
                gains += unc[t1_to_ids[idx]]
            for x in block:
                gains[x] = -1
            best = np.max(gains)
            cands = np.where(gains == best)[0]
            if len(cands) == 0:
                cand_list = [x for x in range(v) if x not in block]
                block.append(random.choice(cand_list))
            else:
                block.append(int(cands[random.randint(0, len(cands) - 1)]))
        block.sort()
        return block

    def greedy_cover(unc, seed):
        random.seed(seed)
        if not unc.any():
            return []
        blocks = []
        unc = unc.copy()
        while unc.any():
            unc_idx = np.where(unc)[0]
            start_s = all_t[unc_idx[random.randint(0, len(unc_idx) - 1)]]
            block = make_block_from_seed(start_s, unc)
            blocks.append(block)
            for comb in itertools.combinations(block, t):
                unc[subset_to_id[comb]] = False
        return blocks

    def build_counts(blocks):
        counts = np.zeros(N_t, dtype=np.int32)
        block_subs = []
        for b in blocks:
            subs = [subset_to_id[comb] for comb in itertools.combinations(b, t)]
            block_subs.append(subs)
            for s in subs:
                counts[s] += 1
        return counts, block_subs

    def reduce_blocks(blocks, seed):
        random.seed(seed)
        if not blocks:
            return blocks
        counts, block_subs = build_counts(blocks)
        reduced = list(blocks)
        rsubs = list(block_subs)
        changed = True
        while changed and time.time() < deadline - 8.0:
            changed = False
            order = list(range(len(reduced)))
            random.shuffle(order)
            for i in order:
                if i >= len(reduced):
                    continue
                subs = rsubs[i]
                redundant = True
                for s in subs:
                    if counts[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in subs:
                        counts[s] -= 1
                    reduced.pop(i)
                    rsubs.pop(i)
                    changed = True
                    break
        return reduced

    def shake_recover(blocks, seed):
        random.seed(seed)
        if len(blocks) < 2:
            return blocks
        counts, block_subs = build_counts(blocks)
        # try several random removals
        best_local = list(blocks)
        for _ in range(5):
            if time.time() > deadline - 10.0:
                break
            i = random.randint(0, len(block_subs) - 1)
            subs = block_subs[i]
            # remove block i
            for s in subs:
                counts[s] -= 1
            new_unc = counts == 0
            # greedily cover new_unc
            added = []
            unc = new_unc.copy()
            while unc.any():
                unc_idx = np.where(unc)[0]
                start_s = all_t[unc_idx[random.randint(0, len(unc_idx) - 1)]]
                block = make_block_from_seed(start_s, unc)
                added.append(block)
                for comb in itertools.combinations(block, t):
                    unc[subset_to_id[comb]] = False
            # net change: removed 1, added len(added)
            if len(added) < 1:
                # keep
                best_local = [b for j, b in enumerate(blocks) if j != i] + added
            # restore counts for next iteration
            for s in subs:
                counts[s] += 1
            for b in added:
                for comb in itertools.combinations(b, t):
                    counts[subset_to_id[comb]] += 1
            # need to recompute properly; simpler: rebuild
        # rebuild counts to be safe
        return best_local

    best = None
    best_n = float('inf')
    initial_unc = np.ones(N_t, dtype=bool)

    n_restarts = 0
    while time.time() < deadline - 5.0:
        n_restarts += 1
        seed = 1000 + n_restarts * 7919
        blocks = greedy_cover(initial_unc, seed)
        n = len(blocks)
        if n < best_n:
            best_n = n
            best = blocks
        if time.time() < deadline - 10.0:
            red = reduce_blocks(best, seed + 1)
            if len(red) < best_n:
                best_n = len(red)
                best = red

    if best is None:
        best = greedy_cover(initial_unc, 1)
    return best
