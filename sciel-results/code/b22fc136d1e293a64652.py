import itertools
import time
import random
import numpy as np

V = 24
T = 3
K = 7

all_triples = list(itertools.combinations(range(V), T))
triple_to_idx = {t: i for i, t in enumerate(all_triples)}
N_TRIPLES = len(all_triples)

pair_list = list(itertools.combinations(range(V), 2))
pair_to_idx = {p: i for i, p in enumerate(pair_list)}
N_PAIRS = len(pair_list)

# triple_idx_table[x, pair_idx] = idx of triple (x, a, b) if x not in pair, else -1
_triple_idx_table = np.empty((V, N_PAIRS), dtype=np.int32)
for _x in range(V):
    for _pi, (_a, _b) in enumerate(pair_list):
        if _x == _a or _x == _b:
            _triple_idx_table[_x, _pi] = -1
        else:
            _tri = tuple(sorted((_x, _a, _b)))
            _triple_idx_table[_x, _pi] = triple_to_idx[_tri]


def _pair_idx_array(block):
    return np.array([pair_to_idx[(a, b)] for (a, b) in itertools.combinations(block, 2)], dtype=np.int32)


def _greedy(v, k, t, deadline, rng, seed_strategy=0):
    cover_count = np.zeros(N_TRIPLES, dtype=np.int32)
    uncovered = np.ones(N_TRIPLES, dtype=bool)
    blocks = []

    while True:
        if time.time() > deadline:
            break
        uncovered_idx = np.nonzero(uncovered)[0]
        if len(uncovered_idx) == 0:
            break

        # Select seed triple
        if seed_strategy == 0:
            seed_idx = uncovered_idx[0]
        elif seed_strategy == 1:
            seed_idx = uncovered_idx[rng.randrange(len(uncovered_idx))]
        else:
            # max-score (sampled): triple anchoring most uncovered triples
            sample = uncovered_idx if len(uncovered_idx) <= 200 else \
                uncovered_idx[rng.choice(len(uncovered_idx), 200, replace=False)]
            best_score = -1
            best_idx = sample[0]
            for si in sample:
                a, b, c = all_triples[si]
                score = 0
                for x in range(v):
                    if x == a or x == b or x == c:
                        continue
                    i1 = triple_to_idx[tuple(sorted((x, a, b)))]
                    i2 = triple_to_idx[tuple(sorted((x, a, c)))]
                    i3 = triple_to_idx[tuple(sorted((x, b, c)))]
                    score += (1 if uncovered[i1] else 0) + (1 if uncovered[i2] else 0) + (1 if uncovered[i3] else 0)
                if score > best_score:
                    best_score = score
                    best_idx = si
            seed_idx = best_idx

        block = list(all_triples[seed_idx])

        while len(block) < k:
            if time.time() > deadline:
                break
            block_set = set(block)
            pair_idxs = _pair_idx_array(block)
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                idxs = _triple_idx_table[x, pair_idxs]
                gain = int(uncovered[idxs].sum())
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                elif gain == best_gain and gain > 0 and rng.random() < 0.5:
                    best_x = x
            block.append(best_x)

        if len(block) != k:
            break  # deadline hit mid-block; do not append partial block

        block.sort()
        blocks.append(block)
        for tri in itertools.combinations(block, t):
            idx = triple_to_idx[tri]
            if cover_count[idx] == 0:
                uncovered[idx] = False
            cover_count[idx] += 1

    return blocks


def _remove_redundant(blocks, v, t, deadline):
    if not blocks:
        return blocks
    cover_count = np.zeros(N_TRIPLES, dtype=np.int32)
    block_triples = []
    for b in blocks:
        tris = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            cover_count[idx] += 1
            tris.append(idx)
        block_triples.append(np.array(tris, dtype=np.int32))

    active = [True] * len(blocks)
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
        for i in range(len(blocks)):
            if not active[i]:
                continue
            tris = block_triples[i]
            if np.all(cover_count[tris] > 1):
                active[i] = False
                changed = True
                cover_count[tris] -= 1
    return [blocks[i] for i in range(len(blocks)) if active[i]]


def _build_block_for(cover_local, seed_triple, v, k, deadline, rng):
    block = list(seed_triple)
    while len(block) < k:
        if time.time() > deadline:
            break
        block_set = set(block)
        pair_idxs = _pair_idx_array(block)
        best_x = None
        best_g = -1
        for x in range(v):
            if x in block_set:
                continue
            idxs = _triple_idx_table[x, pair_idxs]
            g = int((cover_local[idxs] == 0).sum())
            if g > best_g:
                best_g = g
                best_x = x
        block.append(best_x)
    return block


def _local_search(blocks, v, t, deadline, rng):
    best = list(blocks)
    best_n = len(best)
    for _pass in range(4):
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(best)):
            if time.time() > deadline:
                break
            trial = best[:i] + best[i + 1:]
            cover = np.zeros(N_TRIPLES, dtype=np.int32)
            for b in trial:
                for tri in itertools.combinations(b, t):
                    cover[triple_to_idx[tri]] += 1
            unc = np.nonzero(cover == 0)[0]
            if len(unc) == 0:
                best = trial
                best_n -= 1
                improved = True
                break
            # try to cover unc with one new block
            seed = all_triples[unc[rng.randrange(len(unc))]]
            block = _build_block_for(cover, seed, v, K, deadline, rng)
            if len(block) != K:
                continue
            block.sort()
            new_blocks = trial + [block]
            cover2 = np.zeros(N_TRIPLES, dtype=np.int32)
            for b in new_blocks:
                for tri in itertools.combinations(b, t):
                    cover2[triple_to_idx[tri]] += 1
            if np.all(cover2 > 0) and len(new_blocks) < best_n:
                best = new_blocks
                best_n = len(new_blocks)
                improved = True
                break
        if not improved:
            break
    return best


def construct(v, k, t):
    start = time.time()
    deadline = start + 140.0
    best_blocks = None
    best_count = 10 ** 9

    attempts = []
    for seed in range(15):
        attempts.append((0, seed))
    for seed in range(15):
        attempts.append((1, seed))
    for seed in range(6):
        attempts.append((2, seed))
    random.Random(12345).shuffle(attempts)

    for strat, seed in attempts:
        if time.time() > deadline - 6.0:
            break
        rng = random.Random(seed * 1000 + strat + 1)
        blocks = _greedy(v, k, t, deadline, rng, strat)
        blocks = _remove_redundant(blocks, v, t, deadline)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)

    if best_blocks is not None and time.time() < deadline - 3.0:
        rng_ls = random.Random(777)
        ls_blocks = _local_search(best_blocks, v, t, deadline, rng_ls)
        if len(ls_blocks) < best_count:
            best_blocks = ls_blocks
            best_count = len(ls_blocks)

    if best_blocks is None:
        rng = random.Random(0)
        best_blocks = _greedy(v, k, t, deadline, rng, 0)
        best_blocks = _remove_redundant(best_blocks, v, t, deadline)

    return best_blocks
