"""Fast covering C(v,k,t) via bitset coverage + multi-strategy greedy + polish."""
import time
import random
import itertools
import numpy as np

_DEADLINE = 145.0  # safety margin below the 150s kill


def construct(v, k, t):
    start = time.time()

    # ---- precompute masks for every t-subset (each t-subset = one bit) ----
    combos = list(itertools.combinations(range(v), t))
    n_t = len(combos)
    if n_t == 0:
        return []
    idx = {}
    for i, c in enumerate(combos):
        idx[c] = i
    masks = np.zeros(n_t, dtype=np.int64)
    for c in combos:
        masks[idx[c]] = 1 << idx[c]
    mask_of = [masks[idx[c]] for c in combos]  # index by tuple
    all_bits = np.zeros(n_t, dtype=np.int64)
    all_bits[:] = -1

    def block_mask(block):
        m = np.zeros(1, dtype=np.int64)[0]
        for c in itertools.combinations(block, t):
            m |= mask_of[c]
        return m

    def popcount(x):
        return int(np.count_nonzero(x))

    def time_left():
        return _DEADLINE - (time.time() - start)

    # ---- greedy build of one block given current covered ----
    def build_block(covered, seed_pt, rng):
        block = [seed_pt]
        while len(block) < k:
            in_block = set(block)
            best = -1
            best_g = -1
            for x in range(v):
                if x in in_block:
                    continue
                m = np.zeros(1, dtype=np.int64)[0]
                for c in itertools.combinations(block, t - 1):
                    m |= mask_of[tuple(sorted(c + (x,)))]
                g = popcount(m & ~covered)
                if g > best_g:
                    best_g = g
                    best = x
                elif g == best_g and best_g > 0 and rng.random() < 0.5:
                    best = x
            block.append(best)
        block.sort()
        return block

    # ---- polish: try swapping one point of a block to raise its gain ----
    def polish(blocks, covered, rng):
        if time_left() < 2.0:
            return
        improved = True
        while improved and time_left() > 2.0:
            improved = False
            for bi, block in enumerate(blocks):
                in_block = set(block)
                cur_m = block_mask(block)
                cur_g = popcount(cur_m & ~covered)
                for out in block:
                    for x in range(v):
                        if x in in_block:
                            continue
                        nb = [p for p in block if p != out] + [x]
                        nb.sort()
                        nb_m = block_mask(nb)
                        g = popcount(nb_m & ~covered)
                        if g > cur_g:
                            blocks[bi] = nb
                            improved = True
                            break
                    if improved:
                        break

    # ---- run a single greedy strategy, return (blocks, covered) ----
    def run_strategy(seed_mode, seed_pt, rng):
        covered = np.zeros(n_t, dtype=np.int64)
        blocks = []
        while time_left() > 1.0:
            if seed_mode == 'seed':
                # first uncovered t-subset
                unc = ~covered
                i = int(np.argmax(unc))
                sp = combos[i][0]
            else:
                # random point from an uncovered t-subset
                unc = ~covered
                if not np.any(unc):
                    break
                ci = int(rng.choice(np.nonzero(unc)[0]))
                sp = rng.choice(combos[ci])
            block = build_block(covered, sp, rng)
            blocks.append(block)
            covered |= block_mask(block)
            if not np.any(~covered):
                break
        return blocks, covered

    # ---- main multi-restart loop ----
    best_blocks = None
    best_len = None
    rng = random.Random(12345)

    # strategy 1: seed-first greedy
    b, cov = run_strategy('seed', 0, rng)
    if best_len is None or len(b) < best_len:
        best_len, best_blocks = len(b), b
    polish(best_blocks, cov, rng)

    # strategy 2..: random-start greedy restarts
    attempts = 0
    while time_left() > 5.0:
        attempts += 1
        sp = rng.randrange(v)
        b, cov = run_strategy('random', sp, rng)
        if best_len is None or len(b) < best_len:
            best_len, best_blocks = len(b), b
            polish(best_blocks, cov, rng)

    # final safety: if we somehow have no blocks, do one plain greedy
    if best_blocks is None:
        b, cov = run_strategy('seed', 0, rng)
        best_blocks = b

    return best_blocks
