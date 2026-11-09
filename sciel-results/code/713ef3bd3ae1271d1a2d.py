"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 78.0  # safety margin before 90s wall-clock kill

    def elapsed():
        return time.time() - start_time

    N_SMALL_END = 128
    N_LARGE_END = 256

    small_nums = list(range(2, N_SMALL_END + 1))
    large_nums = list(range(N_SMALL_END + 1, N_LARGE_END + 1))

    n_small = len(small_nums)
    n_large = len(large_nums)

    # Exact reciprocal weights (rational objective)
    small_w = [Fraction(1, x) for x in small_nums]
    large_w = [Fraction(1, x) for x in large_nums]

    # Small-small conflict: lcm(a,b) <= 256 means cannot both be chosen.
    small_conflict_mask = [0] * n_small
    for i in range(n_small):
        a = small_nums[i]
        for j in range(i + 1, n_small):
            b = small_nums[j]
            if a * b // gcd(a, b) <= N_LARGE_END:
                small_conflict_mask[i] |= (1 << j)
                small_conflict_mask[j] |= (1 << i)

    # For each small i, which large numbers are blocked (lcm <= 256).
    small_blocks_large = [0] * n_small
    for i in range(n_small):
        a = small_nums[i]
        m = 0
        for j in range(n_large):
            b = large_nums[j]
            if a * b // gcd(a, b) <= N_LARGE_END:
                m |= (1 << j)
        small_blocks_large[i] = m

    all_large_mask = (1 << n_large) - 1

    def popcount(m):
        return bin(m).count("1")

    def large_score(unblocked_mask):
        # Large numbers [129,256] form a perfectly independent set: no two
        # distinct large numbers have lcm <= 256 (2*129=258>256, and for any
        # pair the larger is >128 so lcm is at least the larger * >=2 > 256).
        # Hence the best large selection is exactly the unblocked set.
        s = Fraction(0)
        m = unblocked_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += large_w[idx]
            m ^= lsb
        return s

    def blocked_of(mask):
        b = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            b |= small_blocks_large[idx]
            m ^= lsb
        return b

    def small_sum(mask):
        s = Fraction(0)
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += small_w[idx]
            m ^= lsb
        return s

    def total_score(mask):
        blocked = blocked_of(mask)
        return small_sum(mask) + large_score(all_large_mask & ~blocked)

    best_score = Fraction(0)
    best_small_mask = 0

    def update_best(sm, score):
        nonlocal best_score, best_small_mask
        if score > best_score:
            best_score = score
            best_small_mask = sm

    # ---------- Build initial feasible small sets ----------
    initial_masks = []
    seen = set()

    def add_mask(m):
        if m not in seen:
            seen.add(m)
            initial_masks.append(m)

    # 1) Greedy by weight (smallest numbers first)
    mask = 0
    for i in range(n_small):
        if not (small_conflict_mask[i] & mask):
            mask |= (1 << i)
    add_mask(mask)

    # 2) Randomized greedy restarts
    rng = random.Random(12345)
    for _ in range(120):
        order = list(range(n_small))
        rng.shuffle(order)
        mask = 0
        for i in order:
            if not (small_conflict_mask[i] & mask):
                mask |= (1 << i)
        add_mask(mask)

    # 3) Singletons
    for i in range(n_small):
        add_mask(1 << i)

    # ---------- Local search with exact delta evaluation ----------
    def polish(init_mask):
        cur = init_mask
        cur_blocked = blocked_of(cur)
        cur_unblocked = all_large_mask & ~cur_blocked
        cur_large = large_score(cur_unblocked)
        cur_small = small_sum(cur)
        cur_score = cur_small + cur_large
        update_best(cur, cur_score)

        improved = True
        while improved and elapsed() < DEADLINE:
            improved = False
            best_move_score = cur_score
            best_new_mask = cur
            best_new_blocked = cur_blocked

            # ADD i
            for i in range(n_small):
                if cur & (1 << i):
                    continue
                if small_conflict_mask[i] & cur:
                    continue
                nb = cur_blocked | small_blocks_large[i]
                nu = all_large_mask & ~nb
                nl = large_score(nu)
                ns = cur_score + small_w[i] + (nl - cur_large)
                if ns > best_move_score:
                    best_move_score = ns
                    best_new_mask = cur | (1 << i)
                    best_new_blocked = nb

            # REMOVE i
            for i in range(n_small):
                if not (cur & (1 << i)):
                    continue
                nb = 0
                m2 = cur & ~(1 << i)
                while m2:
                    lsb = m2 & -m2
                    idx = lsb.bit_length() - 1
                    nb |= small_blocks_large[idx]
                    m2 ^= lsb
                nu = all_large_mask & ~nb
                nl = large_score(nu)
                ns = cur_score - small_w[i] + (nl - cur_large)
                if ns > best_move_score:
                    best_move_score = ns
                    best_new_mask = cur & ~(1 << i)
                    best_new_blocked = nb

            # SWAP i->j (remove i, add j)
            if elapsed() < DEADLINE - 4.0:
                sel = []
                m = cur
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    sel.append(idx)
                    m ^= lsb
                for i in sel:
                    mw = cur & ~(1 << i)
                    bw = 0
                    m2 = mw
                    while m2:
                        lsb = m2 & -m2
                        idx = lsb.bit_length() - 1
                        bw |= small_blocks_large[idx]
                        m2 ^= lsb
                    for j in range(n_small):
                        if mw & (1 << j):
                            continue
                        if small_conflict_mask[j] & mw:
                            continue
                        nb = bw | small_blocks_large[j]
                        nu = all_large_mask & ~nb
                        nl = large_score(nu)
                        ns = cur_score - small_w[i] + small_w[j] + (nl - cur_large)
                        if ns > best_move_score:
                            best_move_score = ns
                            best_new_mask = mw | (1 << j)
                            best_new_blocked = nb

            # 2-OPT (remove i1,i2, add j1,j2)
            if elapsed() < DEADLINE - 8.0:
                sel = []
                m = cur
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    sel.append(idx)
                    m ^= lsb
                if len(sel) >= 2:
                    cands = [j for j in range(n_small) if not (cur & (1 << j))]
                    for a in range(len(sel)):
                        for b in range(a + 1, len(sel)):
                            i1, i2 = sel[a], sel[b]
                            mw = cur & ~(1 << i1) & ~(1 << i2)
                            bw = 0
                            m2 = mw
                            while m2:
                                lsb = m2 & -m2
                                idx = lsb.bit_length() - 1
                                bw |= small_blocks_large[idx]
                                m2 ^= lsb
                            base = cur_score - small_w[i1] - small_w[i2]
                            for j1 in cands:
                                if small_conflict_mask[j1] & mw:
                                    continue
                                mw1 = mw | (1 << j1)
                                nb1 = bw | small_blocks_large[j1]
                                for j2 in cands:
                                    if j2 == j1:
                                        continue
                                    if small_conflict_mask[j2] & mw1:
                                        continue
                                    nb = nb1 | small_blocks_large[j2]
                                    nu = all_large_mask & ~nb
                                    nl = large_score(nu)
                                    ns = base + small_w[j1] + small_w[j2] + (nl - cur_large)
                                    if ns > best_move_score:
                                        best_move_score = ns
                                        best_new_mask = mw1 | (1 << j2)
                                        best_new_blocked = nb

            if best_new_mask != cur:
                cur = best_new_mask
                cur_blocked = best_new_blocked
                cur_unblocked = all_large_mask & ~cur_blocked
                cur_large = large_score(cur_unblocked)
                cur_small = small_sum(cur)
                cur_score = cur_small + cur_large
                improved = True
                update_best(cur, cur_score)
            else:
                improved = False

    for im in initial_masks:
        if elapsed() > DEADLINE - 2.0:
            break
        polish(im)

    # ---------- Final exact polish passes on the best solution ----------
    for _ in range(4):
        if elapsed() > DEADLINE - 1.0:
            break
        cur = best_small_mask
        cur_blocked = blocked_of(cur)
        cur_unblocked = all_large_mask & ~cur_blocked
        cur_large = large_score(cur_unblocked)
        cur_small = small_sum(cur)
        cur_score = cur_small + cur_large
        improved = True
        while improved and elapsed() < DEADLINE - 1.0:
            improved = False
            # additions
            for i in range(n_small):
                if cur & (1 << i) or (small_conflict_mask[i] & cur):
                    continue
                nb = cur_blocked | small_blocks_large[i]
                nu = all_large_mask & ~nb
                nl = large_score(nu)
                ns = cur_score + small_w[i] + (nl - cur_large)
                if ns > cur_score:
                    cur |= (1 << i)
                    cur_blocked = nb
                    cur_unblocked = nu
                    cur_large = nl
                    cur_small += small_w[i]
                    cur_score = ns
                    improved = True
                    break
            # removals
            if not improved:
                for i in range(n_small):
                    if not (cur & (1 << i)):
                        continue
                    nb = 0
                    m2 = cur & ~(1 << i)
                    while m2:
                        lsb = m2 & -m2
                        idx = lsb.bit_length() - 1
                        nb |= small_blocks_large[idx]
                        m2 ^= lsb
                    nu = all_large_mask & ~nb
                    nl = large_score(nu)
                    ns = cur_score - small_w[i] + (nl - cur_large)
                    if ns > cur_score:
                        cur &= ~(1 << i)
                        cur_blocked = nb
                        cur_unblocked = nu
                        cur_large = nl
                        cur_small -= small_w[i]
                        cur_score = ns
                        improved = True
                        break
        if cur_score > best_score:
            update_best(cur, cur_score)

    # ---------- Construct final result ----------
    result = []
    m = best_small_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(small_nums[idx])
        m ^= lsb

    blocked = blocked_of(best_small_mask)
    unblocked = all_large_mask & ~blocked
    lm = unblocked
    while lm:
        lsb = lm & -lm
        idx = lsb.bit_length() - 1
        result.append(large_nums[idx])
        lm ^= lsb

    if not result:
        result = [129]
    result.sort()
    return result
# EVOLVE-BLOCK-END
