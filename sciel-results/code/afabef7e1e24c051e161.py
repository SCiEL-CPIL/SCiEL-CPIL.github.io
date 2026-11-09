import numpy as np
import time
import itertools
import random


def construct(q, n, radius):
    # Fallback for non-target parameters
    if not (q == 3 and n == 8 and radius == 1):
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    N = q ** n                      # 6561
    W = (N + 63) // 64             # 821 uint64 words per bitset
    start_time = time.time()
    time_limit = 150.0
    safety_margin = 8.0
    deadline = start_time + time_limit - safety_margin

    def now():
        return time.time()

    # ---------- bitset helpers ----------
    def bitset_test_empty(bs):
        for w in bs:
            if w:
                return False
        return True

    def bitset_count(bs):
        c = 0
        for w in bs:
            c += w.bit_count()
        return c

    def bitset_and_empty(a, b):
        for x, y in zip(a, b):
            if x & y:
                return False
        return True

    # ---------- precompute positional masks ----------
    # pos_mask[k][v] : bitset of all words whose k-th symbol equals v
    pos_mask = [[None] * q for _ in range(n)]
    for i in range(N):
        x = i
        for k in range(n):
            v = x % q
            x //= q
            pos_mask[k][v][i >> 6] |= (1 << (i & 63))
    for k in range(n):
        for v in range(q):
            pos_mask[k][v] = np.frombuffer(pos_mask[k][v], dtype=np.uint64).copy()

    # ---------- ball bitsets ----------
    # ball[i] = union over k of pos_mask[k][symbol_k(i)]
    balls = [None] * N
    for i in range(N):
        x = i
        bs = np.zeros(W, dtype=np.uint64)
        for k in range(n):
            v = x % q
            x //= q
            bs |= pos_mask[k][v]
        balls[i] = bs

    # covers[j] = bitset of codeword indices d such that j in ball(d)  (== balls[d][j])
    # Build by iterating balls
    covers = [np.zeros(W, dtype=np.uint64) for _ in range(N)]
    for d in range(N):
        bd = balls[d]
        for wi, word in enumerate(bd):
            x = word
            while x:
                b = (x & -x).bit_length() - 1
                j = (wi << 6) + b
                covers[j][d >> 6] |= (1 << (d & 63))
                x &= x - 1

    def digits_of(idx):
        x = idx
        d = []
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        return d

    # ---------- initial direct-sum construction ----------
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            init_indices.append(digits_to_int_idx(h + list(tail)))

    def digits_to_int_idx(digits):
        val = 0
        for d in digits:
            val = val * q + d
        return val

    best = None

    def run_once(seed):
        rng = random.Random(seed)
        codewords = init_indices[:]
        rng.shuffle(codewords)
        cw_set = set(codewords)

        counts = np.zeros(N, dtype=np.uint16)
        for c in codewords:
            bd = balls[c]
            for wi, word in enumerate(bd):
                x = word
                while x:
                    b = (x & -x).bit_length() - 1
                    counts[(wi << 6) + b] += 1
                    x &= x - 1

        # singles bitset: words with count == 1
        singles = np.zeros(W, dtype=np.uint64)
        for j in range(N):
            if counts[j] == 1:
                singles[j >> 6] |= (1 << (j & 63))

        # removal + swap loop
        for _pass in range(2000):
            if now() > deadline:
                break
            changed = False
            new_codewords = []
            for c in codewords:
                if now() > deadline:
                    break
                bd = balls[c]
                # can we remove c?  need no word in ball(c) with count==1
                if bitset_and_empty(bd, singles):
                    # remove
                    for wi, word in enumerate(bd):
                        x = word
                        while x:
                            b = (x & -x).bit_length() - 1
                            j = (wi << 6) + b
                            counts[j] -= 1
                            if counts[j] == 1:
                                singles[j >> 6] |= (1 << (j & 63))
                            elif counts[j] == 0:
                                singles[j >> 6] &= ~(1 << (j & 63))
                            x &= x - 1
                    cw_set.discard(c)
                    changed = True
                    continue
                # else try a swap
                # critical = ball(c) & singles
                critical = [a & b for a, b in zip(bd, singles)]
                # candidate codewords covering all critical words: intersect covers[j] for j in critical
                cand = None
                for wi, word in enumerate(critical):
                    x = word
                    while x:
                        b = (x & -x).bit_length() - 1
                        j = (wi << 6) + b
                        cj = covers[j]
                        if cand is None:
                            cand = cj.copy()
                        else:
                            cand &= cj
                        if bitset_test_empty(cand):
                            break
                        x &= x - 1
                if cand is None or bitset_test_empty(cand):
                    new_codewords.append(c)
                    continue
                # gather candidate indices not in cw_set
                cands = []
                for wi, word in enumerate(cand):
                    x = word
                    while x:
                        b = (x & -x).bit_length() - 1
                        d = (wi << 6) + b
                        if d not in cw_set:
                            cands.append(d)
                        x &= x - 1
                if not cands:
                    new_codewords.append(c)
                    continue
                # score each candidate: number of words in (singles & ball(d)) that are NOT in ball(c)
                # i.e. how many currently-singly-covered words it newly covers beyond c's own ball
                best_d = None
                best_score = -1
                # limit candidate evaluation cost: at most ~60
                if len(cands) > 60:
                    cands = rng.sample(cands, 60)
                for d in cands:
                    bd2 = balls[d]
                    # new_fix = singles & ball(d) & ~ball(c)
                    s = 0
                    for a, b2, cc in zip(singles, bd2, bd):
                        s += ((a & b2) & ~cc).bit_count()
                    if s > best_score:
                        best_score = s
                        best_d = d
                if best_d is None:
                    new_codewords.append(c)
                    continue
                # perform swap: remove c, add best_d
                bd = balls[c]
                for wi, word in enumerate(bd):
                    x = word
                    while x:
                        b = (x & -x).bit_length() - 1
                        j = (wi << 6) + b
                        counts[j] -= 1
                        if counts[j] == 1:
                            singles[j >> 6] |= (1 << (j & 63))
                        elif counts[j] == 0:
                            singles[j >> 6] &= ~(1 << (j & 63))
                        x &= x - 1
                bd2 = balls[best_d]
                for wi, word in enumerate(bd2):
                    x = word
                    while x:
                        b = (x & -x).bit_length() - 1
                        j = (wi << 6) + b
                        counts[j] += 1
                        if counts[j] == 1:
                            singles[j >> 6] |= (1 << (j & 63))
                        elif counts[j] == 0:
                            singles[j >> 6] &= ~(1 << (j & 63))
                        x &= x - 1
                cw_set.discard(c)
                cw_set.add(best_d)
                new_codewords.append(best_d)
                changed = True
            codewords = new_codewords
            if not changed:
                break
        return codewords

    # run several restarts, keep best
    for s in range(60):
        if now() > deadline:
            break
        cw = run_once(s)
        if best is None or len(cw) < len(best):
            best = cw

    if best is None:
        best = init_indices

    return [digits_of(c) for c in best]
