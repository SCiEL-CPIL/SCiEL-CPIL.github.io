"""Rewritten solver for the ternary covering-code task K_3(7,1).

Set-cover over the 2187-word space, then local search (prune + swap), all under a
hard wall-clock deadline. Function name and return interface are preserved.
"""
# EVOLVE-BLOCK-START
import time
import random
import itertools


def construct(q, n, radius):
    deadline = time.time() + 145.0
    num = q ** n
    num_pos = n * (q - 1)
    ball_size = num_pos + 1

    # closed radius-1 ball of every word as a Python-int bitset
    balls = [0] * num
    for x in range(num):
        mask = 1 << x
        ds = []
        t = x
        for _ in range(n):
            ds.append(t % q)
            t //= q
        ds = ds[::-1]
        for pos in range(n):
            d = ds[pos]
            mult = q ** (n - 1 - pos)
            for nv in range(q):
                if nv != d:
                    mask |= 1 << (x + (nv - d) * mult)
        balls[x] = mask

    # greedy set cover
    covered = 0
    uncov_bits = (1 << num) - 1
    C = []
    for _ in range(num):
        best = -1
        best_gain = 0
        for c in range(num):
            gain = (balls[c] & uncov_bits).bit_count()
            if gain > best_gain:
                best_gain = gain
                best = c
                if best_gain >= ball_size:
                    break
        if best < 0 or best_gain == 0:
            break
        C.append(best)
        covered |= balls[best]
        uncov_bits &= ~covered
        if uncov_bits == 0:
            break

    def build_cov(C):
        cov = [0] * num
        for c in C:
            m = balls[c]
            while m:
                lsb = m & (-m)
                cov[lsb.bit_length() - 1] += 1
                m ^= lsb
        return cov

    best_C = C[:]
    best_len = len(C)

    def optimize(C):
        nonlocal best_C, best_len
        cov = build_cov(C)
        C_set = set(C)
        while True:
            if time.time() > deadline:
                break
            changed = False
            i = len(C)
            while i > 0:
                i -= 1
                c = C[i]
                m = balls[c]
                bad = False
                while m:
                    lsb = m & (-m)
                    if cov[lsb.bit_length() - 1] <= 1:
                        bad = True
                        break
                    m ^= lsb
                if not bad:
                    m = balls[c]
                    while m:
                        lsb = m & (-m)
                        cov[lsb.bit_length() - 1] -= 1
                        m ^= lsb
                    C.pop(i)
                    C_set.discard(c)
                    changed = True
            if len(C) < best_len:
                best_len = len(C)
                best_C = C[:]
            if not changed:
                break
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(C)):
                if time.time() > deadline:
                    break
                c_out = C[i]
                exposed = []
                m = balls[c_out]
                while m:
                    lsb = m & (-m)
                    j = lsb.bit_length() - 1
                    if cov[j] == 2:
                        exposed.append(j)
                    m ^= lsb
                if not exposed:
                    best_in = -1
                    best_gain = -1
                    for c in range(num):
                        if c in C_set:
                            continue
                        gain = (balls[c] & covered).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_in = c
                    if best_in < 0:
                        continue
                    C[i] = best_in
                    C_set.remove(c_out)
                    C_set.add(best_in)
                    covered |= balls[best_in]
                    m = balls[c_out]
                    while m:
                        lsb = m & (-m)
                        cov[lsb.bit_length() - 1] -= 1
                        m ^= lsb
                    m = balls[best_in]
                    while m:
                        lsb = m & (-m)
                        cov[lsb.bit_length() - 1] += 1
                        m ^= lsb
                    improved = True
                    continue
                req_mask = 0
                for j in exposed:
                    req_mask |= 1 << j
                best_in = -1
                for c in range(num):
                    if c in C_set:
                        continue
                    if (balls[c] & req_mask) == req_mask:
                        best_in = c
                        break
                if best_in < 0:
                    continue
                C[i] = best_in
                C_set.remove(c_out)
                C_set.add(best_in)
                m = balls[c_out]
                while m:
                    lsb = m & (-m)
                    cov[lsb.bit_length() - 1] -= 1
                    m ^= lsb
                m = balls[best_in]
                while m:
                    lsb = m & (-m)
                    cov[lsb.bit_length() - 1] += 1
                    m ^= lsb
                improved = True
            if len(C) < best_len:
                best_len = len(C)
                best_C = C[:]
        return C

    C = optimize(best_C)
    while time.time() < deadline:
        random.shuffle(C)
        C = optimize(C)

    return [[(best_C[i] // q ** (n - 1 - pos)) % q for pos in range(n)] for i in range(best_len)]
# EVOLVE-BLOCK-END
