# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 120s limit

    if n == 0:
        return [[]]

    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)

    # Fallback for other cases
    if q == 3 and radius == 1 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return _greedy_covering(q, n, radius, deadline)


def _solve_q3_n6_r1(deadline):
    q = 3
    n = 6
    N = q ** n  # 729
    words = list(itertools.product(range(q), repeat=n))
    idx_to_word = {i: words[i] for i in range(N)}

    # Precompute radius-1 balls as 64-bit masks (12 u64 words each)
    nblocks = (N + 63) // 64
    balls = np.zeros((N, nblocks), dtype=np.uint64)
    for i, w in enumerate(words):
        m = balls[i]
        m[i >> 6] |= np.uint64(1) << (i & 63)
        for pos in range(n):
            base = list(w)
            for v in range(q):
                if v == base[pos]:
                    continue
                base[pos] = v
                j = int(np.ravel_multi_index(tuple(base), (q,) * n))
                m[j >> 6] |= np.uint64(1) << (j & 63)
                base[pos] = w[pos]

    def greedy(seed):
        rng = random.Random(seed)
        covered = np.zeros(nblocks, dtype=np.uint64)
        uncovered = np.full(nblocks, np.uint64((1 << 64) - 1), dtype=np.uint64)
        uncovered[-1] &= np.uint64((1 << (N & 63)) - 1) if (N & 63) else np.uint64((1 << 64) - 1)
        code = []
        while True:
            if not uncovered.any():
                break
            scores = np.sum(uncovered[None, :] & balls, axis=1)
            best = int(np.max(scores))
            if best == 0:
                break
            cands = np.where(scores == best)[0]
            c = int(cands[rng.randrange(len(cands))])
            code.append(c)
            uncovered &= ~balls[c]
        return code

    def redundant_counts(code):
        counts = np.zeros(N, dtype=np.int32)
        for c in code:
            counts[balls[c].nonzero()[0]] += 1
        return counts

    def is_redundant(c, counts):
        nz = balls[c].nonzero()[0]
        return bool(np.all(counts[nz] >= 2))

    best_code = None
    best_size = 10**9
    seed = 1
    while time.time() < deadline - 2.0:
        # Greedy start
        code = greedy(seed)
        counts = redundant_counts(code)

        # Remove all redundants (iterate to fixed point)
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            for c in list(code):
                if is_redundant(c, counts):
                    nz = balls[c].nonzero()[0]
                    counts[nz] -= 1
                    code.remove(c)
                    changed = True
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
        if best_size <= 73:
            # keep searching a bit for 72/71, but bail if out of time
            if time.time() > deadline - 8.0:
                break

        # ILS: repair via add + 1-out / 2-out
        if time.time() < deadline - 2.0:
            cur = list(code)
            cur_counts = counts.copy()
            cur_best = len(cur)
            for _ in range(30):
                if time.time() > deadline - 2.0:
                    break
                # remove redundants
                for c in list(cur):
                    if is_redundant(c, cur_counts):
                        nz = balls[c].nonzero()[0]
                        cur_counts[nz] -= 1
                        cur.remove(c)
                if len(cur) < cur_best:
                    cur_best = len(cur)
                # stuck? add a random word
                while True:
                    r = random.randrange(N)
                    if r not in cur:
                        break
                cur.append(r)
                nz = balls[r].nonzero()[0]
                cur_counts[nz] += 1
                # find redundants
                red = [c for c in cur if is_redundant(c, cur_counts)]
                if len(red) >= 2:
                    a = red[0]
                    nz_a = balls[a].nonzero()[0]
                    cur_counts[nz_a] -= 1
                    cur.remove(a)
                    b = None
                    for c in red[1:]:
                        if c != a and is_redundant(c, cur_counts):
                            b = c
                            break
                    if b is not None:
                        nz_b = balls[b].nonzero()[0]
                        cur_counts[nz_b] -= 1
                        cur.remove(b)
                elif len(red) == 1:
                    a = red[0]
                    nz_a = balls[a].nonzero()[0]
                    cur_counts[nz_a] -= 1
                    cur.remove(a)
                if len(cur) < cur_best:
                    cur_best = len(cur)
                if len(cur) > cur_best:
                    # revert to best within this ILS run
                    cur = list(best_code)
                    cur_counts = redundant_counts(cur)
                    cur_best = len(cur)
                    break
            if len(cur) < best_size:
                best_size = len(cur)
                best_code = list(cur)
        seed += 1

    # Final validity sweep: remove redundants from best_code
    if best_code is None:
        best_code = greedy(0)
    counts = redundant_counts(best_code)
    changed = True
    while changed:
        changed = False
        for c in list(best_code):
            if is_redundant(c, counts):
                nz = balls[c].nonzero()[0]
                counts[nz] -= 1
                best_code.remove(c)
                changed = True
    return [list(idx_to_word[i]) for i in best_code]


def _local_search_optimize(code_indices, neighbors, N, deadline):
    # Compatibility shim: not used by the main path, but kept for interface stability.
    return list(code_indices)


def _greedy_covering(q, n, radius, deadline):
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    if n <= 8:
        all_words = list(itertools.product(range(q), repeat=n))
        uncovered = set(all_words)
        code = []
        while uncovered and time.time() < deadline:
            best_word = None
            best_count = -1
            for w in all_words:
                lst = [w]
                cnt = 0
                for pos in range(n):
                    for v in range(q):
                        if v == w[pos]:
                            continue
                        lw = list(w)
                        lw[pos] = v
                        if tuple(lw) in uncovered:
                            cnt += 1
                if tuple(w) in uncovered:
                    cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_word = w
            if best_word is None or best_count == 0:
                break
            code.append(list(best_word))
            # remove covered
            covered = {tuple(best_word)}
            for pos in range(n):
                for v in range(q):
                    if v == best_word[pos]:
                        continue
                    lw = list(best_word)
                    lw[pos] = v
                    covered.add(tuple(lw))
            uncovered -= covered
        return code
    return [[0] * n]
# EVOLVE-BLOCK-END
