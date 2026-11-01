"""Fresh design: greedy covering construction + efficient redundancy removal + randomized restarts."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    deadline = time.time() + 112.0  # safety margin before 120s kill
    rng = random.Random(12345)

    total = q ** n
    pos = np.arange(n, dtype=np.int64)
    all_ints = np.arange(total, dtype=np.int64)
    digs = (all_ints[:, None] // (q ** (n - 1 - pos)[None, :])) % q

    # precompute ball: ball[w] = bool array of words within radius of w
    t0 = time.time()
    ball = np.zeros((total, total), dtype=bool)
    for r in range(radius + 1):
        if r == 0:
            idx = np.arange(total)
            ball[idx, idx] = True
        else:
            for combo in itertools.combinations(range(n), r):
                for c in combo:
                    other = (digs[:, c] + 1 + np.arange(q - 1)) % q
                    wgt = q ** (n - 1 - c)
                    for j in range(q - 1):
                        col = all_ints + (other[:, j] - digs[:, c]) * wgt
                        ball[:, col] = True

    def greedy_randomized(seed_order):
        remaining = np.ones(total, dtype=bool)
        rem_count = total
        cur = []
        cand = seed_order
        while rem_count > 0 and time.time() < deadline - 2:
            cov = ball[cand] & remaining[None, :]
            counts = cov.sum(axis=1)
            if counts.max() == 0:
                break
            best = int(np.argmax(counts))
            cur.append(int(cand[best]))
            newly = ball[cur[-1]] & remaining
            rem_count -= int(newly.sum())
            remaining[newly] = False
        return cur, rem_count

    def remove_redundant(cur):
        # cur: list of ints. Use coverage-count array to test redundancy efficiently.
        cur = list(cur)
        if not cur:
            return cur
        # count how many chosen codewords cover each word
        cnt = np.zeros(total, dtype=np.int32)
        for c in cur:
            cnt[ball[int(c)]] += 1
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(cur)):
                if time.time() >= deadline:
                    break
                c = cur[i]
                # word c is covered by cnt[c] codewords; if cnt[c] >= 2, removing c still leaves c covered
                # but need ALL words covered by c to remain covered. c's ball words each need cnt>=2 after removal.
                # cnt after removing c: cnt' = cnt - ball[c]. Need cnt'[ball[c]] >= 1, i.e. cnt[ball[c]] >= 2.
                if (cnt[ball[c]] >= 2).all():
                    cnt[ball[c]] -= 1
                    cur.pop(i)
                    changed = True
                    break
        return cur

    # initial greedy (deterministic order = all words)
    cur, rem = greedy_randomized(np.arange(total))
    cur = remove_redundant(cur)
    best = list(cur)

    # randomized restarts
    for k in range(200):
        if time.time() >= deadline - 3:
            break
        order = np.arange(total)
        rng.shuffle(order.tolist())
        order = np.array(order, dtype=np.int64)
        cur, rem = greedy_randomized(order)
        if rem > 0:
            continue
        cur = remove_redundant(cur)
        if len(cur) < len(best):
            best = list(cur)

    # final redundant cleanup
    best = remove_redundant(best)

    result = []
    seen = set()
    for c in best:
        c = int(c)
        if c in seen:
            continue
        seen.add(c)
        word = []
        v = c
        for i in range(n - 1, -1, -1):
            word.append(v % q)
            v //= q
        word.reverse()
        result.append(word)
    return result
# EVOLVE-BLOCK-END
