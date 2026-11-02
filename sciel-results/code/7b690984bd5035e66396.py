# EVOLVE-BLOCK-START
import time
import numpy as np


def _build_tables(q, n, radius):
    total = q ** n
    digs = np.empty((total, n), dtype=np.int32)
    for i in range(total):
        v = i
        for j in range(n):
            digs[i, j] = v % q
            v //= q
    # radius-1 ball: self + one-coordinate changes
    ball = np.empty((total, 1 + n * (q - 1)), dtype=np.int32)
    ball[:, 0] = np.arange(total)
    pos = 1
    for j in range(n):
        pj = q ** j
        for d in range(q):
            col = np.arange(total, dtype=np.int64) - digs[:, j] * pj + d * pj
            ball[:, pos] = col.astype(np.int32)
            pos += 1
    return total, digs, ball


def construct(q, n, radius):
    t0 = time.time()
    DEADLINE = 150.0 - 5.0

    total, digs, ball = _build_tables(q, n, radius)

    cnt = np.zeros(total, dtype=np.int32)
    chosen = []
    chosen_set = set()

    def add_codeword(w):
        for b in ball[w]:
            cnt[b] += 1
        chosen.append(w)
        chosen_set.add(w)

    def remove_codeword(w):
        for b in ball[w]:
            cnt[b] -= 1
        chosen.remove(w)
        chosen_set.discard(w)

    def uncovered_count():
        return int(np.count_nonzero(cnt == 0))

    # ---------- GREEDY BUILD ----------
    uncovered = np.arange(total, dtype=np.int32)

    def greedy_step():
        if len(uncovered) == 0:
            return -1
        K = 8
        if len(uncovered) <= K:
            idxs = uncovered
        else:
            idxs = np.random.choice(uncovered, size=K, replace=False)
        best = -1
        best_gain = -1
        for w in idxs:
            nb = ball[w]
            gain = int(np.count_nonzero(cnt[nb] == 0))
            if gain > best_gain:
                best_gain = gain
                best = int(w)
        return best

    while True:
        if uncovered_count() == 0:
            break
        w = greedy_step()
        if w < 0:
            break
        add_codeword(w)
        uncovered = np.nonzero(cnt == 0)[0]
        if time.time() > DEADLINE - 60:
            break

    if uncovered_count() != 0:
        for w in range(total):
            if uncovered_count() == 0:
                break
            if w not in chosen_set:
                add_codeword(w)

    # ---------- REDUNDANCY REMOVAL ----------
    def try_remove_pass():
        improved = True
        while improved:
            improved = False
            for w in list(chosen):
                for b in ball[w]:
                    cnt[b] -= 1
                if np.count_nonzero(cnt == 0) == 0:
                    chosen.remove(w)
                    chosen_set.discard(w)
                    improved = True
                else:
                    for b in ball[w]:
                        cnt[b] += 1
                if time.time() > DEADLINE - 30:
                    return

    try_remove_pass()

    # ---------- SWAP LOCAL SEARCH ----------
    def swap_pass():
        improved = True
        while improved:
            improved = False
            for i in range(len(chosen)):
                w = chosen[i]
                for _ in range(3):
                    cand = int(np.random.randint(total))
                    if cand in chosen_set:
                        continue
                    for b in ball[w]:
                        cnt[b] -= 1
                    for b in ball[cand]:
                        cnt[b] += 1
                    if np.count_nonzero(cnt == 0) == 0:
                        found = False
                        for v in list(chosen):
                            for b in ball[v]:
                                cnt[b] -= 1
                            if np.count_nonzero(cnt == 0) == 0:
                                chosen.remove(v)
                                chosen_set.discard(v)
                                found = True
                                break
                            for b in ball[v]:
                                cnt[b] += 1
                        if found:
                            improved = True
                            break
                        else:
                            for b in ball[w]:
                                cnt[b] += 1
                            for b in ball[cand]:
                                cnt[b] -= 1
                    else:
                        for b in ball[w]:
                            cnt[b] += 1
                        for b in ball[cand]:
                            cnt[b] -= 1
                if time.time() > DEADLINE - 20:
                    return

    for _ in range(3):
        if time.time() > DEADLINE - 20:
            break
        swap_pass()
        if time.time() > DEADLINE - 20:
            break
        try_remove_pass()

    try_remove_pass()

    if np.count_nonzero(cnt == 0) != 0:
        for w in range(total):
            if np.count_nonzero(cnt == 0) == 0:
                break
            if w not in chosen_set:
                add_codeword(w)

    result = []
    for w in chosen:
        result.append(digs[w].tolist())
    return result
# EVOLVE-BLOCK-END
