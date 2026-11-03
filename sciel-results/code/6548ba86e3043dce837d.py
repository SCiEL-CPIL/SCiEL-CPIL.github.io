import time
import itertools
import numpy as np
import random

_DEADLINE = 140.0

def _build_tables(q, n):
    total = q ** n
    per = 1 + n * (q - 1)
    covers = np.empty((total, per), dtype=np.int32)
    pre = [None] * total
    for c in range(total):
        digits = [(c // (q ** i)) % q for i in range(n)]
        idx = 0
        covers[c, idx] = c; idx += 1
        for pos in range(n):
            for v in range(q):
                if v == digits[pos]:
                    continue
                w = c + (v - digits[pos]) * (q ** pos)
                covers[c, idx] = w; idx += 1
        for w in covers[c]:
            if pre[w] is None:
                pre[w] = []
            pre[w].append(c)
    return covers, pre

def construct(q, n, radius):
    start = time.time()
    total = q ** n
    covers, pre = _build_tables(q, n)

    # seed
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        seed = set()
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                word = h + list(tail)
                seed.add(sum(word[i] * (3 ** i) for i in range(n)))
    else:
        seed = set(range(total))

    in_code = np.zeros(total, dtype=bool)
    cnt = np.zeros(total, dtype=np.int32)
    active = list(seed)
    for c in active:
        in_code[c] = True
        cnt[covers[c]] += 1

    # repair: ensure validity
    def repair():
        while True:
            uncov = np.where(cnt == 0)[0]
            if len(uncov) == 0:
                break
            # pick codeword covering most uncovered
            gain = (cnt[covers] == 0).sum(axis=1)
            b = int(np.argmax(gain))
            if gain[b] == 0:
                break
            in_code[b] = True
            cnt[covers[b]] += 1
            active.append(b)

    repair()

    best_size = len(active)
    best_set = set(active)

    def is_removable(c):
        return bool(np.all(cnt[covers[c]] > 1))

    def removal_pass():
        nonlocal best_size, best_set
        changed = True
        while changed:
            changed = False
            for i in range(len(active) - 1, -1, -1):
                c = active[i]
                if is_removable(c):
                    cnt[covers[c]] -= 1
                    in_code[c] = False
                    active.pop(i)
                    changed = True
            if len(active) < best_size:
                best_size = len(active)
                best_set = set(active)

    def random_valid_swap():
        if not active:
            return
        a = active[np.random.randint(len(active))]
        lost = [int(w) for w in covers[a] if cnt[w] == 1]
        if not lost:
            return
        cand = set(pre[lost[0]])
        for w in lost[1:]:
            cand &= set(pre[w])
        cand -= set(active)
        if not cand:
            return
        b = next(iter(cand))
        cnt[covers[a]] -= 1
        in_code[a] = False
        active.remove(a)
        cnt[covers[b]] += 1
        in_code[b] = True
        active.append(b)

    random.seed(12345)
    np.random.seed(12345)

    removal_pass()

    while time.time() - start < _DEADLINE - 5:
        for _ in range(80):
            random_valid_swap()
        removal_pass()

    result = []
    for c in best_set:
        result.append([(c // (q ** i)) % q for i in range(n)])
    return result
