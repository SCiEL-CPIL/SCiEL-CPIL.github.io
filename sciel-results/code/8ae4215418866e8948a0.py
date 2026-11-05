import time
import random
import math
import itertools
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0

    if n != 7:
        return _standard_construct(n)

    # 1. Generate permutations of 1..6
    perms = list(itertools.permutations(range(1, 7)))
    N = len(perms)  # 720

    # 2. Compute Overlap Matrix using encoding for speed
    suf5 = np.zeros(N, dtype=np.int32)
    pre5 = np.zeros(N, dtype=np.int32)
    suf4 = np.zeros(N, dtype=np.int32)
    pre4 = np.zeros(N, dtype=np.int32)
    suf3 = np.zeros(N, dtype=np.int32)
    pre3 = np.zeros(N, dtype=np.int32)
    suf2 = np.zeros(N, dtype=np.int32)
    pre2 = np.zeros(N, dtype=np.int32)
    suf1 = np.zeros(N, dtype=np.int32)
    pre1 = np.zeros(N, dtype=np.int32)

    for i in range(N):
        p = perms[i]
        c = 0
        for k in range(1, 6):
            c = c * 7 + p[k]
        suf5[i] = c
        c = 0
        for k in range(0, 5):
            c = c * 7 + p[k]
        pre5[i] = c
        c = 0
        for k in range(2, 6):
            c = c * 7 + p[k]
        suf4[i] = c
        c = 0
        for k in range(0, 4):
            c = c * 7 + p[k]
        pre4[i] = c
        c = 0
        for k in range(3, 6):
            c = c * 7 + p[k]
        suf3[i] = c
        c = 0
        for k in range(0, 3):
            c = c * 7 + p[k]
        pre3[i] = c
        c = 0
        for k in range(4, 6):
            c = c * 7 + p[k]
        suf2[i] = c
        c = 0
        for k in range(0, 2):
            c = c * 7 + p[k]
        pre2[i] = c
        suf1[i] = p[5]
        pre1[i] = p[0]

    # Build M matrix vectorized
    M = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        m5 = (pre5 == suf5[i])
        M[i, m5] = 5
        m4 = (pre4 == suf4[i]) & ~m5
        M[i, m4] = 4
        m3 = (pre3 == suf3[i]) & ~m5 & ~m4
        M[i, m3] = 3
        m2 = (pre2 == suf2[i]) & ~m5 & ~m4 & ~m3
        M[i, m2] = 2
        m1 = (pre1 == suf1[i]) & ~m5 & ~m4 & ~m3 & ~m2
        M[i, m1] = 1

    # 3. Greedy Initialization
    order = np.zeros(N, dtype=np.int64)
    used = np.zeros(N, dtype=bool)
    order[0] = 0
    used[0] = True
    for pos in range(1, N):
        last = order[pos - 1]
        best_next = -1
        best_ov = -1
        for j in range(N):
            if not used[j]:
                ov = M[last, j]
                if ov > best_ov:
                    best_ov = ov
                    best_next = j
        order[pos] = best_next
        used[best_next] = True

    def calc_score(ord_list):
        s = 0
        for i in range(N - 1):
            s += M[ord_list[i], ord_list[i + 1]]
        return s

    current_score = calc_score(order)
    best_order = order.copy()
    best_score = current_score

    # 4. Simulated Annealing
    T = 2.0
    cooling_rate = 0.9997
    min_T = 1e-4
    random.seed(42)

    while T > min_T:
        if time.time() > deadline - 5.0:
            break

        r = random.random()

        if r < 0.5:
            # 2-opt: reverse segment
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)

            delta = 0
            if i > 0:
                delta += M[order[i - 1], order[j]] - M[order[i - 1], order[i]]
            if j < N - 1:
                delta += M[order[i], order[j + 1]] - M[order[j], order[j + 1]]
            for k in range(i, j):
                delta += M[order[k + 1], order[k]] - M[order[k], order[k + 1]]

            if delta > 0 or random.random() < math.exp(delta / T):
                order[i:j + 1] = order[i:j + 1][::-1]
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_order = order.copy()

        else:
            # Swap two positions
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            a = order[i]
            b = order[j]

            delta = 0
            if i > 0:
                delta += M[order[i - 1], b] - M[order[i - 1], a]
            if j < N - 1:
                delta += M[b, order[j + 1]] - M[a, order[j + 1]]
            if i + 1 == j:
                delta += M[b, a] - M[a, b]
            else:
                delta += M[b, order[i + 1]] - M[a, order[i + 1]]
                delta += M[order[j - 1], a] - M[order[j - 1], b]

            if delta > 0 or random.random() < math.exp(delta / T):
                order[i] = b
                order[j] = a
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_order = order.copy()

        T *= cooling_rate

    # 5. Construct Word
    word = []
    for idx, p_idx in enumerate(best_order):
        p = perms[int(p_idx)]
        block = list(p) + [7] + list(p)
        if idx == 0:
            word = block
        else:
            ov = int(M[best_order[idx - 1], best_order[idx]])
            word += block[ov:]

    return "".join(map(str, word))

def _standard_construct(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
