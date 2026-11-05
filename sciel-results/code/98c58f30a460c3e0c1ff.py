"""Fresh design: heap-accelerated greedy set cover + local search + polish."""
import time
import heapq
import random
from collections import defaultdict

import numpy as np


def _ball_indices(q, n, radius, word):
    """Indices of all words within Hamming distance <= radius of word."""
    word = np.asarray(word)
    idx = [int(np.ravel_multi_index(tuple(word), (q,) * n))]
    if radius >= 1:
        for pos in range(n):
            old = word[pos]
            for v in range(q):
                if v == old:
                    continue
                word[pos] = v
                idx.append(int(np.ravel_multi_index(tuple(word), (q,) * n)))
            word[pos] = old
    if radius >= 2:
        for p1 in range(n):
            for p2 in range(p1 + 1, n):
                a, b = word[p1], word[p2]
                for v1 in range(q):
                    if v1 == a:
                        continue
                    for v2 in range(q):
                        if v2 == b:
                            continue
                        word[p1] = v1
                        word[p2] = v2
                        idx.append(int(np.ravel_multi_index(tuple(word), (q,) * n)))
                word[p1] = a
                word[p2] = b
    return idx


def construct(q, n, radius):
    start = time.time()
    DEADLINE = start + 130.0  # safety margin under 150s

    total = q ** n
    # Precompute ball (coverage set) for every candidate codeword.
    balls = [_ball_indices(q, n, radius, w) for w in range(total)]

    # ---- Greedy set cover (max-marginal) with heap ----
    # covered[w] = number of chosen balls covering word w
    covered = np.zeros(total, dtype=np.int32)
    # max-heap via negative counts; entry = (-count, word)
    heap = [(-1, w) for w in range(total)]
    heapq.heapify(heap)
    chosen = []
    while True:
        if time.time() > DEADLINE:
            break
        neg, w = heapq.heappop(heap)
        cnt = -neg
        if cnt < 1:
            break
        if covered[w] != 0:
            continue  # stale
        # commit
        chosen.append(w)
        for c in balls[w]:
            covered[c] += 1
            if covered[c] == 1:
                heapq.heappush(heap, (-1, c))
        # stop when all covered
        if covered.min() >= 1:
            break
    if covered.min() < 1:
        # safety: brute-force complete the cover
        for w in range(total):
            if covered[w] == 0:
                chosen.append(w)
                for c in balls[w]:
                    covered[c] += 1

    # ---- Local search: 1-swap then 2-swap to reduce |chosen| ----
    chosen = list(dict.fromkeys(chosen))  # dedupe, keep order

    def _covers_all(chosen_set):
        cov = np.zeros(total, dtype=np.uint8)
        for w in chosen_set:
            cov[balls[w]] = 1
        return bool(cov.all())

    def _try_remove(chosen_list):
        # try to drop one word that is redundant
        improved = True
        while improved:
            if time.time() > DEADLINE:
                return chosen_list
            improved = False
            cov = np.zeros(total, dtype=np.int32)
            for w in chosen_list:
                cov[balls[w]] += 1
            for i in range(len(chosen_list) - 1, -1, -1):
                for c in balls[chosen_list[i]]:
                    cov[c] -= 1
                if cov.min() >= 1:
                    chosen_list.pop(i)
                    improved = True
                    break
                for c in balls[chosen_list[i]]:
                    cov[c] += 1
        return chosen_list

    chosen = _try_remove(chosen)

    def _try_swap(chosen_list, max_iter=20000):
        # replace one word with another to allow removing a second word
        improved = True
        while improved:
            if time.time() > DEADLINE:
                return chosen_list
            improved = False
            cov = np.zeros(total, dtype=np.int32)
            for w in chosen_list:
                cov[balls[w]] += 1
            # find a word whose removal leaves exactly one uncovered word
            for i in range(len(chosen_list)):
                for c in balls[chosen_list[i]]:
                    cov[c] -= 1
                if cov.min() >= 1:
                    # redundant, just remove
                    chosen_list.pop(i)
                    improved = True
                    break
                # find uncovered words
                uncov = np.where(cov == 0)[0]
                if len(uncov) == 1:
                    target = int(uncov[0])
                    # find a word covering target not already chosen
                    cand = None
                    for w in range(total):
                        if w not in set(chosen_list) and target in balls[w]:
                            cand = w
                            break
                    if cand is not None:
                        chosen_list[i] = cand
                        improved = True
                        break
                for c in balls[chosen_list[i]]:
                    cov[c] += 1
                if improved:
                    break
        return chosen_list

    chosen = _try_swap(chosen)
    chosen = _try_remove(chosen)

    # ---- Polish passes ----
    while time.time() < DEADLINE:
        before = len(chosen)
        chosen = _try_remove(chosen)
        chosen = _try_swap(chosen)
        chosen = _try_remove(chosen)
        if len(chosen) == before:
            break

    # ---- Build codewords from indices ----
    out = []
    for w in chosen:
        out.append(list(np.unravel_index(w, (q,) * n)))
    # dedupe
    seen = set()
    final = []
    for c in out:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            final.append(c)
    return final
