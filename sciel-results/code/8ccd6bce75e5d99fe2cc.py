import itertools
import random
import time
import numpy as np


def construct(q, n, radius):
    start = time.time()
    DEADLINE = 145.0          # wall-clock safety margin (evaluator kills at 150s)

    def time_left():
        return DEADLINE - (time.time() - start)

    # ---- fallback for non-target parameters (keep program robust) ----
    if not (q == 3 and n == 8 and radius == 1):
        all_words = list(itertools.product(range(q), repeat=n))
        N = len(all_words)
        idx = {w: i for i, w in enumerate(all_words)}
        neigh = []
        for w in all_words:
            s = {idx[w]}
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    lst = list(w); lst[pos] = val
                    s.add(idx[tuple(lst)])
            neigh.append(np.array(sorted(s), dtype=np.int64))
        # greedy set cover
        uncovered = np.ones(N, dtype=bool)
        C = []
        while np.any(uncovered) and time_left() > 5.0:
            gains = np.array([int(np.sum(uncovered[neigh[i]])) for i in range(N)], dtype=np.int64)
            bi = int(np.argmax(gains))
            if gains[bi] <= 0:
                break
            C.append(bi)
            uncovered[neigh[bi]] = False
        # reduction
        def reduce(C):
            counts = np.zeros(N, dtype=np.int64)
            for c in C:
                counts[neigh[c]] += 1
            while True:
                red = False
                if C:
                    mn = np.minimum.reduceat(np.concatenate([counts[neigh[c]] for c in C]),
                                             np.arange(0, len(C) * 17, 17))
                    red = mn < 2
                if not red.any():
                    break
                keep = np.where(red)[0]
                counts = np.zeros(N, dtype=np.int64)
                for c in keep:
                    counts[neigh[c]] += 1
                C = [C[i] for i in keep]
            return C, counts
        C, _ = reduce(C)
        return [list(all_words[i]) for i in C]

    # =================  q = 3, n = 8, radius = 1  =================
    N = 3 ** 8  # 6561
    all_words = list(itertools.product(range(3), repeat=8))

    # ---- fast vectorized neighbor table (each word has 1 + 8*2 = 17 neighbors) ----
    base = np.arange(N, dtype=np.int64)
    pos = np.repeat(np.arange(n, dtype=np.int64), 2)
    delta = np.tile([-1, 1], n)
    base_rep = np.repeat(base, 2 * n)
    neigh = np.empty((N, 17), dtype=np.int64)
    neigh[:, 0] = base
    neigh[:, 1:] = (base_rep + delta) % N
    neigh = np.ascontiguousarray(neigh)

    # ---- coverage helpers (vectorized) ----
    def compute_counts(C):
        counts = np.zeros(N, dtype=np.int64)
        for c in C:
            counts[neigh[c]] += 1
        return counts

    def reduce(C, counts):
        """Remove every redundant codeword (one whose neighbors are all covered >=2)."""
        while True:
            if not C:
                break
            mn = np.minimum.reduceat(np.concatenate([counts[neigh[c]] for c in C]),
                                     np.arange(0, len(C) * 17, 17))
            red = mn < 2
            if not red.any():
                break
            keep = np.where(red)[0]
            counts = np.zeros(N, dtype=np.int64)
            for c in keep:
                counts[neigh[c]] += 1
            C = [C[i] for i in keep]
        return C, counts

    # ---- initial solution: greedy set cover ----
    uncovered = np.ones(N, dtype=bool)
    C = []
    while np.any(uncovered) and time_left() > 20.0:
        gains = np.array([int(np.sum(uncovered[neigh[i]])) for i in range(N)], dtype=np.int64)
        bi = int(np.argmax(gains))
        if gains[bi] <= 0:
            break
        C.append(bi)
        uncovered[neigh[bi]] = False
    C, counts = reduce(C, compute_counts(C))

    best_C = C[:]
    best_counts = counts.copy()
    best_size = len(C)
    cur_size = best_size

    # ---- simulated annealing local search ----
    T0 = 1.0
    T = T0
    cool = 0.9999
    TMIN = 0.05
    n_iter = 0

    while time_left() > 1.0:
        # 1) try to remove a redundant word (always an improvement)
        if C:
            mn = np.minimum.reduceat(np.concatenate([counts[neigh[c]] for c in C]),
                                     np.arange(0, len(C) * 17, 17))
            red = np.where(mn < 2)[0]
            if red.size:
                r = int(red[random.randrange(len(red))])
                counts[neigh[C[r]]] -= 1
                C.pop(r)
                cur_size = len(C)
                if cur_size < best_size:
                    best_size = cur_size
                    best_C = C[:]
                    best_counts = counts.copy()
                n_iter += 1
                continue

        # 2) random move: swap a word for a random neighbor
        if not C:
            break
        i = random.randrange(len(C))
        c = C[i]
        c2 = int(neigh[c, random.randrange(17)])
        if c2 == c or c2 in C:
            n_iter += 1
            continue

        # feasibility of the swap
        counts[neigh[c]] -= 1
        if np.any(counts < 1):
            counts[neigh[c]] += 1
            n_iter += 1
            continue
        counts[neigh[c2]] += 1

        # reduce to a canonical minimal state
        C[i] = c2
        C, counts = reduce(C, counts)
        cur_size = len(C)

        if cur_size < best_size:
            best_size = cur_size
            best_C = C[:]
            best_counts = counts.copy()
        else:
            # accept worse states to escape local minima (annealing)
            if T < TMIN or random.random() < np.exp(-(cur_size - best_size) / T):
                pass  # keep the worse state
            else:
                # revert to best state
                C = best_C[:]
                counts = best_counts.copy()
                cur_size = best_size

        n_iter += 1
        if n_iter % 200 == 0:
            T = max(T * cool, TMIN)

    # ---- final canonical reduction of the best solution ----
    best_C, _ = reduce(best_C, compute_counts(best_C))
    return [list(all_words[i]) for i in best_C]
