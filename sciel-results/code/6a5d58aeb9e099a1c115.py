import time
import random
import numpy as np

def _analyze(A):
    C = A @ A
    n = A.shape[0]
    maxb = 0
    ssum = 0
    elist = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                b = int(C[i, j])
                elist.append((i, j, b))
                if b > maxb:
                    maxb = b
                ssum += b * b
    return C, maxb, ssum, elist

def construct():
    start = time.time()
    deadline = start + 140.0
    n = 40
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def init_A():
        A = np.zeros((n, n), dtype=np.int32)
        for a in range(20):
            for b in range(20, 40):
                A[a, b] = 1
                A[b, a] = 1
        A[0, 1] = 1
        A[1, 0] = 1
        return A

    best_A = None
    best_obj = float('inf')
    best_maxb = 10 ** 9

    def objective(maxb, ssum):
        return maxb + 0.001 * ssum

    restarts = 0
    while time.time() < deadline:
        restarts += 1
        A = init_A()
        if restarts > 1:
            for _ in range(random.randint(0, 8)):
                i, j = random.choice(all_pairs)
                if A[i, j]:
                    A[i, j] = 0
                    A[j, i] = 0
                    while True:
                        k, l = random.choice(all_pairs)
                        if not A[k, l]:
                            A[k, l] = 1
                            A[l, k] = 1
                            break
        C, maxb, ssum, elist = _analyze(A)
        cur_obj = objective(maxb, ssum)
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_A = A.copy()
            best_maxb = maxb
            if best_maxb <= 7:
                break

        T = 2.0
        while time.time() < deadline - 2.0:
            C, maxb, ssum, elist = _analyze(A)
            if not elist:
                break
            wmax = max(b for _, _, b in elist)
            weights = [(b + 1) ** 2 for _, _, b in elist]
            total_w = sum(weights)
            r = random.random() * total_w
            acc = 0.0
            chosen = elist[0]
            for idx, (i, j, b) in enumerate(elist):
                acc += weights[idx]
                if r <= acc:
                    chosen = (i, j, b)
                    break
            i, j, _ = chosen
            nonedges = [e for e in all_pairs if not A[e[0], e[1]]]
            if not nonedges:
                break
            k, l = random.choice(nonedges)
            A[i, j] = 0
            A[j, i] = 0
            A[k, l] = 1
            A[l, k] = 1
            C2, maxb2, ssum2, _ = _analyze(A)
            new_obj = objective(maxb2, ssum2)
            delta = new_obj - cur_obj
            if delta <= 0 or random.random() < np.exp(-max(0.0, delta) / max(T, 1e-9)):
                cur_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = A.copy()
                    best_maxb = maxb2
                    if best_maxb <= 7:
                        break
            else:
                A[i, j] = 1
                A[j, i] = 1
                A[k, l] = 0
                A[l, k] = 0
            T *= 0.999
            if T < 0.05:
                break
        if best_maxb <= 7:
            break

    A = best_A
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                edges.append([i, j])
    assert len(edges) == 401, len(edges)
    return edges
