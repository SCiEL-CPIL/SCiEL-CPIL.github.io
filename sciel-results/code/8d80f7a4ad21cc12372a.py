"""QAP solver for tho150: Tabu Search + 3-opt + vectorized full 2-opt polish."""
import numpy as np
import time


def solve(a, b):
    n = len(a)
    # Evaluator kills at 150s; keep a hard safety margin.
    deadline = time.time() + 140.0

    # ---- Initialization: rank-matching ----
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    # ---- Vectorized state ----
    Bp = b[np.ix_(p, p)].copy()          # Bp[i,j] = b[p[i], p[j]]
    cost = int(np.sum(a * Bp))
    best_cost = cost
    best_p = p.copy()

    def commit(i, j, delta):
        nonlocal cost
        p[i], p[j] = p[j], p[i]
        Bp[i, :], Bp[j, :] = Bp[j, :].copy(), Bp[i, :].copy()
        cost += delta

    def delta2(i, j):
        # swap positions i, j; delta = 2*dot(Aij, D) - self terms
        D = Bp[i] - Bp[j]
        Aij = a[i] - a[j]
        val = 2.0 * np.dot(Aij, D)
        u = p[i]; v = p[j]
        val -= 2.0 * a[i, j] * (b[v, u] - b[u, v])
        val -= 2.0 * a[j, i] * (b[u, v] - b[v, u])
        return int(val)

    def delta3(i, j, k):
        # rotate positions: new[i]=old[k], new[j]=old[i], new[k]=old[j]
        u = p[i]; v = p[j]; w = p[k]
        row_i = Bp[i]; row_j = Bp[j]; row_k = Bp[k]
        D_i = row_k - row_i
        D_j = row_i - row_j
        D_k = row_j - row_k
        val = (np.dot(a[i], D_i) + np.dot(a[j], D_j) + np.dot(a[k], D_k)
               + np.dot(a[i] - a[j], row_k) + np.dot(a[j] - a[k], row_i)
               + np.dot(a[k] - a[i], row_j))
        val -= a[i, j] * (b[u, w] - b[u, v])
        val -= a[j, k] * (b[v, u] - b[v, w])
        val -= a[k, i] * (b[w, v] - b[w, u])
        val -= a[j, i] * (b[u, v] - b[w, u])
        val -= a[k, j] * (b[v, w] - b[u, v])
        val -= a[i, k] * (b[w, u] - b[v, w])
        return int(val)

    def commit3(i, j, k, delta):
        nonlocal cost
        p[i], p[j], p[k] = p[k], p[i], p[j]
        Bp[i, :], Bp[j, :], Bp[k, :] = Bp[k, :].copy(), Bp[i, :].copy(), Bp[j, :].copy()
        cost += delta

    def full_2opt():
        nonlocal best_cost, best_p
        while time.time() < deadline - 1.0:
            improved = False
            for i in range(n):
                if time.time() > deadline - 1.0:
                    break
                for j in range(i + 1, n):
                    d = delta2(i, j)
                    if d < 0:
                        commit(i, j, d)
                        improved = True
                        if cost < best_cost:
                            best_cost = cost
                            best_p = p.copy()
            if not improved:
                break

    # ---- Tabu Search ----
    tabu_len = 40
    tabu = {}          # (i,j) with i<j -> iteration when untabu
    it = 0
    tabu_reheat = 1500
    last_reheat = 0
    plateau = 0

    while time.time() < deadline - 15.0:
        it += 1

        # Reheat tabu on long plateau (restart recency memory)
        if plateau >= tabu_reheat:
            tabu.clear()
            last_reheat = it
            plateau = 0

        best_d = 0
        best_i = -1
        best_j = -1
        best_k = -1
        best_is_asp = False

        # 2-opt neighborhood (all pairs)
        for i in range(n):
            for j in range(i + 1, n):
                d = delta2(i, j)
                if d < best_d or (d < 0 and cost + d < best_cost):
                    best_d = d
                    best_i = i
                    best_j = j
                    best_k = -1
                    best_is_asp = (d < 0 and cost + d < best_cost)

        # 3-opt neighborhood (sampled triples)
        ntrip = 200
        for _ in range(ntrip):
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            k = np.random.randint(0, n)
            if i == j or j == k or i == k:
                continue
            d = delta3(i, j, k)
            if d < best_d or (d < 0 and cost + d < best_cost):
                best_d = d
                best_i = i
                best_j = j
                best_k = k
                best_is_asp = (d < 0 and cost + d < best_cost)

        if best_i < 0:
            break

        # Tabu check
        if best_k < 0:
            key = (best_i, best_j)
        else:
            key = (best_i, best_j, best_k)
        is_tabu = it - tabu.get(key, -10**9) < tabu_len
        if is_tabu and not best_is_asp:
            # no improving move available -> stop
            break

        # Commit best move
        if best_k < 0:
            commit(best_i, best_j, best_d)
        else:
            commit3(best_i, best_j, best_k, best_d)
        tabu[key] = it

        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            plateau = 0
        else:
            plateau += 1

    # ---- Final full 2-opt polish to guarantee a 2-opt local optimum ----
    full_2opt()

    return best_p.tolist()
