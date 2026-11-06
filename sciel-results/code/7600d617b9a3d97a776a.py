import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110

    q = 5
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, q - 2, q)
                    rep = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q - 2, q)
                    rep = (0, 1, (z * inv_y) % q)
                else:
                    rep = (0, 0, 1)
                if rep not in seen:
                    seen.add(rep)
                    points.append(rep)

    n = len(points)

    A = np.zeros((n, n), dtype=np.int8)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            val = (points[i][0] * points[j][0] +
                   points[i][1] * points[j][1] +
                   points[i][2] * points[j][2]) % q
            if val == 0:
                A[i, j] = 1
                A[j, i] = 1
                edges.append([i, j])

    if len(edges) < 91:
        best_edge = None
        best_c4s = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] == 0:
                    common = int(np.sum(A[i] & A[j]))
                    if common < best_c4s:
                        best_c4s = common
                        best_edge = [i, j]
        if best_edge:
            A[best_edge[0], best_edge[1]] = 1
            A[best_edge[1], best_edge[0]] = 1
            edges.append(best_edge)

    def count_c4(A):
        AA = A.astype(np.int32) @ A.astype(np.int32)
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = AA[i, j]
                if c >= 2:
                    total += c * (c - 1) // 2
        return total // 2

    current_c4 = count_c4(A)

    def c4s_for_edge(A, u, v):
        count = 0
        for w in np.where(A[u] == 1)[0]:
            if w == v:
                continue
            mask = A[v].copy()
            mask[u] = 0
            count += int(np.sum(A[w] & mask))
        return count

    improved = True
    max_iter = 50
    it = 0
    while improved and it < max_iter and time.time() < deadline - 25:
        improved = False
        it += 1
        non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] == 0]

        for ei in range(len(edges)):
            if time.time() > deadline - 25:
                break
            u, v = edges[ei]
            if A[u, v] == 0:
                continue
            rem = c4s_for_edge(A, u, v)

            best_total = float('inf')
            best_new = None
            for (i, j) in non_edges:
                add = c4s_for_edge(A, i, j)
                total = -rem + add
                if total < best_total:
                    best_total = total
                    best_new = [i, j]

            if best_new is not None and best_total < 0:
                A[u, v] = 0
                A[v, u] = 0
                edges[ei] = best_new
                A[best_new[0], best_new[1]] = 1
                A[best_new[1], best_new[0]] = 1
                current_c4 += best_total
                improved = True
                break

    assert len(edges) == 91
    edge_set = set()
    for e in edges:
        es = tuple(sorted(e))
        assert es not in edge_set
        edge_set.add(es)
        assert 0 <= e[0] <= 30 and 0 <= e[1] <= 30 and e[0] != e[1]

    return edges
