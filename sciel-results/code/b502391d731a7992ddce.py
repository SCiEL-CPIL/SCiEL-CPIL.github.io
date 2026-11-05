import time
import math
import random
import numpy as np

_N = 16
_M = 65
_DEADLINE = 138.0  # wall-clock safety margin (evaluator kills at 150s)

def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)

    A = np.zeros((_N, _N), dtype=np.int8)

    def baseline():
        # K_{8,8} plus one intra-part edge
        A[0:8, 8:16] = 1
        A[8:16, 0:8] = 1
        A[0, 1] = 1
        A[1, 0] = 1
        return A.copy()

    def edge_count(A):
        return int(np.count_nonzero(A) // 2)

    def score(A):
        # max over triangles of deg(u)+deg(v)+deg(w)
        deg = A.sum(axis=1)
        # triangles: A^3 diagonal
        A3 = A @ A @ A
        tri = np.diag(A3)
        idx = np.where(tri > 0)[0]
        if len(idx) == 0:
            return 0
        # for each triangle (i,j,k) we need deg sums; enumerate triangles
        # faster: for each edge (i,j) common neighbors = (A@A)[i,j]
        # we need max deg(i)+deg(j)+deg(k) over triangles
        best = 0
        # iterate over edges
        edges = np.argwhere(A > 0)
        for e in edges:
            i, j = int(e[0]), int(e[1])
            if i >= j:
                continue
            # common neighbors
            cn = (A[i] & A[j]).nonzero()[0]
            for k in cn:
                s = int(deg[i]) + int(deg[j]) + int(deg[k])
                if s > best:
                    best = s
        return best

    def soft_score(A, beta):
        # smooth surrogate: soft-max of triangle degree sums
        deg = A.sum(axis=1)
        edges = np.argwhere(A > 0)
        vals = []
        for e in edges:
            i, j = int(e[0]), int(e[1])
            if i >= j:
                continue
            cn = (A[i] & A[j]).nonzero()[0]
            for k in cn:
                vals.append(int(deg[i]) + int(deg[j]) + int(deg[k]))
        if not vals:
            return 0.0
        vals = np.array(vals, dtype=np.float64)
        m = vals.max()
        return float(m + math.log(np.exp(vals - m).sum())) / beta * beta  # scale
        # Actually return the soft-max value (smooth)
        # recompute properly below

    def softmax_score(A, beta):
        deg = A.sum(axis=1)
        edges = np.argwhere(A > 0)
        vals = np.empty(0, dtype=np.float64)
        for e in edges:
            i, j = int(e[0]), int(e[1])
            if i >= j:
                continue
            cn = (A[i] & A[j]).nonzero()[0]
            for k in cn:
                vals = np.concatenate([vals, np.array([int(deg[i]) + int(deg[j]) + int(deg[k])], dtype=np.float64)])
        if len(vals) == 0:
            return 0.0
        m = vals.max()
        return float(m + math.log(np.exp(vals - m).sum()))

    # Precompute edge list helpers
    def get_edges(A):
        es = np.argwhere(A > 0)
        out = []
        for e in es:
            i, j = int(e[0]), int(e[1])
            if i < j:
                out.append((i, j))
        return out

    def get_nonedges(A):
        es = np.argwhere(A == 0)
        out = []
        for e in es:
            i, j = int(e[0]), int(e[1])
            if i < j:
                out.append((i, j))
        return out

    A = baseline()
    cur_score = score(A)
    best_A = A.copy()
    best_score = cur_score

    def swap(A, e_remove, e_add):
        i, j = e_remove
        A[i, j] = 0
        A[j, i] = 0
        k, l = e_add
        A[k, l] = 1
        A[l, k] = 1
        return A

    # Simulated annealing
    T = 1.5
    T_end = 0.02
    cooling = 0.9995
    iters = 0
    while T > T_end:
        if time.time() > _DEADLINE - 15:
            break
        edges = get_edges(A)
        nonedges = get_nonedges(A)
        if len(edges) == 0 or len(nonedges) == 0:
            break
        e_rm = edges[rng.integers(len(edges))]
        e_add = nonedges[rng.integers(len(nonedges))]
        # skip if same
        if e_rm == e_add:
            T *= cooling
            continue
        A2 = A.copy()
        swap(A2, e_rm, e_add)
        if edge_count(A2) != _M:
            T *= cooling
            continue
        s2 = score(A2)
        if s2 < cur_score:
            A = A2
            cur_score = s2
            if s2 < best_score:
                best_score = s2
                best_A = A2.copy()
        else:
            d = s2 - cur_score
            if d == 0:
                accept = True
            else:
                accept = rng.random() < math.exp(-d / T)
            if accept:
                A = A2
                cur_score = s2
        T *= cooling
        iters += 1
        if iters % 500 == 0:
            pass

    # Hill-climbing polish
    improved = True
    while improved and time.time() < _DEADLINE - 5:
        improved = False
        edges = get_edges(best_A)
        nonedges = get_nonedges(best_A)
        for e_rm in edges:
            if time.time() > _DEADLINE - 5:
                break
            for e_add in nonedges:
                if e_rm == e_add:
                    continue
                A2 = best_A.copy()
                swap(A2, e_rm, e_add)
                if edge_count(A2) != _M:
                    continue
                s2 = score(A2)
                if s2 < best_score:
                    best_score = s2
                    best_A = A2
                    improved = True
                    break
            if improved:
                break

    # Build edge list
    es = np.argwhere(best_A > 0)
    out_edges = []
    for e in es:
        i, j = int(e[0]), int(e[1])
        if i < j:
            out_edges.append([i, j])
    return {'n': _N, 'edges': out_edges}
