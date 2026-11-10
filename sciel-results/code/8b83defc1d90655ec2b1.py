"""Distance-based phylogenetic topology recovery: FastUPGMA Neighbor Joining."""
import time
import numpy as np

_DEADLINE = time.time() + 250.0  # explicit safety margin under the 270 s wall-clock limit


def _jc_matrix(sequences, labels):
    n = len(labels)
    L = len(sequences[labels[0]])
    arr = np.zeros((n, L), dtype=np.uint8)
    for i, lab in enumerate(labels):
        arr[i] = np.frombuffer(sequences[lab].encode("ascii"), dtype=np.uint8)
    diff = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        diff[i] = (arr[i] != arr).sum(axis=1)
    p = diff.astype(np.float64) / float(L)
    np.fill_diagonal(p, 0.0)
    np.clip(p, 0.0, 0.7499, out=p)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(d, 0.0)
    return d


def _nj_fastupgma(D):
    n = D.shape[0]
    if n < 3:
        return None
    D = D.copy()
    alive = list(range(n))
    clusters = {i: [i] for i in range(n)}
    r = np.zeros(n, dtype=np.float64)   # barycenter term
    cnt = np.ones(n, dtype=np.float64)  # leaf count
    while len(alive) > 3:
        m = len(alive)
        idx = np.array(alive, dtype=np.int64)
        sub = D[np.ix_(idx, idx)]
        # FastUPGMA: candidate pairs by average pairwise distance
        avg = sub.sum(axis=1) / float(m - 1)
        order = np.argsort(avg)
        cand = []
        for a in order:
            j = int(idx[np.argmin(sub[a])])
            i = int(idx[a])
            if i != j:
                cand.append((i, j))
            if len(cand) >= 12:
                break
        # validate with exact NJ criterion
        best = None
        for (i, j) in cand:
            c = (m - 2.0) * D[i, j] - r[i] - r[j]
            if best is None or c < best[0]:
                best = (c, i, j)
        if best is None:
            best = (None, alive[0], alive[1])
            for a in range(m):
                for b in range(a + 1, m):
                    i, j = int(idx[a]), int(idx[b])
                    c = (m - 2.0) * D[i, j] - r[i] - r[j]
                    if best[0] is None or c < best[0]:
                        best = (c, i, j)
        _, i, j = best
        d = D[i, j]
        li, lj = cnt[i], cnt[j]
        for k in alive:
            if k == i or k == j:
                continue
            D[i, k] = D[k, i] = 0.5 * (D[i, k] + D[j, k] - d) - (li / (li + lj)) * r[i] + (lj / (li + lj)) * r[j]
        r[i] = (li * r[i] + lj * r[j]) / (li + lj)
        cnt[i] = li + lj
        clusters[i] = clusters[i] + clusters[j]
        alive.remove(j)
        del clusters[j]
    return clusters[alive[0]] + clusters[alive[1]] + clusters[alive[2]]


def _reconstruct_tree(leafset):
    n = len(leafset)
    if n < 3:
        return leafset[:]
    if n == 3:
        return leafset[:]
    nodes = list(leafset)
    nxt = n
    while len(nodes) > 3:
        rest = nodes[:-2]
        new = nxt
        nxt += 1
        nodes = rest + [new]
    a, b, c = nodes
    return (a, b, c)


def _to_newick(tree, labels):
    def emit(node):
        if isinstance(node, int):
            return labels[node]
        return "(" + ",".join(emit(c) for c in node) + ")"
    return emit(tree) + ";"


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    if n == 3:
        return "(" + ",".join(labels) + ");"
    D = _jc_matrix(sequences, labels)
    leafset = _nj_fastupgma(D)
    if leafset is None:
        leafset = list(range(n))
    tree = _reconstruct_tree(leafset)
    return _to_newick(tree, labels)
