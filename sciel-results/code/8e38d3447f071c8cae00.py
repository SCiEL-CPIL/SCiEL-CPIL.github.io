import time
import math
import numpy as np

_module_start = time.time()
_DEADLINE = 265.0

def _time_left():
    return _DEADLINE - (time.time() - _module_start)

def _jc_dist(a, b):
    L = len(a)
    if L == 0:
        return 0.0
    arr_a = np.frombuffer(a.encode(), dtype=np.uint8)
    arr_b = np.frombuffer(b.encode(), dtype=np.uint8)
    mism = int(np.count_nonzero(arr_a != arr_b))
    p = mism / L
    if p >= 0.7499:
        return 3.0
    if p <= 0.0:
        return 0.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _dist_matrix(sequences, labels):
    n = len(labels)
    d = np.zeros((n, n))
    for i in range(n):
        si = sequences[labels[i]]
        for j in range(i + 1, n):
            sj = sequences[labels[j]]
            dist = _jc_dist(si, sj)
            d[i, j] = dist
            d[j, i] = dist
    return d

def _nj(d, n):
    if n == 2:
        return 2, {2: (0, 1)}
    active = list(range(n))
    dist = d.astype(float).copy()
    next_id = n
    tree = {}
    while len(active) > 2:
        m = len(active)
        r = np.array([sum(dist[a, b] for b in active if b != a) for a in active])
        best = (float('inf'), -1, -1)
        for ia in range(m):
            for ib in range(ia + 1, m):
                a, b = active[ia], active[ib]
                v = (m - 2) * dist[a, b] - r[ia] - r[ib]
                if v < best[0]:
                    best = (v, a, b)
        _, a, b = best
        nid = next_id
        next_id += 1
        for x in active:
            if x != a and x != b:
                dist[nid, x] = 0.5 * (dist[a, x] + dist[b, x] - dist[a, b])
                dist[x, nid] = dist[nid, x]
        tree[nid] = (a, b)
        active.remove(a)
        active.remove(b)
        active.append(nid)
    la, lb = active
    root = next_id
    next_id += 1
    tree[root] = (la, lb)
    return root, tree

def _newick(node, tree, labels, n):
    if node < n:
        return labels[node]
    l, r = tree[node]
    return f"({_newick(l, tree, labels, n)},{_newick(r, tree, labels, n)})"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}))"
    if _time_left() < 0.5:
        return "(" + ",".join(labels) + ")"
    d = _dist_matrix(sequences, labels)
    root, tree = _nj(d, n)
    return _newick(root, tree, labels, n) + ";"
