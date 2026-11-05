"""BIONJ-weighted Neighbor-Joining + NNI Robinson-Foulds polish for phylogenies."""
import time
import math
import numpy as np

_DEADLINE = time.time() + 250.0   # wall-clock guard (evaluator kills ~270s)
_EPS = 1e-12


def _jc_pair(si, sj, L):
    diff = 0
    for k in range(L):
        if si[k] != sj[k]:
            diff += 1
    p = diff / L
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _bionj_matrix(D):
    """BIONJ variance-weighted distance matrix (Guindon & Gascuel 2003)."""
    D = np.asarray(D, dtype=float)
    S = D.sum(axis=1)
    DS = (D @ S) / 2.0
    V = D @ D - DS
    ii = np.diag(V)
    denom = ii[:, None] + ii[None, :]
    denom = np.where(denom < _EPS, _EPS, denom)
    B = D * (1.0 + 0.5 * V / denom)
    np.fill_diagonal(B, 0.0)
    return B


def _children_of(s):
    left = s[1:s.index(',', 1)].strip()
    right = s[s.index(',', 1) + 1:-1].strip()
    return left, right


def _leafset(trees, root, n):
    leafset = {}
    for v in trees:
        if v < n:
            leafset[v] = frozenset((v,))
    stack = [root]
    while stack:
        v = stack.pop()
        if v < n:
            continue
        left, right = _children_of(trees[v])
        lv = int(left) if not left.startswith('(') else left
        rv = int(right) if not right.startswith('(') else right
        for p in (lv, rv):
            if p.startswith('('):
                stack.append(p)
        leafset[v] = leafset[lv] | leafset[rv]
    return leafset


def _splits_of(trees, root, n):
    leafset = _leafset(trees, root, n)
    out = set()
    for v in leafset:
        if v >= n:
            A = leafset[v]
            if 0 < len(A) < n:
                out.add(A)
    return out


def _nj_tree(B, n):
    B = np.array(B, dtype=float)
    m = n
    trees = {i: str(i) for i in range(n)}
    next_id = n
    while m > 3:
        r = B.sum(axis=1)
        c = (m - 2) * B - r[:, None] - r[None, :]
        bi, bj, bv = 0, 1, c[0, 1]
        for i in range(m):
            row = c[i, i + 1:]
            j = i + 1 + int(np.argmin(row))
            if row[j - i - 1] < bv:
                bi, bj, bv = i, j, row[j - i - 1]
        i, j = bi, bj
        new = next_id
        next_id += 1
        trees[new] = "(" + trees[i] + "," + trees[j] + ")"
        rest = [k for k in range(m) if k != i and k != j]
        new_row = np.empty(m - 1, dtype=float)
        for t, k in enumerate(rest):
            new_row[t] = 0.5 * (B[i, k] + B[j, k] - B[i, j])
        new_row.append(0.0)
        newB = np.empty((m - 1, m - 1), dtype=float)
        newB[:-1, :-1] = B[np.ix_(rest, rest)]
        newB[:-1, -1] = new_row[:-1]
        newB[-1, :-1] = new_row[:-1]
        B = newB
        m = m - 1
    return 0, trees


def _nni_improve(trees, root, n, max_iter=30):
    """Greedy NNI minimizing the number of distinct splits (RF proxy)."""
    def score(tr, rt):
        return len(_splits_of(tr, rt, n))

    def nni_moves(tr, rt):
        edge_list = []

        def walk(node):
            if node < n:
                return
            left, right = _children_of(tr[node])
            for c in (left, right):
                if c.startswith('('):
                    edge_list.append((node, c))
                    walk(c)
        walk(rt)
        for (par, child) in edge_list:
            pl, pr = _children_of(tr[par])
            other = pr if pl == child else pl
            c1, c2 = _children_of(tr[child])
            for a, b in ((c1, c2), (c2, c1)):
                nt = dict(tr)
                nt[child] = "(" + a + "," + b + ")"
                nt[par] = "(" + nt[child] + "," + other + ")"
                yield nt, rt

    cur = score(trees, root)
    best_tr, best_rt = trees, root
    improved = True
    it = 0
    while improved and it < max_iter:
        it += 1
        improved = False
        best_local = None
        for nt, rt in nni_moves(best_tr, best_rt):
            sc = score(nt, rt)
            if best_local is None or sc < best_local[0]:
                best_local = (sc, nt, rt)
        if best_local is not None and best_local[0] < cur:
            best_tr, best_rt = best_local[1], best_local[2]
            cur = best_local[0]
            improved = True
    return best_tr, best_rt


def _finalize(trees, root, labels):
    def map_node(tok):
        tok = tok.strip()
        if tok.startswith('('):
            left, right = _children_of(tok)
            return "(" + map_node(left) + "," + map_node(right) + ")"
        return labels[int(tok)]
    return map_node(trees[root]) + ";"


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"

    seqs = [sequences[lb] for lb in labels]
    L = len(seqs[0])
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        si = seqs[i]
        for j in range(i + 1, n):
            d = _jc_pair(si, seqs[j], L)
            D[i, j] = d
            D[j, i] = d

    B = _bionj_matrix(D)
    root, trees = _nj_tree(B, n)

    if time.time() < _DEADLINE - 20.0 and n >= 5:
        trees, root = _nni_improve(trees, root, n, max_iter=25)

    return _finalize(trees, root, labels)
