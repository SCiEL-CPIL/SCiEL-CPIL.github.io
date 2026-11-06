"""DPP topology search + JC likelihood SPR refinement for JC-simulated trees."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_BASE = time.time()
_DEADLINE = 258.0  # safety margin under the 270s wall-clock kill


def _remaining():
    return _DEADLINE - (time.time() - _BASE)


# ---------------------------------------------------------------------------
# Distance matrix (Jukes-Cantor corrected)
# ---------------------------------------------------------------------------
def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    chars = np.array([list(sequences[l]) for l in labels])  # n x L
    L = chars.shape[1]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.count_nonzero(chars[i] != chars[j])
            p = d / L
            if p >= 0.7499:
                val = 3.0
            else:
                val = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist[i, j] = dist[j, i] = val
    return dist


# ---------------------------------------------------------------------------
# Fast Neighbour Joining (vectorized)
# ---------------------------------------------------------------------------
def _nj(dist, labels):
    n = len(labels)
    d = dist.astype(float).copy()
    trees = {labels[i]: labels[i] for i in range(n)}
    nodes = list(range(n))
    while len(nodes) > 3:
        m = len(nodes)
        row = d[np.ix_(nodes, nodes)]
        r = row.sum(axis=1)
        q = (m - 2) * row - r[:, None] - r[None, :]
        iu = np.triu_indices(m, 1)
        qi, qj = iu
        vals = q[qi, qj]
        k = int(np.argmin(vals))
        a, b = nodes[qi[k]], nodes[qj[k]]
        joined = "(%s,%s)" % (trees[a], trees[b])
        new = m
        rest = [x for x in nodes if x not in (a, b)]
        newnodes = rest + [new]
        for x in rest:
            d[new, x] = d[x, new] = 0.5 * (d[a, x] + d[b, x] - d[a, b])
        nodes = newnodes
        trees[new] = joined
    return "(" + ",".join(trees[x] for x in nodes) + ");"


# ---------------------------------------------------------------------------
# Distance-Proportional Parsimony (DPP)
# ---------------------------------------------------------------------------
def _dpp(dist, labels):
    n = len(labels)
    d = dist.astype(float).copy()
    trees = {labels[i]: labels[i] for i in range(n)}
    nodes = list(range(n))
    while len(nodes) > 3:
        m = len(nodes)
        row = d[np.ix_(nodes, nodes)]
        r = row.sum(axis=1) / (m - 1)
        dp = row - (r[:, None] + r[None, :]) / 2.0
        iu = np.triu_indices(m, 1)
        qi, qj = iu
        vals = dp[qi, qj]
        k = int(np.argmin(vals))
        a, b = nodes[qi[k]], nodes[qj[k]]
        joined = "(%s,%s)" % (trees[a], trees[b])
        new = m
        rest = [x for x in nodes if x not in (a, b)]
        newnodes = rest + [new]
        for x in rest:
            d[new, x] = d[x, new] = 0.5 * (d[a, x] + d[b, x])
        nodes = newnodes
        trees[new] = joined
    return "(" + ",".join(trees[x] for x in nodes) + ");"


# ---------------------------------------------------------------------------
# JC model likelihood for SPR (fixed per-branch length scoring)
# ---------------------------------------------------------------------------
def _jc_exp(P):
    e1 = math.exp(-4.0 * P)
    e0 = (1.0 - e1) / 3.0
    e11 = (1.0 + 2.0 * e1) / 3.0
    return np.array([[e11, e0, e0, e0],
                     [e0, e11, e0, e0],
                     [e0, e0, e11, e0],
                     [e0, e0, e0, e11]])


def _build_tree(newick, labels):
    tokens = []
    i = 0
    s = newick
    while i < len(s):
        c = s[i]
        if c in "(),":
            tokens.append(c)
            i += 1
        elif c == ';':
            break
        else:
            j = i
            while j < len(s) and s[j] not in "(),;":
                j += 1
            tok = s[i:j]
            if tok:
                tokens.append(tok)
            i = j
    pos = [0]

    def parse_node():
        c = tokens[pos[0]]
        if c == '(':
            pos[0] += 1
            children = []
            while True:
                child = parse_node()
                children.append(child)
                c = tokens[pos[0]]
                pos[0] += 1
                if c == ',':
                    continue
                elif c == ')':
                    break
            if pos[0] < len(tokens) and tokens[pos[0]] not in "(),;":
                pos[0] += 1
            return ('node', children)
        else:
            label = c
            pos[0] += 1
            return ('leaf', None, label)

    root = parse_node()
    edges = set()

    def add_edge(a, b):
        edges.add((a, b))
        edges.add((b, a))

    def walk(node, parent):
        if node[0] == 'leaf':
            if parent is not None:
                add_edge(parent, node[2])
            return
        nid = ('I', node)
        if parent is not None:
            add_edge(parent, nid)
        for c in node[1]:
            walk(c, nid)

    walk(root, None)
    return edges


def _jc_likelihood(edges, labels, counts, L):
    n = len(labels)
    adj = {}
    for (a, b) in edges:
        adj.setdefault(a, set()).add(b)
    root = labels[0]
    parent = {root: None}
    order = [root]
    for u in order:
        for v in adj.get(u, ()):
            if v != parent.get(u):
                parent[v] = u
                order.append(v)
    children = {u: [] for u in order}
    for u in order:
        if parent[u] is not None:
            children[parent[u]].append(u)
    NEG = -1e300
    leafset = set(labels)
    prof = {}
    for u in order:
        if u in leafset:
            v = np.full(4, NEG)
            v[counts[u]] = 0.0
            prof[u] = v
    BL = 0.2
    M = _jc_exp(BL)
    for u in reversed(order):
        if u in leafset:
            continue
        tot = np.full(4, NEG)
        first = True
        for c in children[u]:
            vc = prof[c]
            prod = M * np.exp(vc)
            if first:
                tot = np.log(np.sum(prod, axis=0))
                first = False
            else:
                tot = tot + np.log(np.sum(prod, axis=0))
        prof[u] = tot
    rootvec = prof[root]
    return float(np.log(np.sum(np.exp(rootvec))))


def _side(a, b, edges):
    adj = {}
    for (u, v) in edges:
        adj.setdefault(u, set()).add(v)
    seen = {a}
    stack = [a]
    while stack:
        u = stack.pop()
        for v in adj.get(u, ()):
            if u == a and v == b:
                continue
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def _edges_to_newick(edges, labels):
    adj = {}
    for (u, v) in edges:
        adj.setdefault(u, set()).add(v)
    root = labels[0]

    def walk(u, parent):
        kids = [v for v in adj.get(u, ()) if v != parent]
        if not kids:
            return u
        parts = [walk(k, u) for k in kids]
        return "(" + ",".join(parts) + ")"
    return "(" + walk(root, None) + ");"


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    L = len(next(iter(sequences.values())))
    dist = _jc_distance_matrix(sequences, labels)

    cands = []
    nj_tree = _nj(dist, labels)
    cands.append(nj_tree)
    if n >= 4:
        dpp_tree = _dpp(dist, labels)
        if dpp_tree != nj_tree:
            cands.append(dpp_tree)

    base = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    counts = {}
    for l in labels:
        counts[l] = np.array([base[c] for c in sequences[l]])

    def score_tree(t):
        try:
            edges = _build_tree(t, labels)
            return _jc_likelihood(edges, labels, counts, L)
        except Exception:
            return -1e300

    best_tree = nj_tree
    best_ll = score_tree(nj_tree)
    for t in cands[1:]:
        ll = score_tree(t)
        if ll > best_ll:
            best_ll = ll
            best_tree = t

    current = best_tree
    current_ll = best_ll
    improved = True
    max_rounds = 15
    round_i = 0
    while improved and _remaining() > 25.0 and round_i < max_rounds:
        round_i += 1
        improved = False
        try:
            edges = _build_tree(current, labels)
        except Exception:
            break
        moved = False
        for (a, b) in list(edges):
            if _remaining() < 25.0:
                break
            side_a = _side(a, b, edges)
            if len(side_a) <= 1 or len(side_a) >= n:
                continue
            for (x, y) in list(edges):
                if (x == a and y == b) or (x == b and y == a):
                    continue
                if x in side_a and y in side_a:
                    continue
                if x not in side_a and y not in side_a:
                    continue
                new_edges = set()
                for (u, v) in edges:
                    if (u == a and v == b) or (u == b and v == a):
                        continue
                    new_edges.add((u, v))
                    new_edges.add((v, u))
                new_edges.discard((x, y))
                new_edges.discard((y, x))
                new_edges.add((x, a))
                new_edges.add((a, x))
                new_edges.add((a, y))
                new_edges.add((y, a))
                nt = _edges_to_newick(new_edges, labels)
                ll = score_tree(nt)
                if ll > current_ll + 1e-9:
                    current = nt
                    current_ll = ll
                    improved = True
                    moved = True
                    break
            if moved:
                break
    return current
# EVOLVE-BLOCK-END
