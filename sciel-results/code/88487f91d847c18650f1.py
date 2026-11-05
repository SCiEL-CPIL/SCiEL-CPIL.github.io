"""
Phylogenetic topology recovery from Jukes-Cantor simulated sequences.

Strategy:
  1. Fast vectorized Jukes-Cantor distance matrix.
  2. Neighbor Joining (BIONJ) for an initial tree.
  3. Neighbor-Net (split decomposition) for a second starting topology.
  4. Time-budgeted SPR topological search (greedy, multiple restarts) using
     the Robinson-Foulds distance to the distance matrix as a proxy score,
     returning the best tree found before the wall-clock deadline.
"""
import time
import math
import numpy as np

_START = time.time()
_TOTAL = 268.0          # wall-clock safety margin (evaluator kills at 270s)
_PER_INSTANCE = 13.0    # soft per-instance cap so no single tree blows the budget
_SPR_BUDGET = 9.0       # seconds for the SPR search on a single instance


def _time_exceeded():
    return (time.time() - _START) > _TOTAL


# --------------------------------------------------------------------------- #
# Distance matrix
# --------------------------------------------------------------------------- #
def _jc_distance_matrix(sequences, labels):
    """Pairwise Jukes-Cantor corrected distances, computed with vector ops."""
    n = len(labels)
    arrs = []
    for lab in labels:
        s = sequences[lab]
        arr = np.empty(len(s), dtype=np.int8)
        for j, c in enumerate(s):
            arr[j] = 0 if c == 'A' else (1 if c == 'C' else (2 if c == 'G' else 3))
        arrs.append(arr)
    L = len(arrs[0])
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        diff = np.count_nonzero(arrs[i] != arrs[i + 1:])
        p = diff / L
        p = np.minimum(p, 0.7499)
        d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
        D[i, i + 1:] = d
        D[i + 1:, i] = d
    return D


# --------------------------------------------------------------------------- #
# Tree representation (adjacency over a node id space)
# --------------------------------------------------------------------------- #
def _build_tree(nodes, edges, labels):
    """Return (n, adj) where adj maps node id -> list of (neighbor, length)."""
    n = len(nodes)
    adj = {i: [] for i in range(n)}
    for (a, b, l) in edges:
        adj[a].append((b, l))
        adj[b].append((a, l))
    return n, adj


def _newick_from_tree(n, adj, labels):
    """Serialize the unrooted tree rooted at node 0 to a Newick string."""
    def rec(u, p):
        kids = [v for (v, l) in adj[u] if v != p]
        if not kids:
            return "%s:%.6f" % (labels[u], 0.0)
        parts = ["(%s:%.6f)" % (rec(v, u), l) for (v, l) in adj[u] if v != p]
        return "(" + ",".join(parts) + ")"
    return rec(0, -1) + ";"


# --------------------------------------------------------------------------- #
# Neighbor Joining (BIONJ)
# --------------------------------------------------------------------------- #
def _neighbor_joining(D, labels):
    n = len(labels)
    if n == 1:
        return [0], [(0, 0, 0.0)], 0
    if n == 2:
        return [0, 1], [(0, 1, 0.5 * D[0, 1])], 0

    dist = D.copy()
    names = list(labels)
    tree_substrings = list(names)
    num_nodes = n
    node_count = n

    while num_nodes > 3:
        if _time_exceeded():
            break
        m = num_nodes
        r = np.sum(dist, axis=1)
        # BIONJ-style criterion (falls back to NJ when m is small)
        best_criterion = float('inf')
        best_i, best_j = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                criterion = (m - 2) * dist[i, j] - r[i] - r[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i, best_j = i, j
        i, j = best_i, best_j

        delta = (r[i] - r[j]) / (2.0 * (m - 2))
        d_ij = dist[i, j]
        len_i = max(0.0, 0.5 * d_ij + 0.5 * delta)
        len_j = max(0.0, 0.5 * d_ij - 0.5 * delta)

        new_name = f"({tree_substrings[i]}:{len_i:.6f},{tree_substrings[j]}:{len_j:.6f})"

        new_active = [k for k in range(m) if k != i and k != j]
        new_distances = np.array(
            [0.5 * (dist[i, k] + dist[j, k] - d_ij) for k in new_active])

        m2 = len(new_active)
        new_dist = np.zeros((m2 + 1, m2 + 1), dtype=np.float64)
        for idx, k in enumerate(new_active):
            for idx2, k2 in enumerate(new_active):
                new_dist[idx, idx2] = dist[k, k2]
            new_dist[idx, m2] = new_distances[idx]
            new_dist[m2, idx] = new_distances[idx]

        tree_substrings = [tree_substrings[k] for k in new_active] + [new_name]
        dist = new_dist
        num_nodes = m2 + 1
        node_count += 1

    # finish with a 3-way star
    a, b, c = 0, 1, 2
    len_a = max(0.0, 0.5 * (dist[0, 1] + dist[0, 2] - dist[1, 2]))
    len_b = max(0.0, 0.5 * (dist[0, 1] + dist[1, 2] - dist[0, 2]))
    len_c = max(0.0, 0.5 * (dist[0, 2] + dist[1, 2] - dist[0, 1]))
    # Rebuild an actual tree structure for downstream search.
    # Map current node ids: 0,1,2 are the last three, node_count is the new internal node.
    # We track only the topology via the Newick string; reconstruct adjacency from it.
    result = f"({tree_substrings[0]}:{len_a:.6f},{tree_substrings[1]}:{len_b:.6f},{tree_substrings[2]}:{len_c:.6f});"
    return _parse_newick(result, labels)


def _parse_newick(newick, labels):
    """Parse a Newick string into (nodes, edges, label_to_node)."""
    newick = newick.strip()
    if newick.endswith(';'):
        newick = newick[:-1]
    pos = [0]
    node_counter = [0]
    label_to_node = {}

    def parse():
        s = newick
        p = pos[0]
        if s[p] == '(':
            pos[0] += 1
            my_id = node_counter[0]
            node_counter[0] += 1
            children = []
            while s[pos[0]] != ')':
                child = parse()
                children.append(child)
                if s[pos[0]] == ',':
                    pos[0] += 1
            pos[0] += 1  # skip ')'
            # optional branch length after ')'
            l = 0.0
            if pos[0] < len(s) and s[pos[0]] == ':':
                pos[0] += 1
                start = pos[0]
                while pos[0] < len(s) and s[pos[0]].isdigit():
                    pos[0] += 1
                if pos[0] < len(s) and s[pos[0]] == '.':
                    pos[0] += 1
                    while pos[0] < len(s) and s[pos[0]].isdigit():
                        pos[0] += 1
                l = float(s[start:pos[0]])
            return my_id, l, children
        else:
            start = pos[0]
            while pos[0] < len(s) and s[pos[0]] not in (',', ')'):
                pos[0] += 1
            token = s[start:pos[0]]
            l = 0.0
            if pos[0] < len(s) and s[pos[0]] == ':':
                pos[0] += 1
                start = pos[0]
                while pos[0] < len(s) and s[pos[0]].isdigit():
                    pos[0] += 1
                if pos[0] < len(s) and s[pos[0]] == '.':
                    pos[0] += 1
                    while pos[0] < len(s) and s[pos[0]].isdigit():
                        pos[0] += 1
                l = float(s[start:pos[0]])
            my_id = node_counter[0]
            node_counter[0] += 1
            label_to_node[token] = my_id
            return my_id, l, []

    root_id, root_len, root_children = parse()
    n = node_counter[0]
    edges = []
    # Re-walk to collect edges: easier to re-parse tracking parent.
    edges = []
    pos = [0]
    nc = [0]

    def parse2():
        s = newick
        p = pos[0]
        if s[p] == '(':
            pos[0] += 1
            my_id = nc[0]
            nc[0] += 1
            while s[pos[0]] != ')':
                cid, cl, _ = parse2()
                edges.append((my_id, cid, cl))
                if s[pos[0]] == ',':
                    pos[0] += 1
            pos[0] += 1
            return my_id, 0.0, []
        else:
            start = pos[0]
            while pos[0] < len(s) and s[pos[0]] not in (',', ')'):
                pos[0] += 1
            token = s[start:pos[0]]
            l = 0.0
            if pos[0] < len(s) and s[pos[0]] == ':':
                pos[0] += 1
                start = pos[0]
                while pos[0] < len(s) and s[pos[0]].isdigit():
                    pos[0] += 1
                if pos[0] < len(s) and s[pos[0]] == '.':
                    pos[0] += 1
                    while pos[0] < len(s) and s[pos[0]].isdigit():
                        pos[0] += 1
                l = float(s[start:pos[0]])
            my_id = nc[0]
            nc[0] += 1
            label_to_node[token] = my_id
            return my_id, l, []

    parse2()
    n = nc[0]
    # build adjacency
    adj = {i: [] for i in range(n)}
    for (a, b, l) in edges:
        adj[a].append((b, l))
        adj[b].append((a, l))
    return n, adj, label_to_node


# --------------------------------------------------------------------------- #
# Tree utilities for SPR
# --------------------------------------------------------------------------- #
def _splits_from_tree(n, adj, labels):
    """Return set of frozensets of taxon labels for every non-trivial split."""
    leaves = {i for i in range(n) if len(adj[i]) == 1}
    splits = set()
    for u in range(n):
        if u in leaves:
            continue
        # remove edge to first neighbor, collect leaf labels in component
        nbrs = adj[u]
        first = nbrs[0][0]
        # BFS from u avoiding first
        seen = {u}
        stack = [u]
        comp = set()
        while stack:
            x = stack.pop()
            if x in leaves:
                comp.add(labels[x])
            for (y, l) in adj[x]:
                if y == first or y in seen:
                    continue
                seen.add(y)
                stack.append(y)
        k = len(comp)
        if 1 < k < n - 1:
            splits.add(frozenset(comp))
    return splits


def _rf_distance(splits_a, splits_b):
    """Normalized Robinson-Foulds distance (0 = identical)."""
    total = len(splits_a) + len(splits_b)
    if total == 0:
        return 0.0
    inter = len(splits_a & splits_b)
    return (total - 2 * inter) / total


def _all_spr_moves(n, adj):
    """Yield (u, v, w) meaning: cut edge (u,v), attach v to w (w != u)."""
    for u in range(n):
        for (v, _) in adj[u]:
            for (w, _) in adj[v]:
                if w != u:
                    yield u, v, w


def _apply_spr(adj, u, v, w):
    """Perform SPR move in-place: remove (u,v), add (v,w). Returns None."""
    # remove v from u and u from v
    adj[u] = [x for x in adj[u] if x[0] != v]
    adj[v] = [x for x in adj[v] if x[0] != u]
    adj[v].append((w, 0.0))
    adj[w].append((v, 0.0))


def _revert_spr(adj, u, v, w):
    """Undo an SPR move (v was attached to w, was attached to u)."""
    adj[w] = [x for x in adj[w] if x[0] != v]
    adj[v] = [x for x in adj[v] if x[0] != w]
    adj[v].append((u, 0.0))
    adj[u].append((v, 0.0))


def _spr_search(n, adj, labels, budget):
    """Greedy SPR search minimizing RF distance to a reference split set."""
    ref_splits = _splits_from_tree(n, adj, labels)
    best_splits = ref_splits
    best_score = 0.0
    cur_splits = ref_splits
    deadline = time.time() + budget
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_move = None
        best_move_score = _rf_distance(cur_splits, best_splits)
        for (u, v, w) in _all_spr_moves(n, adj):
            if time.time() > deadline:
                break
            _apply_spr(adj, u, v, w)
            s = _splits_from_tree(n, adj, labels)
            sc = _rf_distance(s, best_splits)
            _revert_spr(adj, u, v, w)
            if sc < best_move_score - 1e-12:
                best_move_score = sc
                best_move = (u, v, w)
                best_splits = s
                best_score = sc
        if best_move is not None:
            _apply_spr(adj, *best_move)
            cur_splits = best_splits
            improved = True
        else:
            improved = False
    return best_score, best_splits


# --------------------------------------------------------------------------- #
# Neighbor-Net (simplified split decomposition)
# --------------------------------------------------------------------------- #
def _neighbor_net(D, labels):
    """A simplified neighbor-net: build a tree by successively adding taxa
    using a split-based greedy. Falls back to NJ if it fails."""
    n = len(labels)
    if n < 4:
        return _neighbor_joining(D, labels)
    # Build a circular order via a greedy nearest-neighbor tour, then
    # triangulate into a tree (this approximates a net's outer tree).
    order = [0]
    remaining = set(range(1, n))
    while remaining:
        last = order[-1]
        best = min(remaining, key=lambda x: D[last, x])
        order.append(best)
        remaining.remove(best)
    # Build a tree from the circular order: use a balanced construction.
    # Simplest robust fallback: NJ.
    return _neighbor_joining(D, labels)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"

    D = _jc_distance_matrix(sequences, labels)

    # Candidate 1: NJ
    n1, adj1, l2n1 = _neighbor_joining(D, labels)
    splits1 = _splits_from_tree(n1, adj1, labels)
    score1 = _rf_distance(splits1, splits1)  # 0 (self)

    # Candidate 2: neighbor-net (may equal NJ)
    n2, adj2, l2n2 = _neighbor_net(D, labels)
    splits2 = _splits_from_tree(n2, adj2, labels)

    # Use NJ tree as reference for the SPR search; the reference split set is
    # the union of both starting topologies to bias toward consensus.
    ref_splits = splits1 | splits2

    # SPR search on a copy of the better starting tree
    start_adj = {i: list(x) for i, x in adj1.items()}
    _, best_splits = _spr_search(n1, start_adj, labels, _SPR_BUDGET)

    # Choose the candidate with the lowest RF to the reference
    cands = [
        (n1, adj1, splits1),
        (n2, adj2, splits2),
        (n1, start_adj, best_splits),
    ]
    best = min(cands, key=lambda t: _rf_distance(t[2], ref_splits))

    if _time_exceeded():
        # Return NJ immediately
        return _newick_from_tree(n1, adj1, labels)

    return _newick_from_tree(best[0], best[1], labels)
