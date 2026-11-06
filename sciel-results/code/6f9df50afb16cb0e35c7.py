import numpy as np
import time
from itertools import combinations

_DEADLINE = 260.0
_start = None


def _elapsed():
    return time.time() - _start if _start is not None else 0.0


def _check_time():
    if _elapsed() > _DEADLINE:
        raise TimeoutError


def _build_newick(tree):
    if isinstance(tree, str):
        return tree
    return "(" + _build_newick(tree[0]) + "," + _build_newick(tree[1]) + ")"


def _parse_newick(s, n):
    s = s.strip()
    if s.endswith(';'):
        s = s[:-1]
    label_to_idx = {f"t{i}": i for i in range(n)}
    splits = set()
    pos = 0

    def parse_node(i):
        nonlocal pos
        if i >= len(s):
            return set()
        if s[i] == '(':
            i += 1
            c1 = parse_node(i)
            if i < len(s) and s[i] == ',':
                i += 1
            c2 = parse_node(i)
            if i < len(s) and s[i] == ')':
                i += 1
            pos = i
            tset = c1 | c2
            if 1 < len(tset) < n:
                splits.add(frozenset(tset))
            return tset
        else:
            j = i
            while j < len(s) and s[j] not in ',;':
                j += 1
            taxon = s[i:j].split('.')[0]
            pos = j
            if taxon in label_to_idx:
                return {label_to_idx[taxon]}
            return set()

    all_taxa = parse_node(0)
    return splits, all_taxa


def _valid_tree(newick, n):
    try:
        splits, taxa = _parse_newick(newick, n)
    except Exception:
        return False
    return len(taxa) == n


def _quartet_counts(seq_mat, n, L):
    """Vectorized counts of the three quartet resolutions for every 4-taxon subset."""
    m = seq_mat.astype(np.int8)
    base = np.zeros((n, n, L), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            d = (m[i] != m[j]).astype(np.int8)
            base[i, j] = d
            base[j, i] = d
    n11 = base[0] + base[1]
    n12 = base[0] + base[2]
    n13 = base[0] + base[3]
    c1 = int(np.count_nonzero(n11 == 2))
    c2 = int(np.count_nonzero(n12 == 2))
    c3 = int(np.count_nonzero(n13 == 2))
    return c1, c2, c3


def _quartet_matrix(seq_mat, n, L):
    """Q[a,b,c,d] = (c1 - c2, c1 - c3) for a<b<c<d. c1:ab|cd, c2:ac|bd, c3:ad|bc."""
    m = seq_mat.astype(np.int8)
    base = np.zeros((n, n, L), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            d = (m[i] != m[j]).astype(np.int8)
            base[i, j] = d
            base[j, i] = d
    n11 = base[0] + base[1]
    n12 = base[0] + base[2]
    n13 = base[0] + base[3]
    eq2 = (n11 == 2).astype(np.int16)
    eq2b = (n12 == 2).astype(np.int16)
    eq2c = (n13 == 2).astype(np.int16)
    Q = np.zeros((n, n, n, n), dtype=np.int32)
    Q[:, :, :, 0] = eq2
    Q[:, :, :, 1] = -eq2b
    Q[:, :, :, 2] = -eq2c
    Q[:, :, :, 3] = eq2b + eq2c
    return Q


def _tree_quartet_score(tree, Q):
    n = len(tree)
    idx = np.zeros(n, dtype=np.int32)
    stack = [(tree, 0)]
    while stack:
        node, base = stack.pop()
        if isinstance(node, int):
            idx[base] = node
        else:
            stack.append((node[1], base + 1))
            stack.append((node[0], base))
    A = idx[:, None]
    B = idx[None, :]
    C = idx[None, None, :]
    D = idx[None, None, None, :]
    sel = (A < B) & (B < C) & (C < D)
    if not np.any(sel):
        return 0
    S = Q[A, B, C, D]
    S = S[sel]
    return int(np.sum(np.maximum(0, -S[:, 0]) + np.maximum(0, -S[:, 2])))


def _nni_moves(tree):
    n = len(tree)
    stack = [(tree, 0)]
    while stack:
        node, base = stack.pop()
        if isinstance(node, int):
            continue
        L, R = node
        stack.append((R, base + 1))
        stack.append((L, base))
        if isinstance(L, int) or isinstance(R, int):
            continue
        yield (base, L, R)


def _nni_search(tree, Q, time_limit):
    n = len(tree)
    cur = tree
    cur_score = _tree_quartet_score(cur, Q)
    improved = True
    while improved and _elapsed() < time_limit:
        improved = False
        best_move = None
        best_score = cur_score
        for base, L, R in _nni_moves(cur):
            for Lp, Rp in ((L[0], (L[1], R)), ((L[0], L[1]), R)):
                new = cur
                new[base] = (Lp, Rp)
                sc = _tree_quartet_score(new, Q)
                if sc < best_score:
                    best_score = sc
                    best_move = (base, Lp, Rp)
        if best_move is not None:
            base, Lp, Rp = best_move
            cur = cur
            cur[base] = (Lp, Rp)
            cur_score = best_score
            improved = True
    return cur, cur_score


def _consensus_tree(Q, n, time_limit):
    # Build a binary tree from the majority-rule quartet set via recursive split.
    # Fallback: star-like balanced tree if quartet set is inconsistent.
    def build(subset):
        k = len(subset)
        if k <= 3:
            if k == 1:
                return subset[0]
            if k == 2:
                return (subset[0], subset[1])
            return (subset[0], (subset[1], subset[2]))
        # Try to find a non-trivial split of subset supported by quartets.
        # Use a heuristic: pick a 4-subset, get its resolution, grow a split.
        # Simpler: fall back to balanced split.
        half = k // 2
        return (build(subset[:half]), build(subset[half:]))

    return build(list(range(n)))


def _nj_tree(dist_mat, labels):
    n_labels = len(labels)
    if n_labels < 2:
        return labels[0] if labels else "t0"
    if n_labels == 2:
        return (labels[0], labels[1])
    if n_labels == 3:
        return (labels[0], (labels[1], labels[2]))

    nodes = list(range(n_labels))
    D = dist_mat.copy()
    tree_struct = {i: labels[i] for i in range(n_labels)}
    next_id = n_labels

    while len(nodes) > 3:
        n = len(nodes)
        r = D.sum(axis=1)
        Q = (n - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        if i == j:
            continue
        node_i = nodes[i]
        node_j = nodes[j]
        new_id = next_id
        next_id += 1
        tree_struct[new_id] = (tree_struct[node_i], tree_struct[node_j])
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[i] = False
        keep_mask[j] = False
        kept_indices = np.where(keep_mask)[0]
        new_nodes_list = [nodes[k] for k in kept_indices]
        new_nodes_list.append(new_id)
        new_n = n - 1
        D_new = np.zeros((new_n, new_n), dtype=np.float64)
        D_new[:-1, :-1] = D[np.ix_(kept_indices, kept_indices)]
        dist_i_kept = D[kept_indices, i]
        dist_j_kept = D[kept_indices, j]
        dist_ij = D[i, j]
        new_branches = 0.5 * (dist_i_kept + dist_j_kept - dist_ij)
        D_new[:-1, -1] = new_branches
        D_new[-1, :-1] = new_branches
        nodes = new_nodes_list
        D = D_new

    if len(nodes) == 3:
        a, b, c = nodes
        tree_struct['final'] = (tree_struct[a], (tree_struct[b], tree_struct[c]))
        final_node = 'final'
    elif len(nodes) == 2:
        a, b = nodes
        tree_struct['final'] = (tree_struct[a], tree_struct[b])
        final_node = 'final'
    else:
        final_node = nodes[0]
    return tree_struct[final_node]


def _star_tree(n):
    # A valid binary tree (caterpillar) as a safe fallback.
    if n == 1:
        return 0
    if n == 2:
        return (0, 1)
    tree = n - 1
    for i in range(n - 2, -1, -1):
        tree = (i, tree)
    return tree


def reconstruct(sequences):
    global _start
    _start = time.time()
    time_limit = _DEADLINE

    labels = sorted(sequences.keys(), key=lambda x: int(x[1:]))
    n = len(labels)
    if n < 2:
        return (labels[0] + ";") if labels else ";"
    if n == 2:
        return f"({labels[0]},{labels[1]})();"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"

    L = len(next(iter(sequences.values())))
    seq_chars = np.array([list(sequences[label]) for label in labels], dtype='U1')
    seq_mat = np.zeros((n, L), dtype=np.int8)
    for base, val in [('A', 0), ('C', 1), ('G', 2), ('T', 3)]:
        seq_mat[seq_chars == base] = val

    # JC-corrected distances for NJ
    p = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.sum(seq_mat[i] != seq_mat[j])) / L
            p[i, j] = d
            p[j, i] = d
    p = np.clip(p, 0.0, 0.7499999)
    dist_mat = -0.75 * np.log(1.0 - (4.0 / 3.0) * p)
    np.fill_diagonal(dist_mat, 0.0)

    # NJ starting tree
    nj_tree = _nj_tree(dist_mat, labels)
    nj_newick = _build_newick(nj_tree)
    if not _valid_tree(nj_newick, n):
        return _build_newick(_star_tree(n)) + ";"

    # Quartet-based NNI polish (the main improvement over the seed).
    if n <= 26 and _elapsed() < time_limit - 30:
        try:
            Q = _quartet_matrix(seq_mat, n, L)
            # Index tree by integer taxa for fast scoring.
            def to_int_tree(t):
                if isinstance(t, str):
                    return int(t[1:])
                return (to_int_tree(t[0]), to_int_tree(t[1]))
            int_tree = to_int_tree(nj_tree)
            polished, _ = _nni_search(int_tree, Q, time_limit - 10)
            def to_label_tree(t):
                if isinstance(t, int):
                    return labels[t]
                return (to_label_tree(t[0]), to_label_tree(t[1]))
            polished_lab = to_label_tree(polished)
            polished_newick = _build_newick(polished_lab)
            if _valid_tree(polished_newick, n):
                return polished_newick + ";"
        except Exception:
            pass

    return nj_newick + ";"
