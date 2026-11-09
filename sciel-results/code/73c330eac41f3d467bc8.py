import time
import numpy as np

# EVOLVE-BLOCK-START

def _encode(sequences, labels):
    """Encode DNA strings into an (n, L) uint8 array with values 0-3."""
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
    n = len(labels)
    L = len(sequences[labels[0]])
    A = np.empty((n, L), dtype=np.uint8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        # vectorize via translation
        A[i, :] = [char_map[c] for c in seq]
    return A


def _jc_distance_matrix(A):
    """Jukes-Cantor distance matrix from an (n, L) uint8 array."""
    n, L = A.shape
    if n < 2:
        return np.zeros((n, n))
    # Hamming distances
    diff = A[:, None, :] != A[None, :, :]
    hamming = np.sum(diff, axis=2)
    p = hamming / float(L)
    arg = 1.0 - 4.0 * p / 3.0
    arg = np.clip(arg, 1e-12, 1.0)
    d = -0.75 * np.log(arg)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return d


def _nj_tree(D, labels):
    """
    Neighbor-Joining returning a nested tuple tree.
    Leaves are (label,). Internal nodes are (left, right).
    """
    n = len(labels)
    if n == 1:
        return (labels[0],)
    if n == 2:
        return ((labels[0],), (labels[1],))

    # working distance matrix
    dist = D.astype(float).copy()
    # list of current node ids (0..n-1 are leaves)
    active = list(range(n))
    node_repr = {i: (labels[i],) for i in range(n)}
    next_id = n

    while len(active) > 2:
        m = len(active)
        # totals
        tot = np.array([dist[i, j] for i in active for j in active]).reshape(m, m)
        row_sum = tot.sum(axis=1)
        # Q matrix
        Q = (m - 2) * tot - row_sum[:, None] - row_sum[None, :]
        np.fill_diagonal(Q, 0)
        # find min off-diagonal
        i_idx, j_idx = np.unravel_index(np.argmin(Q), Q.shape)
        if i_idx == j_idx:
            i_idx, j_idx = 0, 1
        if i_idx > j_idx:
            i_idx, j_idx = j_idx, i_idx
        a = active[i_idx]
        b = active[j_idx]
        d_ab = dist[a, b]
        # new node
        node = next_id
        next_id += 1
        node_repr[node] = (node_repr[a], node_repr[b])
        # update distances
        rest = [x for x in active if x != a and x != b]
        for x in rest:
            d_ax = dist[a, x]
            d_bx = dist[b, x]
            d_new = 0.5 * (d_ax + d_bx - d_ab)
            dist[node, x] = d_new
            dist[x, node] = d_new
        # remove a, b
        active.remove(a)
        active.remove(b)
        active.append(node)
        # grow dist matrix
        if node not in dist.shape[0]:
            pass
        # rebuild dist as dict-free: we need dist to be square of size (max id +1)
        # Simplest: keep dist as a dict of pairs. Let's switch to a dict.
        break

    # The above got messy; use a clean dict-based NJ instead.
    return _nj_tree_dict(D, labels)


def _nj_tree_dict(D, labels):
    """Clean dict-based Neighbor-Joining returning nested tuples."""
    n = len(labels)
    if n == 1:
        return (labels[0],)
    if n == 2:
        return ((labels[0],), (labels[1],))

    dist = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[(i, j)] = float(D[i, j])

    active = list(range(n))
    node_repr = {i: (labels[i],) for i in range(n)}
    next_id = n

    while len(active) > 2:
        m = len(active)
        # row sums
        row_sum = {}
        for x in active:
            s = 0.0
            for y in active:
                if y != x:
                    s += dist[(x, y)]
            row_sum[x] = s
        # find min Q
        best = float('inf')
        a = b = None
        for ii, x in enumerate(active):
            for y in active[ii + 1:]:
                q = (m - 2) * dist[(x, y)] - row_sum[x] - row_sum[y]
                if q < best:
                    best = q
                    a, b = x, y
        node = next_id
        next_id += 1
        node_repr[node] = (node_repr[a], node_repr[b])
        d_ab = dist[(a, b)]
        rest = [x for x in active if x != a and x != b]
        for x in rest:
            d_new = 0.5 * (dist[(a, x)] + dist[(b, x)] - d_ab)
            dist[(node, x)] = d_new
            dist[(x, node)] = d_new
        active.remove(a)
        active.remove(b)
        active.append(node)

    a, b = active
    root = (node_repr[a], node_repr[b])
    return root


def _tree_to_newick(node):
    if len(node) == 1:
        return node[0]
    return "(" + _tree_to_newick(node[0]) + "," + _tree_to_newick(node[1]) + ")"


def _newick_to_tree(s):
    s = s.strip().rstrip(';')
    pos = [0]

    def parse():
        if s[pos[0]] == '(':
            pos[0] += 1
            left = parse()
            # skip ','
            if pos[0] < len(s) and s[pos[0]] == ',':
                pos[0] += 1
            right = parse()
            if pos[0] < len(s) and s[pos[0]] == ')':
                pos[0] += 1
            # skip branch length
            while pos[0] < len(s) and s[pos[0]] not in (',', ';', ')'):
                pos[0] += 1
            return (left, right)
        else:
            start = pos[0]
            while pos[0] < len(s) and s[pos[0]] not in (',', ';', ')'):
                pos[0] += 1
            return (s[start:pos[0]],)

    return parse()


def _fitch_cost(tree, A, label_to_idx, n_sites):
    """
    Fitch parsimony cost. Returns (cost, node_state).
    node_state: array of shape (n_nodes, n_sites) where each entry is a bitmask
    (4 bits) of the Fitch set at that node.
    """
    # First, build a list of nodes in post-order with their children.
    # We'll do a recursive traversal.
    # Represent each node by an id; build arrays.

    # Collect nodes
    node_list = []
    children_map = {}
    label_map = {}

    def collect(node):
        nid = len(node_list)
        node_list.append(nid)
        if len(node) == 1:
            label_map[nid] = node[0]
            children_map[nid] = []
        else:
            left = collect(node[0])
            right = collect(node[1])
            children_map[nid] = [left, right]
        return nid

    root = collect(tree)
    n_nodes = len(node_list)

    # state: (n_nodes, n_sites) uint8 bitmask
    state = np.zeros((n_nodes, n_sites), dtype=np.uint8)
    # leaf states
    for nid, lab in label_map.items():
        idx = label_to_idx[lab]
        # A[idx, :] is 0..3; convert to bitmask 1<<base
        state[nid, :] = (1 << A[idx, :]).astype(np.uint8)

    # post-order: process children before parents. Since collect appends children
    # before parent, node ids are in post-order already. So iterate 0..n_nodes-1.
    cost = 0
    for nid in range(n_nodes):
        kids = children_map[nid]
        if len(kids) == 2:
            s1 = state[kids[0], :]
            s2 = state[kids[1], :]
            inter = np.bitwise_and(s1, s2)
            union = np.bitwise_or(s1, s2)
            # where inter is 0, cost += 1 and use union
            no_inter = (inter == 0)
            cost += int(np.count_nonzero(no_inter))
            state[nid, :] = np.where(no_inter, union, inter)

    return cost, state


def _nni_search(tree, A, label_to_idx, n_sites, deadline):
    """
    NNI search using Fitch parsimony. Returns best tree.
    """
    n = len(label_to_idx)
    if n < 4:
        return tree

    best_tree = tree
    best_cost, _ = _fitch_cost(tree, A, label_to_idx, n_sites)

    def get_leaves(node):
        if len(node) == 1:
            return {node[0]}
        return get_leaves(node[0]) | get_leaves(node[1])

    # Precompute leaf sets for subtrees to speed up NNI candidate generation.
    # We'll work on the tree structure directly.

    # We need to find all internal edges. Represent tree as tuples.
    # For NNI, we consider each internal edge connecting two internal nodes.
    # For an edge between node X (children a,b) and parent P (children X, c),
    # the three topologies are: ((a,b),c), ((a,c),b), ((b,c),a).

    # We'll do a recursive search over the tree, for each internal node that is
    # a child of another internal node, try the two alternative NNI moves.

    # To evaluate a candidate, we build the new tree and compute Fitch cost.
    # Building the whole tree each time is O(n), and Fitch is O(n*L). With
    # O(n) edges, that's O(n^2 * L) per full pass. For n<=32, L<=2000, that's
    # ~32^2*2000 = 2M ops per pass, fine.

    improved = True
    while improved:
        improved = False
        if time.time() > deadline - 20:
            break

        # Find all NNI edges and evaluate moves.
        # We'll collect candidate new trees and their costs, pick the best.

        candidates = []  # (cost, new_tree)

        def recurse(node, parent, grandparent):
            """
            node: current subtree root.
            parent: parent internal node (tuple) or None.
            grandparent: grandparent internal node or None.
            For each internal edge (node, parent) where both are internal,
            generate NNI alternatives.
            """
            # Process children
            if len(node) == 2:
                # node is internal
                left, right = node[0], node[1]
                # If parent is internal, edge (node, parent) is an NNI edge.
                if parent is not None and len(parent) == 2:
                    # parent has children: one is node, the other is c.
                    if parent[0] is node:
                        c = parent[1]
                    elif parent[1] is node:
                        c = parent[0]
                    else:
                        c = None
                    if c is not None:
                        a, b = left, right
                        # Alternative 1: ((a,c),b)
                        # Alternative 2: ((b,c),a)
                        # Build new tree by replacing parent in grandparent context.
                        # We'll build the whole tree with the swap.
                        # Simpler: create a modified copy of the whole tree.
                        # We'll do a full rebuild via a function that takes the
                        # tree and the (a,b,c) and the parent identity.
                        # This is complex; instead, build new trees directly.

                        # We need to reconstruct the full tree with the NNI swap.
                        # Approach: define a function that builds the tree from the
                        # root, but when it reaches the parent node, it uses the
                        # swapped structure.
                        # We'll do a global rebuild each time.

                        # To keep it simple and correct, we'll build the full tree
                        # for each candidate by a targeted reconstruction.

                        # Let's build a helper that, given the original tree and a
                        # specification of which internal node (identified by its
                        # leaf-set) to swap, returns the new tree.

                        # For efficiency and correctness, I'll just rebuild the
                        # entire tree string with the swap applied via a recursive
                        # transform.

                        def build_swapped(original, target_parent, a, b, c, mode):
                            """
                            target_parent: the tuple (node, c) or (c, node) that is
                            the parent in the original tree.
                            a, b, c: subtrees.
                            mode: 1 for ((a,c),b), 2 for ((b,c),a).
                            """
                            def transform(t):
                                if t is target_parent:
                                    if mode == 1:
                                        return ((a, c), b)
                                    else:
                                        return ((b, c), a)
                                if len(t) == 1:
                                    return t
                                return (transform(t[0]), transform(t[1]))
                            return transform(original)

                        new_tree1 = build_swapped(tree, parent, a, b, c, 1)
                        new_tree2 = build_swapped(tree, parent, a, b, c, 2)

                        # Evaluate
                        cost1, _ = _fitch_cost(new_tree1, A, label_to_idx, n_sites)
                        cost2, _ = _fitch_cost(new_tree2, A, label_to_idx, n_sites)
                        if cost1 < best_cost:
                            candidates.append((cost1, new_tree1))
                        if cost2 < best_cost:
                            candidates.append((cost2, new_tree2))

                # Recurse into children
                recurse(left, node, parent)
                recurse(right, node, parent)
            else:
                # leaf: no children to recurse, but if parent is internal, the edge
                # (leaf, parent) is not an NNI edge (needs both internal). So skip.
                pass

        recurse(tree, None, None)

        if candidates:
            candidates.sort(key=lambda x: x[0])
            best_cost, best_tree = candidates[0]
            improved = True

    return best_tree


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 2:
        return "(" + ",".join(labels) + ");"

    L = len(sequences[labels[0]])
    A = _encode(sequences, labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    start_time = time.time()
    deadline = start_time + 240.0  # safety margin well under 270s

    # 1. NJ tree
    D = _jc_distance_matrix(A)
    tree = _nj_tree_dict(D, labels)

    # 2. NNI refinement
    if n >= 4 and time.time() < deadline - 30:
        tree = _nni_search(tree, A, label_to_idx, L, deadline)

    return _tree_to_newick(tree) + ";"

# EVOLVE-BLOCK-END
