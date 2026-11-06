"""Phylogenetic topology recovery: BionJ distances + NJ + correct NNI with JC-ML scoring."""
import math
import time
import numpy as np

# ----------------------------- Time budget -----------------------------
_T0 = time.time()
_DEADLINE = _T0 + 250.0

def _time_left():
    return _DEADLINE - time.time()

# ----------------------------- Distance computation -----------------------------
def _jc_distances(sequences, labels):
    """Compute JC-corrected pairwise distances, vectorized."""
    n = len(labels)
    code = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(sequences[labels[0]])
    arr = np.empty((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        arr[i] = np.fromiter((code[c] for c in s), dtype=np.int8, count=L)
    dist = np.zeros((n, n), dtype=np.float64)
    block = 64
    for i in range(0, n, block):
        jmax = min(i + block, n)
        diff = arr[i:jmax, None, :] != arr[None, :, :]
        mism = diff.sum(axis=2)
        p = mism / float(L)
        np.clip(p, 0.0, 0.749999, out=p)
        d = -0.75 * np.log(1.0 - (4.0 / 3.0) * p)
        np.fill_diagonal(d, 0.0)
        dist[i:jmax, :] = d
    return dist

def _bionj_distances(dist, n):
    """BionJ correction for JC model."""
    if n <= 2:
        return dist
    r = dist.sum(axis=1) / (n - 1)
    d = dist.copy()
    for i in range(n):
        for j in range(i + 1, n):
            dij = d[i, j]
            corr = (r[i] + r[j]) / (2.0 * (n - 2)) * (1.0 + dij / (n - 2))
            nd = dij + corr
            d[i, j] = nd
            d[j, i] = nd
    return d

# ----------------------------- Neighbor-Joining (clean) -----------------------------
def _nj_clean(dist, labels):
    """Clean Neighbor-Joining. Returns (adj, leaf_nodes, n_nodes)."""
    n = len(labels)
    if n == 2:
        adj = {0: [(1, 0.0)], 1: [(0, 0.0)]}
        return adj, [0, 1], 2
    if n == 3:
        adj = {
            0: [(3, 0.0)],
            1: [(3, 0.0)],
            2: [(3, 0.0)],
            3: [(0, 0.0), (1, 0.0), (2, 0.0)]
        }
        return adj, [0, 1, 2], 4

    D = dist.copy()
    active = list(range(n))
    next_id = n
    adj = {i: [] for i in range(n)}
    # We'll build the tree as we go by tracking merges

    # For NJ, we need to track which nodes are joined.
    # Use a list of (node_id, repr) but for adjacency we just need edges.
    # Let's maintain a set of edges.
    edges = set()

    while len(active) > 3:
        m = len(active)
        r = D[np.ix_(active, active)].sum(axis=1)
        best = float('inf')
        bi = bj = -1
        bi_idx = bj_idx = -1
        for ai in range(m):
            for aj in range(ai + 1, m):
                i = active[ai]
                j = active[aj]
                q = (m - 2) * D[i, j] - r[ai] - r[aj]
                if q < best:
                    best = q
                    bi, bj = i, j
                    bi_idx, bj_idx = ai, aj

        # Join bi and bj
        new_id = next_id
        next_id += 1

        # Compute new distances
        rest = [k for k in active if k != bi and k != bj]
        new_d = {}
        for k in rest:
            new_d[k] = 0.5 * (D[bi, k] + D[bj, k] - D[bi, bj])

        # Update D in-place (careful with indices)
        # We'll rebuild D for the new active set
        new_active = rest + [new_id]
        new_D = np.zeros((len(new_active), len(new_active)))
        # Map old to new indices
        idx_map = {v: i for i, v in enumerate(new_active)}
        for i_old, u in enumerate(active):
            if u == bi or u == bj:
                continue
            for j_old, v in enumerate(active):
                if v == bi or v == bj:
                    continue
                if i_old == j_old:
                    continue
                new_D[idx_map[u], idx_map[v]] = D[u, v]

        # Add new node distances
        for k in rest:
            new_D[idx_map[new_id], idx_map[k]] = new_d[k]
            new_D[idx_map[k], idx_map[new_id]] = new_d[k]

        # Record edges: bi-new_id and bj-new_id
        # In the final tree, we need to connect bi and bj via new_id.
        # We store this in a merge list: (bi, bj, new_id)
        # But for adjacency, we just add edges (bi, new_id) and (bj, new_id)
        edges.add((bi, new_id))
        edges.add((bj, new_id))

        active = new_active
        D = new_D

    # Final 3 nodes
    a, b, c = active
    root_id = next_id
    next_id += 1
    edges.add((a, root_id))
    edges.add((b, root_id))
    edges.add((c, root_id))

    # Build adjacency
    n_nodes = next_id
    adj = {i: [] for i in range(n_nodes)}
    for u, v in edges:
        adj[u].append((v, 0.0))
        adj[v].append((u, 0.0))

    leaf_nodes = [i for i in range(n) if len(adj[i]) == 1]
    return adj, leaf_nodes, n_nodes

# ----------------------------- JC likelihood (Felsenstein) -----------------------------
def _jc_pmat(L):
    a = math.exp(-4.0 * L / 3.0)
    b = (1.0 - a) / 3.0
    M = np.full((4, 4), b)
    np.fill_diagonal(M, a)
    return M

def _jc_likelihood(adj, leaf_nodes, label_to_leaf, seq_codes, L, root):
    """Compute JC log-likelihood using Felsenstein's pruning algorithm.
    adj: dict node -> list of (neighbor, length)
    leaf_nodes: list of leaf node ids
    label_to_leaf: dict label -> leaf node id
    seq_codes: dict label -> np.array of int codes (0-3)
    L: number of sites
    root: root node id
    """
    n = len(leaf_nodes)
    # Build rooted tree
    parent = {root: -1}
    children = {u: [] for u in adj}
    stack = [root]
    visited = {root}
    while stack:
        u = stack.pop()
        for (v, l) in adj[u]:
            if v in visited:
                continue
            visited.add(v)
            parent[v] = u
            children[u].append(v)
            stack.append(v)

    # Order nodes post-order
    order = []
    stack = [root]
    while stack:
        u = stack.pop()
        order.append(u)
        for v in children[u]:
            stack.append(v)
    order.reverse()

    # Felsenstein's algorithm
    # P[u] = vector of probabilities of states at node u given the data below u
    # For leaves, P[u] = 1 if state matches data, else 0 (for each site)
    # We compute log-likelihood by summing over sites.
    # To be efficient, we process all sites together using matrices.

    # For each node, we store a 4x4 matrix? No, we store a vector of 4 probabilities.
    # But we need to handle multiple sites. We'll accumulate log-likelihood per site.
    # This is slow. Instead, we can use the fact that JC is stationary and symmetric.
    # Actually, for JC, the likelihood can be computed more efficiently.
    # But let's stick to the standard approach for correctness.

    # We'll compute the likelihood for each site separately and sum log-likelihoods.
    # This is O(n*L*4^2) which is acceptable for small n and L.

    loglik = 0.0
    for site in range(L):
        # Initialize leaf probabilities
        P = {}
        for u in order:
            if u in leaf_nodes:
                # Find the label for this leaf
                label = None
                for lab, ln in label_to_leaf.items():
                    if ln == u:
                        label = lab
                        break
                if label is not None:
                    obs = seq_codes[label][site]
                    P[u] = np.zeros(4)
                    P[u][obs] = 1.0
                else:
                    P[u] = np.ones(4) / 4.0
            else:
                P[u] = np.zeros(4)

        # Process internal nodes in post-order
        for u in order:
            if u in leaf_nodes:
                continue
            if not children[u]:
                continue
            # Start with prior 1/4 for each state
            P[u] = np.ones(4) / 4.0
            for v in children[u]:
                # Get branch length
                blen = 0.0
                for (w, l) in adj[u]:
                    if w == v:
                        blen = l
                        break
                M = _jc_pmat(blen)
                # P[u] *= M @ P[v] (element-wise)
                P[u] *= M @ P[v]

        # Root likelihood
        ll_site = np.sum(P[root])
        if ll_site > 0:
            loglik += math.log(ll_site)
        else:
            loglik += -1000.0

    return loglik

# ----------------------------- NNI with JC ML -----------------------------
def _nni_ml(adj, leaf_nodes, label_to_leaf, seq_codes, L, time_limit):
    """Perform NNI local search with JC ML scoring."""
    n = len(leaf_nodes)
    if n <= 3:
        return adj, 0.0

    # Find root: pick an internal node
    internal = [u for u in adj if len(adj[u]) != 1]
    if not internal:
        return adj, 0.0
    root = internal[0]

    # Compute initial likelihood
    current_loglik = _jc_likelihood(adj, leaf_nodes, label_to_leaf, seq_codes, L, root)

    # Get all internal edges
    def get_internal_edges(adj):
        edges = []
        seen = set()
        for u in adj:
            for (v, l) in adj[u]:
                if u < v and len(adj[u]) > 1 and len(adj[v]) > 1:
                    edges.append((u, v))
                elif u > v and len(adj[u]) > 1 and len(adj[v]) > 1:
                    pass  # already counted
        return edges

    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        edges = get_internal_edges(adj)
        for (u, v) in edges:
            # Find the two subtrees attached to u and v
            # u has neighbors: v and one other (a)
            # v has neighbors: u and one other (b)
            # So the four subtrees are: a, b, and the two sides of u-v
            # Actually, for NNI, we consider the edge (u, v).
            # u has degree 3: neighbors v, a, b
            # v has degree 3: neighbors u, c, d
            # Wait, in a binary tree, internal nodes have degree 3.
            # So u has neighbors: v, a, b
            # v has neighbors: u, c, d
            # The three NNI alternatives swap the connections.

            # Get neighbors of u and v
            u_neighbors = [w for (w, l) in adj[u] if w != v]
            v_neighbors = [w for (w, l) in adj[v] if w != u]

            if len(u_neighbors) != 2 or len(v_neighbors) != 2:
                continue

            a, b = u_neighbors
            c, d = v_neighbors

            # Try NNI move: swap b and c
            # New edges: u-c, v-b, and remove u-v, u-b, v-c
            # Wait, the standard NNI on edge (u, v) where u connects to a, b and v connects to c, d:
            # The three possible splits are:
            # 1. (a,b)|(c,d) [current if u is between a,b and v is between c,d]
            # 2. (a,c)|(b,d)
            # 3. (a,d)|(b,c)
            # We need to identify which split is current.
            # Current: u connects to a, b, v. v connects to u, c, d.
            # So the split induced by edge (u, v) is (a, b, ...)| (c, d, ...)
            # Actually, the split is (leaves under u excluding v) | (leaves under v excluding u)
            # For NNI, we swap one of the neighbors.

            # Let's try all 3 configurations and pick the best.
            # Config 0: current (u-v, u-a, u-b, v-c, v-d)
            # Config 1: (u-c, u-a, v-b, v-d) -> swap b and c
            # Config 2: (u-d, u-a, v-b, v-c) -> swap b and d

            best_loglik = current_loglik
            best_config = 0

            # Try config 1: swap b and c
            # Remove edges: (u, v), (u, b), (v, c)
            # Add edges: (u, c), (v, b)
            # But we must preserve the rest of the tree.
            # This is complex. Let's just try and compute likelihood.

            # For simplicity, we'll only try one NNI move per edge per iteration.
            # Swap b and c.
            new_adj = _try_nni_swap(adj, u, v, b, c)
            if new_adj is not None:
                new_loglik = _jc_likelihood(new_adj, leaf_nodes, label_to_leaf, seq_codes, L, root)
                if new_loglik > best_loglik:
                    best_loglik = new_loglik
                    best_config = 1
                    best_adj = new_adj

            # Try config 2: swap b and d
            new_adj2 = _try_nni_swap(adj, u, v, b, d)
            if new_adj2 is not None:
                new_loglik2 = _jc_likelihood(new_adj2, leaf_nodes, label_to_leaf, seq_codes, L, root)
                if new_loglik2 > best_loglik:
                    best_loglik = new_loglik2
                    best_config = 2
                    best_adj = new_adj2

            if best_config > 0:
                adj = best_adj
                current_loglik = best_loglik
                improved = True
                if _time_left() < 5.0:
                    break

    return adj, current_loglik

def _try_nni_swap(adj, u, v, b, c):
    """Try swapping neighbors b and c in the NNI move on edge (u, v).
    Returns new adjacency if valid, else None.
    """
    # Check that u is connected to b and v is connected to c
    u_has_b = any(w == b for (w, l) in adj[u])
    v_has_c = any(w == c for (w, l) in adj[v])
    if not u_has_b or not v_has_c:
        return None

    # Also need u connected to v
    u_has_v = any(w == v for (w, l) in adj[u])
    v_has_u = any(w == u for (w, l) in adj[v])
    if not u_has_v or not v_has_u:
        return None

    # Build new adjacency
    new_adj = {k: list(vv) for k, vv in adj.items()}

    # Remove edges (u, v), (u, b), (v, c)
    new_adj[u] = [(w, l) for (w, l) in new_adj[u] if w not in (v, b)]
    new_adj[v] = [(w, l) for (w, l) in new_adj[v] if w not in (u, c)]
    new_adj[b] = [(w, l) for (w, l) in new_adj[b] if w != u]
    new_adj[c] = [(w, l) for (w, l) in new_adj[c] if w != v]

    # Add edges (u, c), (v, b)
    new_adj[u].append((c, 0.0))
    new_adj[c].append((u, 0.0))
    new_adj[v].append((b, 0.0))
    new_adj[b].append((v, 0.0))

    # Verify degrees
    for node in new_adj:
        if len(new_adj[node]) != 3 and len(new_adj[node]) != 1:
            return None

    return new_adj

# ----------------------------- Newick output -----------------------------
def _tree_to_newick(adj, leaf_nodes, label_to_leaf):
    """Convert adjacency to Newick string."""
    n = len(leaf_nodes)
    if n == 2:
        l1 = [lab for lab, ln in label_to_leaf.items() if ln == leaf_nodes[0]][0]
        l2 = [lab for lab, ln in label_to_leaf.items() if ln == leaf_nodes[1]][0]
        return f"({l1},{l2})"
    if n == 3:
        l1 = [lab for lab, ln in label_to_leaf.items() if ln == leaf_nodes[0]][0]
        l2 = [lab for lab, ln in label_to_leaf.items() if ln == leaf_nodes[1]][0]
        l3 = [lab for lab, ln in label_to_leaf.items() if ln == leaf_nodes[2]][0]
        return f"(({l1},{l2}),{l3})"

    # Root at an internal node
    internal = [u for u in adj if len(adj[u]) != 1]
    if not internal:
        internal = list(adj.keys())
    root = internal[0]

    # DFS
    parent = {root: -1}
    stack = [root]
    visited = {root}
    while stack:
        u = stack.pop()
        for (v, l) in adj[u]:
            if v in visited:
                continue
            visited.add(v)
            parent[v] = u
            stack.append(v)

    children = {u: [] for u in adj}
    for v in adj:
        if parent.get(v, -1) != -1:
            children[parent[v]].append(v)

    node_to_label = {ln: lab for lab, ln in label_to_leaf.items()}

    def emit(u):
        if len(children[u]) == 0:
            return node_to_label.get(u, f"t{u}")
        parts = [emit(c) for c in children[u]]
        return "(" + ",".join(parts) + ")"

    return emit(root)

# ----------------------------- Main -----------------------------
def reconstruct(sequences):
    global _T0, _DEADLINE
    _T0 = time.time()
    _DEADLINE = _T0 + 250.0

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    if n == 3:
        a, b, c = labels
        return f"(({a},{b}),{c});"

    # 1. BionJ distances
    dist = _jc_distances(sequences, labels)
    dist = _bionj_distances(dist, n)

    # 2. NJ tree
    adj, leaf_nodes, n_nodes = _nj_clean(dist, labels)

    # Map labels to leaf nodes
    label_to_leaf = {labels[i]: leaf_nodes[i] for i in range(n)}

    # 3. NNI refinement with JC ML
    if _time_left() > 10.0:
        # Encode sequences
        code = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        L = len(sequences[labels[0]])
        seq_codes = {lab: np.array([code[c] for c in sequences[lab]], dtype=np.int8) for lab in labels}

        adj, loglik = _nni_ml(adj, leaf_nodes, label_to_leaf, seq_codes, L, _time_left())

    # 4. Output Newick
    newick = _tree_to_newick(adj, leaf_nodes, label_to_leaf)
    return newick + ";"
