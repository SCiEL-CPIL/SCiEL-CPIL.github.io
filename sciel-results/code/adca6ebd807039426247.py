"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Vectorized Neighbour Joining + NNI refinement on Jukes-Cantor corrected distances.'''
import time
import numpy as np
import re

_DEADLINE_START = None
_DEADLINE_LIMIT = 265.0


def _elapsed():
    return time.time() - _DEADLINE_START


def _encode_sequences(sequences, labels):
    """Fast vectorized encoding of sequences to int8 array."""
    n = len(labels)
    L = len(sequences[labels[0]])
    # Build a lookup table: ord('A')=65, ord('C')=67, ord('G')=71, ord('T')=84
    # Map to 0,1,2,3
    lut = np.full(256, -1, dtype=np.int8)
    lut[ord('A')] = 0
    lut[ord('C')] = 1
    lut[ord('G')] = 2
    lut[ord('T')] = 3
    enc = np.empty((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        # Convert string to bytes for fast indexing
        s_bytes = sequences[lab].encode('ascii')
        enc[i] = lut[s_bytes]
    return enc


def _jc_distances(mat):
    """mat: (n, L) int array with 0..3 encoding. Returns (n, n) JC distances."""
    n, L = mat.shape
    matches = np.zeros((n, n), dtype=np.float32)
    for c in range(4):
        Mc = (mat == c).astype(np.float32)
        matches += Mc @ Mc.T
    mismatches = L - matches
    p = mismatches / L
    np.fill_diagonal(p, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    d = np.where(p >= 0.75, 3.0, d)
    d = np.where(np.isnan(d) | np.isinf(d), 3.0, d)
    np.fill_diagonal(d, 0.0)
    return d


def _nj_with_structure(dist):
    """Neighbor Joining. Returns (newick_str, parent_map, children_map) for refinement."""
    n0 = dist.shape[0]
    if n0 < 3:
        if n0 == 1:
            return "0", {}, {}
        return "(0,1)", {}, {}

    max_nodes = 2 * n0 - 1
    D = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    D[:n0, :n0] = dist
    alive = list(range(n0))
    next_id = n0
    subtree = [str(i) for i in range(max_nodes)]
    # Track parent and children for tree structure
    parent = {}
    children = {}

    while len(alive) > 3:
        if _elapsed() > 255.0:
            break
        k = len(alive)
        idx = np.array(alive, dtype=np.int64)
        sub = D[np.ix_(idx, idx)].copy()
        r = sub.sum(axis=1)
        Q = (k - 2) * sub - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        min_idx = np.argmin(Q)
        i, j = np.unravel_index(min_idx, Q.shape)
        a = alive[i]
        b = alive[j]
        new = next_id
        next_id += 1
        subtree[new] = "(" + subtree[a] + "," + subtree[b] + ")"
        parent[a] = new
        parent[b] = new
        children[new] = [a, b]
        others = [x for x in alive if x != a and x != b]
        if others:
            ox = np.array(others, dtype=np.int64)
            d_ab = D[a, b]
            vals = 0.5 * (D[a, ox] + D[b, ox] - d_ab)
            D[new, ox] = vals
            D[ox, new] = vals
        alive = [x for x in alive if x != a and x != b]
        alive.append(new)

    if len(alive) == 3:
        a, b, c = alive
        # Create a root connecting all three
        root = next_id
        subtree[root] = "(" + subtree[a] + "," + subtree[b] + "," + subtree[c] + ")"
        parent[a] = root
        parent[b] = root
        parent[c] = root
        children[root] = [a, b, c]
        return subtree[root], parent, children
    return "(" + ",".join(subtree[x] for x in alive) + ")", parent, children


def _parse_newick_to_tree(newick, n):
    """Parse Newick into a tree representation for NNI. Returns list of (node_id, children)."""
    # We'll work with the parent/children dicts instead
    return None


def _nni_refine(newick, parent, children, n, dist, max_rounds=3):
    """Perform NNI refinement on the tree. Returns improved Newick string."""
    if _elapsed() > 255.0:
        return newick

    # Build a proper tree structure
    # Leaves are 0..n-1, internal nodes are n..2n-2
    # For NNI, we need to work with the tree topology

    # Extract splits from the current tree
    # A split is defined by a bipartition of the leaf set
    # For NNI on an unrooted binary tree with n leaves, there are n-3 internal edges

    # Build adjacency for the tree
    # Each internal node has 3 neighbors (in unrooted binary tree)
    # Let's build the tree as a graph

    # First, let's get all edges from parent/children
    # The tree is unrooted, so we need to handle the trifurcation at root

    # For simplicity, let's implement NNI by:
    # 1. Finding all internal edges
    # 2. For each, trying the two alternative splits
    # 3. Computing the total squared deviation for each
    # 4. Accepting if improvement

    # This is complex, so let's do a simpler approach:
    # Use the distance matrix to compute a "score" for the tree
    # and try local rearrangements

    # Actually, let's implement a proper NNI pass
    # For an unrooted tree with n leaves, there are n-3 internal edges
    # Each internal edge can be "flipped" in 2 ways (NNI)

    # Build the tree structure
    # Nodes: 0..n-1 are leaves, n..2n-2 are internal
    max_node = 2 * n - 2
    # Build adjacency list
    adj = {i: [] for i in range(max_node + 1)}

    for node, ch in children.items():
        for c in ch:
            adj[node].append(c)
            adj[c].append(node)

    # Find internal edges (edges where both endpoints are internal nodes)
    internal_edges = []
    for u in range(n, max_node + 1):
        for v in adj[u]:
            if v > n and v > u:
                internal_edges.append((u, v))

    if not internal_edges:
        return newick

    # For each internal edge, try NNI moves
    # NNI: given edge (u,v), the four neighbors are:
    #   u is connected to v, a, b (where a,b are the other two neighbors of u)
    #   v is connected to u, c, d (where c,d are the other two neighbors of v)
    # The three possible topologies are:
    #   (current) u-v with (a,b) on u side and (c,d) on v side
    #   (NNI1) u connected to a,c; v connected to b,d
    #   (NNI2) u connected to a,d; v connected to b,c

    # For scoring, we'll use the sum of squared differences between
    # pairwise distances implied by the tree and the observed distances
    # This is expensive, so let's use a simpler heuristic

    # Alternative: just return the NJ tree as-is if refinement is too complex
    # Let's implement a basic NNI with distance-based scoring

    def _tree_score(adj, dist):
        """Compute a score for the tree based on how well it fits the distances."""
        # For each pair of leaves, compute the tree distance
        # and compare to observed distance
        # This requires BFS for each pair, which is O(n^2 * n)
        # Too slow for large n, so use a simpler metric

        # Use the sum of edge lengths squared as a proxy
        # Actually, let's just count the number of splits that match
        # the "expected" splits from the distance matrix

        # Simplest: compute the total variation
        # For each internal edge, the split it defines should separate
        # close taxa from distant taxa

        # Let's use a different approach: compute the Q-matrix fit
        # Actually, for speed, let's just do a few NNI rounds with
        # a simple distance-based criterion

        # Compute pairwise tree distances via BFS
        n_leaves = n
        leaf_nodes = list(range(n_leaves))

        total_sq = 0.0
        count = 0
        for i in range(n_leaves):
            # BFS from leaf i
            from collections import deque
            dist_tree = [-1] * (max_node + 1)
            dist_tree[i] = 0
            queue = deque([i])
            while queue:
                node = queue.popleft()
                for nb in adj[node]:
                    if dist_tree[nb] == -1:
                        dist_tree[nb] = dist_tree[node] + 1
                        queue.append(nb)
            for j in range(i + 1, n_leaves):
                if dist_tree[j] > 0:
                    total_sq += (dist_tree[j] - dist[i, j]) ** 2
                    count += 1
        return total_sq / max(count, 1)

    current_adj = {k: list(v) for k, v in adj.items()}
    current_score = _tree_score(current_adj, dist)

    for round_num in range(max_rounds):
        if _elapsed() > 255.0:
            break
        improved = False
        for u, v in internal_edges:
            # Find the other two neighbors of u and v
            u_neighbors = [x for x in current_adj[u] if x != v]
            v_neighbors = [x for x in current_adj[v] if x != u]
            if len(u_neighbors) != 2 or len(v_neighbors) != 2:
                continue
            a, b = u_neighbors
            c, d = v_neighbors

            # Try NNI move 1: connect u-c, v-b; u-a, v-d
            # New edges: u-c, u-a (remove u-v, u-b, v-c, v-d)
            # Wait, let me think about this more carefully

            # Current: u is connected to v, a, b
            #           v is connected to u, c, d
            # NNI alternative 1:
            #   u connected to a, c
            #   v connected to b, d
            #   (remove u-v, u-b, v-c; add u-c, v-b)
            # Wait, that's not right either.

            # Proper NNI:
            # The edge u-v divides the tree into two parts.
            # On u's side: u is connected to a and b (besides v)
            # On v's side: v is connected to c and d (besides u)
            #
            # Alternative 1:
            #   Remove edges: u-v, u-b, v-c
            #   Add edges: u-c, v-b
            #   Now u is connected to a, c; v is connected to b, d
            #
            # Alternative 2:
            #   Remove edges: u-v, u-b, v-d
            #   Add edges: u-d, v-b
            #   Now u is connected to a, d; v is connected to b, c

            # Try alternative 1
            new_adj1 = {k: list(v2) for k, v2 in current_adj.items()}
            new_adj1[u].remove(v)
            new_adj1[v].remove(u)
            new_adj1[u].remove(b)
            new_adj1[v].remove(c)
            new_adj1[u].append(c)
            new_adj1[c].append(u)
            new_adj1[v].append(b)
            new_adj1[b].append(v)

            score1 = _tree_score(new_adj1, dist)
            if score1 < current_score - 1e-9:
                current_adj = new_adj1
                current_score = score1
                improved = True

            # Try alternative 2
            new_adj2 = {k: list(v2) for k, v2 in current_adj.items()}
            new_adj2[u].remove(v)
            new_adj2[v].remove(u)
            new_adj2[u].remove(b)
            new_adj2[v].remove(d)
            new_adj2[u].append(d)
            new_adj2[d].append(u)
            new_adj2[v].append(b)
            new_adj2[b].append(v)

            score2 = _tree_score(new_adj2, dist)
            if score2 < current_score - 1e-9:
                current_adj = new_adj2
                current_score = score2
                improved = True

        if not improved:
            break

    # Convert the refined adjacency back to Newick
    # Find a root (any internal node)
    root = None
    for node in range(n, max_node + 1):
        if node in current_adj and len(current_adj[node]) == 3:
            root = node
            break
    if root is None:
        for node in range(n, max_node + 1):
            if node in current_adj:
                root = node
                break
    if root is None:
        return newick

    # DFS to build Newick
    def _build_newick(node, visited):
        if node < n:
            return str(node)
        visited.add(node)
        parts = []
        for nb in current_adj[node]:
            if nb not in visited:
                parts.append(_build_newick(nb, visited))
        return "(" + ",".join(parts) + ")"

    visited = set()
    result = _build_newick(root, visited)
    return result


def _reconstruct_single(sequences):
    global _DEADLINE_START, _DEADLINE_LIMIT
    if _DEADLINE_START is None:
        _DEADLINE_START = time.time()
        _DEADLINE_LIMIT = _DEADLINE_START + 265.0

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n < 3:
        if n == 1:
            return labels[0] + ";"
        return "(" + ",".join(labels) + ");"

    enc = _encode_sequences(sequences, labels)
    D = _jc_distances(enc)
    newick, parent, children = _nj_with_structure(D)

    # Try NNI refinement if time permits
    if _elapsed() < 240.0 and n >= 4:
        try:
            newick = _nni_refine(newick, parent, children, n, D, max_rounds=2)
        except Exception:
            pass

    def _repl(m):
        return labels[int(m.group(0))]
    result = re.sub(r'\d+', _repl, newick)
    return result


def reconstruct(sequences):
    global _DEADLINE_START, _DEADLINE_LIMIT
    if _DEADLINE_START is None:
        _DEADLINE_START = time.time()
        _DEADLINE_LIMIT = _DEADLINE_START + 265.0

    try:
        return _reconstruct_single(sequences)
    except Exception:
        labels = sorted(sequences, key=lambda name: int(name[1:]))
        if len(labels) <= 2:
            return "(" + ",".join(labels) + ");"
        return "(" + ",".join(labels) + ");"
# EVOLVE-BLOCK-END
