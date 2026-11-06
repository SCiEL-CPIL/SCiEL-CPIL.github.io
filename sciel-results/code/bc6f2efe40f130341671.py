"""Maximum likelihood phylogenetic tree reconstruction with NNI refinement under Jukes-Cantor."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

# Global deadline guard
_TOTAL_BUDGET = 250.0  # seconds, safety margin below 270s
_start_time = None

def _time_remaining():
    if _start_time is None:
        return _TOTAL_BUDGET
    return _TOTAL_BUDGET - (time.time() - _start_time)

def _jc_distance(a, b):
    """Jukes-Cantor corrected distance between two DNA sequences."""
    n = len(a)
    if n == 0:
        return 0.0
    diff = sum(1 for x, y in zip(a, b) if x != y)
    p = diff / n
    if p >= 0.7499:
        return 10.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _build_distance_matrix(sequences, labels):
    """Build pairwise JC distance matrix."""
    n = len(labels)
    dist = np.zeros((n, n))
    seq_list = [sequences[l] for l in labels]
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(seq_list[i], seq_list[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist

def _nj_build(dist, n):
    """Build a Neighbor Joining tree. Returns adjacency list representation.
    Nodes 0..n-1 are taxa, nodes n..2n-3 are internal.
    Returns (edges, num_nodes) where edges is a list of (u, v) pairs.
    """
    if n == 1:
        return [(0, 0)], 1
    if n == 2:
        return [(0, 1)], 2
    if n == 3:
        # For 3 taxa, create a star topology
        # Internal node 3 connects all three
        return [(0, 3), (1, 3), (2, 3)], 4

    # Active nodes: initially 0..n-1
    active = list(range(n))
    dist = dist.copy()
    edges = []
    next_node = n

    while len(active) > 2:
        m = len(active)
        # Compute Q matrix for active nodes
        # r[i] = sum of distances from active[i] to all other active nodes
        r = np.zeros(m)
        for i in range(m):
            for j in range(m):
                if i != j:
                    r[i] += dist[active[i], active[j]]

        # Find min Q value
        best_q = float('inf')
        best_i, best_j = -1, -1
        for i in range(m):
            for j in range(i + 1, m):
                q = (m - 2) * dist[active[i], active[j]] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i, best_j = i, j

        a, b = active[best_i], active[best_j]

        # Compute branch lengths
        ra = 0.5 * (dist[a, b] + (r[a] - r[b]) / (m - 2))
        rb = dist[a, b] - ra

        # Add edges
        edges.append((a, next_node))
        edges.append((b, next_node))

        # Update distances from new node to all other active nodes
        for j in range(m):
            if j == best_i or j == best_j:
                continue
            c = active[j]
            dist[next_node, c] = 0.5 * (dist[a, c] + dist[b, c] - dist[a, b])
            dist[c, next_node] = dist[next_node, c]

        # Remove a and b from active, add next_node
        active.remove(a)
        active.remove(b)
        active.append(next_node)
        next_node += 1

    # Connect the last two nodes
    if len(active) == 2:
        u, v = active[0], active[1]
        edges.append((u, v))

    num_nodes = next_node
    return edges, num_nodes

def _edges_to_adjacency(edges, num_nodes, labels):
    """Convert edge list to adjacency structure for tree manipulation.
    Returns (adj, node_labels) where adj[u] is a list of (v, branch_len) pairs.
    We don't have branch lengths from NJ in this simple form, so we use 0.5 as placeholder.
    """
    adj = [[] for _ in range(num_nodes)]
    for u, v in edges:
        adj[u].append((v, 0.5))
        adj[v].append((u, 0.5))
    return adj

def _find_leaf_nodes(adj, num_nodes, n_taxa):
    """Find which nodes are leaves (taxa 0..n_taxa-1)."""
    leaves = set(range(n_taxa))
    return leaves

def _compute_likelihood(adj, num_nodes, n_taxa, labels, sequences):
    """Compute log-likelihood of the tree under Jukes-Cantor using Felsenstein's algorithm.
    adj: list of (neighbor, branch_length) pairs
    We root the tree arbitrarily at a leaf node for computation.
    """
    n_seq = len(sequences)
    seq_list = [sequences[l] for l in labels]
    n_sites = len(seq_list[0]) if seq_list else 0

    if n_sites == 0:
        return 0.0

    # Root the tree at node 0 (a taxon)
    root = 0
    parent = [-1] * num_nodes
    order = []
    stack = [root]
    parent[root] = -2  # mark as root
    while stack:
        node = stack.pop()
        order.append(node)
        for nb, bl in adj[node]:
            if parent[nb] == -1:
                parent[nb] = node
                stack.append(nb)

    # Process in post-order (reverse of pre-order)
    # For each node, compute the likelihood profile (4 states)
    # P[node][state] = probability of observing all sequences in the subtree given node is in that state
    profiles = [None] * num_nodes
    # State probabilities for JC: equal transition
    # P(a->b, t) = 0.25 + 0.75 * exp(-4t/3) if a!=b, 0.25*(1+3*exp(-4t/3)) if a==b
    # Actually for JC: P(same) = 0.25 + 0.75*exp(-4t/3), P(diff) = 0.25*(1-exp(-4t/3))

    # Process nodes in reverse order (leaves first)
    for node in reversed(order):
        # Get children (neighbors excluding parent)
        children = [nb for nb, bl in adj[node] if parent[nb] == node]

        if not children:
            # Leaf node: this is a taxon
            taxon_idx = node  # nodes 0..n_taxa-1 are taxa
            if taxon_idx < n_taxa:
                seq = seq_list[taxon_idx]
                profile = np.zeros(n_sites)
                for s in range(n_sites):
                    ch = seq[s]
                    # Map A,C,G,T to 0,1,2,3
                    state = {'A': 0, 'C': 1, 'G': 2, 'T': 3}.get(ch, 0)
                    profile[s] = 1.0 if state == state else 0.0
                    # Actually, profile[s] should be a vector of 4 states
                    # Let's use a different approach: profile[s] is a 4-dim vector
                # Redo: profile is (n_sites, 4)
                profile = np.zeros((n_sites, 4))
                for s in range(n_sites):
                    ch = seq[s]
                    state = {'A': 0, 'C': 1, 'G': 2, 'T': 3}.get(ch, 0)
                    profile[s, state] = 1.0
            else:
                # Should not happen for leaves
                profile = np.ones((n_sites, 4)) * 0.25
        else:
            # Internal node: multiply contributions from children
            profile = np.ones((n_sites, 4))
            for child in children:
                # Find branch length to this child
                bl = 0.5  # default
                for nb, b in adj[node]:
                    if nb == child:
                        bl = b
                        break
                # Transition matrix for JC over branch length bl
                exp_term = math.exp(-4.0 * bl / 3.0)
                p_same = 0.25 + 0.75 * exp_term
                p_diff = 0.25 * (1.0 - exp_term)

                # For each site, multiply profile by transition contribution
                # child_profile[s, state_child] * P(state_parent -> state_child)
                # We need: new_profile[s, state_parent] = sum over state_child of P(state_parent->state_child) * child_profile[s, state_child]
                # But since we're computing bottom-up, we multiply into the parent profile
                for s in range(n_sites):
                    for sp in range(4):
                        val = 0.0
                        for sc in range(4):
                            if sp == sc:
                                p = p_same
                            else:
                                p = p_diff
                            val += p * profiles[child][s, sc]
                        profile[s, sp] *= val

        profiles[node] = profile

    # Root likelihood: average over 4 states (unrooted)
    total_ll = 0.0
    root_profile = profiles[root]
    for s in range(n_sites):
        total_ll += math.log(sum(root_profile[s]) / 4.0)

    return total_ll

def _nni_search(adj, num_nodes, n_taxa, labels, sequences, max_iterations=50):
    """Perform NNI refinement to improve log-likelihood.
    Returns the improved adjacency structure.
    """
    current_ll = _compute_likelihood(adj, num_nodes, n_taxa, labels, sequences)

    for iteration in range(max_iterations):
        if _time_remaining() < 5.0:
            break

        improved = False

        # For each internal edge, consider NNI moves
        # An NNI move involves a 4-taxon subtree: a-b-c-d connected as (a,b)(c,d) or (a,c)(b,d) or (a,d)(b,c)
        # We need to find all internal edges and try swapping.

        # Build a set of edges for quick lookup
        edge_set = set()
        for u in range(num_nodes):
            for v, bl in adj[u]:
                if u < v:
                    edge_set.add((u, v))

        for u, v in list(edge_set):
            if _time_remaining() < 5.0:
                break

            # Check if u and v are both internal nodes (not taxa)
            if u < n_taxa or v < n_taxa:
                continue

            # Get neighbors of u (excluding v) and v (excluding u)
            neighbors_u = [nb for nb, bl in adj[u] if nb != v]
            neighbors_v = [nb for nb, bl in adj[v] if nb != u]

            if len(neighbors_u) < 1 or len(neighbors_v) < 1:
                continue

            # For each pair (a in neighbors_u, c in neighbors_v), try the NNI move
            for a in neighbors_u:
                for c in neighbors_v:
                    # Current topology: u-v edge, with a connected to u and c connected to v
                    # NNI alternatives:
                    # 1. (a,c) and (u,v) -> (a,u) and (c,v) [current]
                    # 2. (a,c) and (u,v) -> (a,v) and (c,u)
                    # 3. (a,c) and (u,v) -> (u,c) and (v,a) -- wait, need to be careful

                    # Let's try alternative 2: swap so that a connects to v and c connects to u
                    # Save original connections
                    orig_u_neighbors = [(nb, bl) for nb, bl in adj[u]]
                    orig_v_neighbors = [(nb, bl) for nb, bl in adj[v]]

                    # Remove u-v edge
                    adj[u] = [(nb, bl) for nb, bl in adj[u] if nb != v]
                    adj[v] = [(nb, bl) for nb, bl in adj[v] if nb != u]

                    # Remove a from u, add to v
                    adj[u] = [(nb, bl) for nb, bl in adj[u] if nb != a]
                    adj[v] = [(nb, bl) for nb, bl in adj[v] if nb != a]

                    # Remove c from v, add to u
                    adj[v] = [(nb, bl) for nb, bl in adj[v] if nb != c]
                    adj[u] = [(nb, bl) for nb, bl in adj[u] if nb != c]

                    # Add new connections
                    adj[u].append((c, 0.5))
                    adj[v].append((a, 0.5))

                    # Compute new likelihood
                    new_ll = _compute_likelihood(adj, num_nodes, n_taxa, labels, sequences)

                    if new_ll > current_ll + 1e-10:
                        current_ll = new_ll
                        improved = True
                    else:
                        # Revert
                        adj[u] = orig_u_neighbors[:]
                        adj[v] = orig_v_neighbors[:]

                    if _time_remaining() < 5.0:
                        break
                if _time_remaining() < 5.0:
                    break
            if _time_remaining() < 5.0:
                break

        if not improved:
            break

    return adj

def _adjacency_to_newick(adj, num_nodes, n_taxa, labels):
    """Convert adjacency structure to Newick string.
    We need to find a root and do a DFS.
    """
    # Find a leaf to use as root
    root = 0
    while root < num_nodes:
        if root < n_taxa:
            break
        root += 1

    # Build parent-child structure via DFS
    parent = [-1] * num_nodes
    children = [[] for _ in range(num_nodes)]
    parent[root] = -2
    stack = [root]
    while stack:
        node = stack.pop()
        for nb, bl in adj[node]:
            if parent[nb] == -1:
                parent[nb] = node
                children[node].append((nb, bl))
                stack.append(nb)

    # Recursive Newick generation
    def newick_node(node):
        if node < n_taxa:
            return labels[node]
        if not children[node]:
            return f"({labels[0]},{labels[1]})" if n_taxa >= 2 else labels[0]
        parts = [newick_node(c) for c, bl in children[node]]
        return "(" + ",".join(parts) + ")"

    return newick_node(root) + ";"

def reconstruct(sequences):
    """Reconstruct phylogenetic tree topology from DNA sequences."""
    global _start_time
    _start_time = time.time()

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)

    if n_taxa == 0:
        return ""
    if n_taxa == 1:
        return labels[0] + ";"
    if n_taxa == 2:
        return f"({labels[0]},{labels[1]});"
    if n_taxa == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"

    # Step 1: Build distance matrix
    dist = _build_distance_matrix(sequences, labels)

    # Step 2: Build NJ tree
    edges, num_nodes = _nj_build(dist, n_taxa)

    # Step 3: Convert to adjacency
    adj = _edges_to_adjacency(edges, num_nodes, labels)

    # Step 4: NNI refinement under ML
    adj = _nni_search(adj, num_nodes, n_taxa, labels, sequences, max_iterations=20)

    # Step 5: Convert to Newick
    newick = _adjacency_to_newick(adj, num_nodes, n_taxa, labels)

    return newick
# EVOLVE-BLOCK-END
