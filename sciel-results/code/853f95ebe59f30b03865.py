<<<<<<< SEARCH
"""Reconstruct phylogenetic topology using Neighbor Joining."""
import numpy as np
import time
import math

def _jc_distance_matrix(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    seq_data = np.array([[ord(c) for c in sequences[label]] for label in labels])
    diff = np.sum(seq_data[:, np.newaxis, :] != seq_data[np.newaxis, :, :], axis=2)
    p = diff / float(len(sequences[labels[0]]))
    d = -0.75 * np.log(1 - (4.0/3.0) * p)
    d[p >= 0.75] = 3.0
    np.fill_diagonal(d, 0.0)
    return d, labels

def _nj_newick(d, labels):
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    if n == 3:
        return "(" + labels[0] + "," + labels[1] + "," + labels[2] + ");"

    dist = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[(i, j)] = d[i, j]
                
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    
    next_internal_id = -1
    
    while len(active) > 3:
        n_act = len(active)
        
        r = {i: 0.0 for i in active}
        for i in active:
            for j in active:
                if i != j:
                    r[i] += dist.get((i, j), 0.0)
                    
        best_c = float('inf')
        best_i, best_j = -1, -1
        for i in active:
            for j in active:
                if i == j: continue
                c = (n_act - 2) * dist.get((i, j), 0.0) - r[i] - r[j]
                if c < best_c:
                    best_c = c
                    best_i, best_j = i, j
                    
        i, j = best_i, best_j
        d_ij = dist.get((i, j), 0.0)
        
        a = next_internal_id
        next_internal_id -= 1
        
        for k in active:
            if k == i or k == j: continue
            d_ik = dist.get((i, k), 0.0)
            d_jk = dist.get((j, k), 0.0)
            d_ak = 0.5 * (d_ik + d_jk - d_ij)
            dist[(a, k)] = d_ak
            dist[(k, a)] = d_ak
            
        trees[a] = "(" + trees[i] + "," + trees[j] + ")"
        
        active.remove(i)
        active.remove(j)
        active.append(a)
        
    a, b, c = active
    return "(" + trees[a] + "," + trees[b] + "," + trees[c] + ");"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    d, _ = _jc_distance_matrix(sequences)
    return _nj_newick(d, labels)
=======
"""Reconstruct phylogenetic topology using Neighbor Joining with SPR+ME refinement."""
import numpy as np
import time
import math

_START = time.time()
_DEADLINE = _START + 250.0  # safety margin before 270s kill

def _jc_distance_matrix(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    seq_data = np.array([[ord(c) for c in sequences[label]] for label in labels])
    diff = np.sum(seq_data[:, np.newaxis, :] != seq_data[np.newaxis, :, :], axis=2)
    p = diff / float(len(sequences[labels[0]]))
    d = -0.75 * np.log(np.maximum(1.0 - (4.0/3.0) * p, 1e-10))
    d[p >= 0.75] = 3.0
    np.fill_diagonal(d, 0.0)
    return d, labels

def _nj_tree(d):
    """Vectorized NJ. Returns (adj, node_labels) where adj is (2n-2)x(2n-2) adjacency,
    node_labels maps node indices to label strings."""
    n = len(d)
    if n == 1:
        return np.zeros((1,1), dtype=np.float64), [d_labels[0]] if d_labels else []
    if n == 2:
        adj = np.array([[0.0, d[0,1]], [d[0,1], 0.0]])
        return adj, [d_labels[0], d_labels[1]] if d_labels else []
    if n == 3:
        adj = np.array([[0.0, d[0,1], d[0,2]],
                        [d[1,0], 0.0, d[1,2]],
                        [d[2,0], d[2,1], 0.0]])
        return adj, [d_labels[0], d_labels[1], d_labels[2]] if d_labels else []
    
    # d_labels will be set by caller; here we work with indices
    dist = d.astype(np.float64).copy()
    active = list(range(n))
    next_id = n
    # Track tree structure for newick: parent pointers
    parent = [None] * (2 * n)  # max 2n-2 nodes
    # We'll build newick strings directly
    node_str = [None] * (2 * n)
    for i in range(n):
        node_str[i] = str(i)  # placeholder, replaced later
    
    while len(active) > 3:
        na = len(active)
        # r[i] = sum of distances from i to all active
        r = np.array([dist[i, active].sum() for i in active])
        # Q matrix for active nodes
        Q = (na - 2) * dist[np.ix_(active, active)] - r[:, None] - r[None, :]
        # Find min Q (off-diagonal)
        np.fill_diagonal(Q, np.inf)
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i_idx, j_idx = min_idx
        i, j = active[i_idx], active[j_idx]
        d_ij = dist[i, j]
        
        a = next_id
        next_id += 1
        
        # Update distances
        for k_idx, k in enumerate(active):
            if k == i or k == j:
                continue
            d_ak = 0.5 * (dist[i, k] + dist[j, k] - d_ij)
            dist[a, k] = d_ak
            dist[k, a] = d_ak
        
        # Record edge lengths
        x_i = 0.5 * (d_ij + (r[i] - r[j]) / (na - 2))
        x_j = d_ij - x_i
        
        # Store for newick
        node_str[a] = None  # will be set
        # We need to track the tree structure
        # Let's use a different approach: store the full adjacency at the end
        # For now, just track which nodes were merged
        parent[i] = a
        parent[j] = a
        # Store edge lengths
        if not hasattr(_nj_tree, '_edge_lengths'):
            _nj_tree._edge_lengths = {}
        _nj_tree._edge_lengths[(i, a)] = x_i
        _nj_tree._edge_lengths[(j, a)] = x_j
        
        active.remove(i)
        active.remove(j)
        active.append(a)
    
    a, b, c = active
    # Final three: connect them
    d_ab = dist[a, b]
    d_ac = dist[a, c]
    d_bc = dist[b, c]
    # Star tree: find center
    # For 3 nodes, the unrooted tree is a star with 3 edges
    # Edge lengths:
    e_a = 0.5 * (d_ab + d_ac - d_bc)
    e_b = 0.5 * (d_ab + d_bc - d_ac)
    e_c = 0.5 * (d_ac + d_bc - d_ab)
    
    # Build adjacency matrix
    total_nodes = next_id
    adj = np.zeros((total_nodes, total_nodes), dtype=np.float64)
    
    # Reconstruct edges from parent pointers
    for node in range(total_nodes):
        if parent[node] is not None:
            p = parent[node]
            el = _nj_tree._edge_lengths.get((node, p), 0.0)
            adj[node, p] = el
            adj[p, node] = el
    
    # Add final star edges
    # a, b, c are connected via a central node? No, for 3 leaves it's a star
    # Actually for 3 nodes in unrooted tree, it's just 3 edges forming a triangle
    # But that's not a tree. The correct unrooted tree for 3 taxa is a star with one internal node
    # Wait, we already have nodes a, b, c as the remaining 3. They should be connected
    # In NJ, the last 3 are connected as a star with one internal node
    # But we don't have that internal node. Let's add it.
    center = next_id
    next_id += 1
    total_nodes = next_id
    adj = np.zeros((total_nodes, total_nodes), dtype=np.float64)
    
    # Rebuild all edges
    for node in range(total_nodes):
        if node < len(parent) and parent[node] is not None:
            p = parent[node]
            el = _nj_tree._edge_lengths.get((node, p), 0.0)
            adj[node, p] = el
            adj[p, node] = el
    
    adj[a, center] = e_a
    adj[center, a] = e_a
    adj[b, center] = e_b
    adj[center, b] = e_b
    adj[c, center] = e_c
    adj[center, c] = e_e if False else e_c  # e_c already set
    
    # Build node labels
    node_labels = [None] * total_nodes
    for i in range(n):
        node_labels[i] = str(i)  # Will be replaced by actual labels
    for i in range(n, total_nodes):
        node_labels[i] = "n" + str(i)
    
    return adj, node_labels, parent, _nj_tree._edge_lengths

def _nj_tree_v2(d, labels):
    """Vectorized NJ returning adjacency matrix and label mapping."""
    n = len(d)
    if n <= 1:
        adj = np.zeros((max(n,1), max(n,1)))
        return adj, [labels[0]] if n == 1 else []
    
    dist = d.astype(np.float64).copy()
    active = list(range(n))
    next_id = n
    
    # Track tree structure
    edges = []  # list of (node1, node2, length)
    
    while len(active) > 3:
        na = len(active)
        r = np.array([dist[i, active].sum() for i in active])
        Q = (na - 2) * dist[np.ix_(active, active)] - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i_idx, j_idx = min_idx
        i, j = active[i_idx], active[j_idx]
        d_ij = dist[i, j]
        
        a = next_id
        next_id += 1
        
        for k in active:
            if k == i or k == j:
                continue
            d_ak = 0.5 * (dist[i, k] + dist[j, k] - d_ij)
            dist[a, k] = d_ak
            dist[k, a] = d_ak
        
        x_i = 0.5 * (d_ij + (r[i] - r[j]) / (na - 2))
        x_j = d_ij - x_i
        
        edges.append((i, a, max(0.0, x_i)))
        edges.append((j, a, max(0.0, x_j)))
        
        active.remove(i)
        active.remove(j)
        active.append(a)
    
    a, b, c = active
    d_ab = dist[a, b]
    d_ac = dist[a, c]
    d_bc = dist[b, c]
    e_a = max(0.0, 0.5 * (d_ab + d_ac - d_bc))
    e_b = max(0.0, 0.5 * (d_ab + d_bc - d_ac))
    e_c = max(0.0, 0.5 * (d_ac + d_bc - d_ab))
    
    center = next_id
    next_id += 1
    total_nodes = next_id
    
    edges.append((a, center, e_a))
    edges.append((b, center, e_b))
    edges.append((c, center, e_c))
    
    adj = np.zeros((total_nodes, total_nodes), dtype=np.float64)
    for u, v, w in edges:
        adj[u, v] = w
        adj[v, u] = w
    
    node_labels = list(labels) + ["i" + str(i) for i in range(n, total_nodes)]
    
    return adj, node_labels

def _me_score(adj, D):
    """Compute Minimum Evolution score: sum over all pairs of (path length - D_ij)."""
    n = adj.shape[0]
    # Compute all-pairs shortest paths (tree, so unique path)
    # Use BFS/DFS from each node
    score = 0.0
    for i in range(n):
        # BFS from i
        visited = np.zeros(n, dtype=bool)
        dist_from_i = np.full(n, np.inf)
        dist_from_i[i] = 0.0
        stack = [i]
        while stack:
            u = stack.pop()
            for v in range(n):
                if adj[u, v] > 0 and not visited[v]:
                    visited[v] = True
                    dist_from_i[v] = dist_from_i[u] + adj[u, v]
                    stack.append(v)
        for j in range(i+1, n):
            if np.isfinite(dist_from_i[j]):
                score += dist_from_i[j] - D[i, j]
    return score

def _me_score_fast(adj, D):
    """Faster ME score using matrix operations. For a tree, path(i,j) can be computed
    via LCA or by summing edges. We use the fact that for a tree, we can compute
    all-pairs distances via repeated Dijkstra (which is just BFS on tree)."""
    n = adj.shape[0]
    # Build adjacency list
    adj_list = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 1e-10:
                adj_list[i].append((j, adj[i, j]))
    
    score = 0.0
    for i in range(n):
        # BFS
        dist_arr = np.full(n, np.inf)
        dist_arr[i] = 0.0
        stack = [i]
        while stack:
            u = stack.pop()
            for v, w in adj_list[u]:
                if dist_arr[v] == np.inf:
                    dist_arr[v] = dist_arr[u] + w
                    stack.append(v)
        for j in range(i+1, n):
            if np.isfinite(dist_arr[j]):
                score += dist_arr[j] - D[i, j]
    return score

def _spr_refine(adj, D, labels, max_iter=10):
    """SPR local search with ME scoring. Returns improved adjacency and labels."""
    n = adj.shape[0]
    # Build adjacency list
    def get_adj_list(a):
        al = [[] for _ in range(len(a))]
        for i in range(len(a)):
            for j in range(len(a)):
                if a[i, j] > 1e-10:
                    al[i].append((j, a[i, j]))
        return al
    
    adj_list = get_adj_list(adj)
    best_score = _me_score_fast(adj, D)
    best_adj = adj.copy()
    
    for iteration in range(max_iter):
        if time.time() > _DEADLINE - 30:
            break
        improved = False
        n_nodes = len(adj_list)
        
        for i in range(n_nodes):
            if time.time() > _DEADLINE - 30:
                break
            # For each edge incident to i, consider pruning the subtree
            for j, w_ij in adj_list[i]:
                # Prune edge (i, j): the subtree rooted at j (away from i)
                # Find all nodes in j's subtree (not containing i)
                subtree = set()
                stack = [j]
                parent_map = {j: None}
                while stack:
                    u = stack.pop()
                    subtree.add(u)
                    for v, _ in adj_list[u]:
                        if v != parent_map[u]:
                            parent_map[v] = u
                            stack.append(v)
                
                if len(subtree) < 2:
                    continue
                if len(subtree) == n_nodes - 1:
                    # Pruning everything, skip
                    continue
                
                # For each possible reattachment edge
                for k in range(n_nodes):
                    if k in subtree:
                        continue
                    # Reattach subtree root j to k
                    # New edge: j-k, remove i-j
                    # Need to check if k is adjacent to something in subtree (would create cycle)
                    # Actually, just try it
                    new_adj = best_adj.copy()
                    # Remove edge i-j
                    new_adj[i, j] = 0
                    new_adj[j, i] = 0
                    # Add edge j-k
                    new_adj[j, k] = w_ij  # Keep same length for now
                    new_adj[k, j] = w_ij
                    
                    # Check validity: should be a tree (n-1 edges, connected)
                    # Quick check: count edges
                    edge_count = np.sum(new_adj > 1e-10) // 2
                    if edge_count != n_nodes - 1:
                        continue
                    
                    # Check connectivity
                    visited = np.zeros(n_nodes, dtype=bool)
                    stack = [0]
                    visited[0] = True
                    while stack:
                        u = stack.pop()
                        for v in range(n_nodes):
                            if new_adj[u, v] > 1e-10 and not visited[v]:
                                visited[v] = True
                                stack.append(v)
                    if not np.all(visited):
                        continue
                    
                    # Compute ME score
                    new_score = _me_score_fast(new_adj, D)
                    if new_score < best_score - 1e-8:
                        best_score = new_score
                        best_adj = new_adj
                        adj_list = get_adj_list(best_adj)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_adj, best_score

def _adj_to_newick(adj, labels):
    """Convert adjacency matrix to Newick string."""
    n = adj.shape[0]
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    
    # Find a leaf to start
    adj_list = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 1e-10:
                adj_list[i].append((j, adj[i, j]))
    
    # Find a leaf (degree 1)
    start = None
    for i in range(n):
        if len(adj_list[i]) == 1:
            start = i
            break
    if start is None:
        start = 0
    
    # DFS to build newick
    visited = set()
    
    def build_newick(node, parent):
        visited.add(node)
        children = [(v, w) for v, w in adj_list[node] if v != parent]
        if not children:
            # Leaf
            return labels[node]
        else:
            parts = []
            for v, w in children:
                child_str = build_newick(v, node)
                parts.append(child_str + ":" + f"{w:.4f}")
            return "(" + ",".join(parts) + ")"
    
    result = build_newick(start, -1)
    return result + ";"

def reconstruct(sequences):
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 250.0
    
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    d, _ = _jc_distance_matrix(sequences)
    n = len(labels)
    
    if n <= 3:
        if n == 1:
            return labels[0] + ";"
        elif n == 2:
            return "(" + labels[0] + "," + labels[1] + ");"
        else:
            return "(" + labels[0] + "," + labels[1] + "," + labels[2] + ");"
    
    # NJ
    adj, node_labels = _nj_tree_v2(d, labels)
    
    # SPR refinement
    try:
        adj, _ = _spr_refine(adj, d, labels, max_iter=5)
    except Exception:
        pass
    
    if time.time() > _DEADLINE - 10:
        # Not enough time, return NJ tree
        pass
    
    newick = _adj_to_newick(adj, node_labels)
    
    # Verify all labels present
    for lab in labels:
        if lab not in newick:
            # Fallback
            newick = _adj_to_newick(_nj_tree_v2(d, labels)[0], _nj_tree_v2(d, labels)[1])
            break
    
    return newick
>>>>>>> REPLACE

I made a mistake - I referenced `d_labels` in `_nj_tree` which doesn't exist, and the code is overly complex with unused functions. Let me provide a cleaner, working version:

