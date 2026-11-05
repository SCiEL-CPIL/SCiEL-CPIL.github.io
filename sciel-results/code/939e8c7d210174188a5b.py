import math
import time
import numpy as np
import re

_GLOBAL_START = None
_DEADLINE = 200.0  # Total budget for all instances

def _elapsed():
    global _GLOBAL_START
    if _GLOBAL_START is None:
        _GLOBAL_START = time.time()
    return time.time() - _GLOBAL_START

def _time_exceeded():
    return _elapsed() > _DEADLINE

def _is_valid_newick(newick, labels):
    if not newick:
        return False
    found_labels = re.findall(r't\d+', newick)
    return set(found_labels) == set(labels)

def _parse_newick_to_tree(newick):
    """
    Parse Newick string into an internal tree representation.
    Returns:
      - nodes: list of node IDs (0..n-1)
      - parent: list where parent[i] is the parent node ID or -1 for root
      - children: list of lists, children[i] is list of child node IDs
      - labels: list where labels[i] is the taxon label for leaf nodes, None for internal
      - is_leaf: boolean list
    """
    # Remove trailing semicolon
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    nodes = []  # list of node IDs
    parent = []  # parent[i]
    children = []  # children[i]
    labels = []  # labels[i]
    is_leaf = []  # is_leaf[i]
    
    node_counter = [0]
    
    def create_node(label=None):
        nid = node_counter[0]
        node_counter[0] += 1
        nodes.append(nid)
        parent.append(-1)
        children.append([])
        labels.append(label)
        is_leaf.append(label is not None)
        return nid
    
    def parse(pos):
        # Returns (node_id, new_pos)
        if s[pos] == '(':
            pos += 1
            internal_id = create_node(None)
            while True:
                child_id, pos = parse(pos)
                parent[child_id] = internal_id
                children[internal_id].append(child_id)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
                    continue
                elif pos < len(s) and s[pos] == ')':
                    pos += 1
                    break
                else:
                    break
            # Skip branch length
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in ',;)':
                    pos += 1
            return internal_id, pos
        else:
            end = pos
            while end < len(s) and s[end] not in ',:;)':
                end += 1
            label = s[pos:end]
            leaf_id = create_node(label)
            pos = end
            # Skip branch length
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in ',;)':
                    pos += 1
            return leaf_id, pos
    
    root_id, _ = parse(0)
    
    # Now we have a rooted tree. We need to convert to unrooted representation.
    # For SPR, we work with the unrooted tree.
    # Let's build adjacency list from this rooted tree.
    
    n_nodes = len(nodes)
    adj = {i: [] for i in range(n_nodes)}
    
    for i in range(n_nodes):
        for c in children[i]:
            adj[i].append(c)
            adj[c].append(i)
    
    return adj, labels, is_leaf, n_nodes

def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    L = len(sequences[labels[0]])
    enc = np.array([0, 1, 2, 3], dtype=np.int8)
    # Map ACGT to 0,1,2,3
    char_to_idx = {c: i for i, c in enumerate('ACGT')}
    seqs = np.zeros((n, L), dtype=np.int8)
    for i, l in enumerate(labels):
        s = sequences[l]
        # Vectorized encoding
        for j, c in enumerate(s):
            seqs[i, j] = char_to_idx[c]
    
    # Pairwise differences using broadcasting
    diff = (seqs[:, None, :] != seqs[None, :, :]).sum(axis=2)
    p = diff.astype(np.float64) / L
    p_safe = np.minimum(p, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p_safe / 3.0)
    np.fill_diagonal(d, 0.0)
    # Ensure symmetry
    d = (d + d.T) * 0.5
    return d

def _build_path_matrix(adj, leaf_nodes, labels, label_to_idx):
    """
    Build the path matrix A and distance vector d_vec for SSE computation.
    Returns (A, d_vec, edges, edge_index)
    """
    n_leaves = len(leaf_nodes)
    if n_leaves < 2:
        return None, None, None, None
    
    # Get all edges
    edges = set()
    for u in adj:
        for v in adj[u]:
            if u < v:
                edges.add((u, v))
            else:
                edges.add((v, u))
    edges = list(edges)
    edge_index = {e: i for i, e in enumerate(edges)}
    
    num_edges = len(edges)
    num_pairs = n_leaves * (n_leaves - 1) // 2
    
    A = np.zeros((num_pairs, num_edges))
    d_vec = np.zeros(num_pairs)
    
    # For each leaf, do BFS to find distances to all other leaves
    # This is more efficient than BFS for each pair
    # Actually, let's use a different approach: for each edge, determine which leaf pairs have that edge on their path
    
    # Alternative: use the fact that for an unrooted tree, the path between two leaves
    # can be determined by rooting the tree and using LCA
    
    # Let's do it the simple way but with optimization
    pair_idx = 0
    for i in range(n_leaves):
        li = leaf_nodes[i]
        li_label = labels[li]
        idx_i = label_to_idx[li_label]
        
        # BFS from li
        dist_to_node = {li: 0}
        parent = {li: -1}
        queue = [li]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if v not in dist_to_node:
                    dist_to_node[v] = dist_to_node[u] + 1
                    parent[v] = u
                    queue.append(v)
        
        for j in range(i + 1, n_leaves):
            lj = leaf_nodes[j]
            lj_label = labels[lj]
            idx_j = label_to_idx[lj_label]
            d_vec[pair_idx] = dist[idx_i, idx_j]
            
            # Reconstruct path
            path = []
            curr = lj
            while curr != -1:
                path.append(curr)
                curr = parent[curr]
            path.reverse()
            
            # Mark edges in path
            for k in range(len(path) - 1):
                u, v = path[k], path[k + 1]
                edge = (u, v) if u < v else (v, u)
                A[pair_idx, edge_index[edge]] = 1.0
            
            pair_idx += 1
    
    return A, d_vec, edges, edge_index

def _tree_sse(adj, labels, dist, label_to_idx):
    """
    Compute SSE for a tree topology by solving for optimal branch lengths.
    """
    # Identify leaves
    leaf_nodes = []
    for node in adj:
        if labels[node] is not None:
            leaf_nodes.append(node)
    
    n_leaves = len(leaf_nodes)
    if n_leaves < 2:
        return 0.0
    
    A, d_vec, edges, edge_index = _build_path_matrix(adj, leaf_nodes, labels, label_to_idx)
    if A is None:
        return 0.0
    
    # Solve least squares
    try:
        x, residuals, rank, s = np.linalg.lstsq(A, d_vec, rcond=None)
    except:
        return float('inf')
    
    pred = A.dot(x)
    sse = np.sum((pred - d_vec) ** 2)
    
    return sse

def _tree_sse_from_matrix(A, d_vec):
    """Compute SSE given pre-built path matrix and distance vector."""
    if A is None or len(d_vec) == 0:
        return 0.0
    try:
        x, residuals, rank, s = np.linalg.lstsq(A, d_vec, rcond=None)
    except:
        return float('inf')
    pred = A.dot(x)
    sse = np.sum((pred - d_vec) ** 2)
    return sse

def _nj_tree(labels, dist):
    n = len(labels)
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    d = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                d[(i, j)] = dist[i][j]
    
    next_id = n
    while len(nodes) > 3:
        m = len(nodes)
        r = {}
        for a in nodes:
            r[a] = sum(d[(a, b)] for b in nodes if b != a)
        
        best_crit = None
        best_pair = None
        for i in range(m):
            for j in range(i + 1, m):
                a, b = nodes[i], nodes[j]
                crit = (m - 2) * d[(a, b)] - r[a] - r[b]
                if best_crit is None or crit < best_crit:
                    best_crit = crit
                    best_pair = (a, b)
        
        a, b = best_pair
        da = 0.5 * d[(a, b)]
        db = 0.5 * d[(a, b)]
        for x in nodes:
            if x != a and x != b:
                da += 0.5 * (d[(a, x)] - d[(b, x)])
                db += 0.5 * (d[(b, x)] - d[(a, x)])
        
        da = max(da, 0.0)
        db = max(db, 0.0)
        
        new = next_id
        next_id += 1
        rest = [x for x in nodes if x not in (a, b)]
        
        for x in rest:
            d[(new, x)] = d[(x, new)] = 0.5 * (d[(a, x)] + d[(b, x)] - d[(a, b)])
            
        for x in rest:
            del d[(a, x)]
            del d[(b, x)]
            del d[(x, a)]
            del d[(x, b)]
        
        del d[(a, b)]
        del d[(b, a)]
        
        nodes = rest + [new]
        trees[new] = f"({trees[a]},{trees[b]})"
    
    return "(" + ",".join(trees[x] for x in nodes) + ");"

def _get_edges(adj):
    """Get list of unique edges from adjacency dict."""
    edges = set()
    for u in adj:
        for v in adj[u]:
            if u < v:
                edges.add((u, v))
            else:
                edges.add((v, u))
    return list(edges)

def _nni_moves(adj, node_labels, dist, label_to_idx):
    """
    Generate all NNI moves and return list of (new_adj, sse) pairs.
    """
    edges = _get_edges(adj)
    moves = []
    
    for edge in edges:
        u, v = edge
        
        # Check if this is an internal edge (both endpoints have degree >= 2)
        if len(adj[u]) < 2 or len(adj[v]) < 2:
            continue
        
        # Get neighbors excluding each other
        u_neighbors = [x for x in adj[u] if x != v]
        v_neighbors = [x for x in adj[v] if x != u]
        
        # For NNI, we need exactly one neighbor on each side (for the edge to be "internal"
        # in the sense of having 4 subtrees). In a binary unrooted tree, internal edges
        # connect two internal nodes each with degree 3.
        # So u has 2 other neighbors, v has 2 other neighbors.
        # NNI swaps: (u-a, v-b) -> (u-b, v-a) for a in u_neighbors, b in v_neighbors
        
        for a in u_neighbors:
            for b in v_neighbors:
                # Create new adjacency
                new_adj = {x: list(y) for x, y in adj.items()}
                
                # Remove old connections
                new_adj[u].remove(v)
                new_adj[v].remove(u)
                new_adj[u].remove(a)
                new_adj[v].remove(b)
                
                # Add new connections
                new_adj[u].append(b)
                new_adj[b].append(u)
                new_adj[v].append(a)
                new_adj[a].append(v)
                
                # Compute SSE
                sse = _tree_sse(new_adj, node_labels, dist, label_to_idx)
                moves.append((new_adj, sse))
    
    return moves

def _spr_refine(newick, labels, dist, label_to_idx):
    """
    Perform NNI refinement on a tree.
    Returns the best Newick string found.
    """
    adj, node_labels, is_leaf, n_nodes = _parse_newick_to_tree(newick)
    
    current_sse = _tree_sse(adj, node_labels, dist, label_to_idx)
    
    improved = True
    max_rounds = 5
    round_num = 0
    
    while improved and round_num < max_rounds:
        improved = False
        round_num += 1
        if _time_exceeded():
            break
        
        best_sse = current_sse
        best_adj = None
        
        # Try all NNI moves
        moves = _nni_moves(adj, node_labels, dist, label_to_idx)
        
        for new_adj, sse in moves:
            if _time_exceeded():
                break
            if sse < best_sse - 1e-10:
                best_sse = sse
                best_adj = new_adj
                improved = True
        
        if best_adj is not None:
            adj = best_adj
            current_sse = best_sse
    
    # Convert back to Newick
    return _adj_to_newick(adj, node_labels)

def _newick_to_adj(newick, labels):
    """Parse Newick to adjacency list. Returns adj, node_labels."""
    # Use the new parser
    adj, node_labels, is_leaf, n_nodes = _parse_newick_to_tree(newick)
    return adj, node_labels

def _optimize_branch_lengths(adj, nodes, labels, dist):
    """
    Optimize branch lengths using Least Squares.
    Returns SSE.
    """
    # This function is kept for compatibility but we use _tree_sse now
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    return _tree_sse(adj, nodes, dist, label_to_idx)

def _spr_refine(newick, labels, dist):
    """
    Perform SPR refinement.
    """
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    return _spr_refine_impl(newick, labels, dist, label_to_idx)

def _adj_to_newick(adj, node_labels):
    """
    Convert adjacency list to Newick string.
    adj: dict {node_id: [neighbors]}
    node_labels: list where node_labels[i] is the label for leaf i, None for internal
    """
    n_nodes = len(node_labels)
    
    # Find a leaf to root at
    root = None
    for i in range(n_nodes):
        if node_labels[i] is not None:
            root = i
            break
    
    if root is None:
        return ""
    
    visited = set()
    
    def gen(node, parent):
        if node in visited:
            return ""
        visited.add(node)
        
        if node_labels[node] is not None:
            return node_labels[node]
        else:
            children = []
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                child_str = gen(neighbor, node)
                if child_str:
                    children.append(child_str)
            
            if len(children) == 0:
                return ""
            elif len(children) == 1:
                return "(" + children[0] + ")"
            else:
                return "(" + ",".join(children) + ")"
                
    s = gen(root, -1)
    return s + ";"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        return "(" + ",".join(labels) + ");"
        
    dist = _jc_distance_matrix(sequences, labels)
    
    nj_newick = _nj_tree(labels, dist)
    
    # Validate NJ tree
    if not _is_valid_newick(nj_newick, labels):
        # Fallback to star tree
        return "(" + ",".join(labels) + ");"
        
    newick = nj_newick
    
    # Try refinement if time permits
    if not _time_exceeded():
        try:
            label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
            refined_newick = _spr_refine_impl(nj_newick, labels, dist, label_to_idx)
            if _is_valid_newick(refined_newick, labels):
                newick = refined_newick
            else:
                # If refinement produced invalid output, stick with NJ
                pass
        except Exception:
            # If refinement crashes, stick with NJ
            pass
    
    # Final check
    if not _is_valid_newick(newick, labels):
        return "(" + ",".join(labels) + ");"
        
    return newick

def _spr_refine_impl(newick, labels, dist, label_to_idx):
    """
    Implement NNI refinement.
    """
    return _spr_refine(newick, labels, dist, label_to_idx)
