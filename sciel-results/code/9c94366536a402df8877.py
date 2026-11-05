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
      - adj: dict {node_id: [neighbors]}
      - labels: list where labels[i] is the taxon label for leaf nodes, None for internal
      - is_leaf: boolean list
      - n_nodes: number of nodes
    """
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    parent = []
    children = []
    labels = []
    is_leaf = []
    
    node_counter = [0]
    
    def create_node(label=None):
        nid = node_counter[0]
        node_counter[0] += 1
        parent.append(-1)
        children.append([])
        labels.append(label)
        is_leaf.append(label is not None)
        return nid
    
    def parse(pos):
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
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in ',;)':
                    pos += 1
            return leaf_id, pos
    
    root_id, _ = parse(0)
    
    n_nodes = len(labels)
    adj = {i: [] for i in range(n_nodes)}
    
    for i in range(n_nodes):
        for c in children[i]:
            adj[i].append(c)
            adj[c].append(i)
    
    return adj, labels, is_leaf, n_nodes

def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    L = len(sequences[labels[0]])
    char_to_idx = {c: i for i, c in enumerate('ACGT')}
    seqs = np.empty((n, L), dtype=np.int8)
    for i, l in enumerate(labels):
        s = sequences[l]
        seqs[i] = np.array([char_to_idx[c] for c in s], dtype=np.int8)
    
    diff = (seqs[:, None, :] != seqs[None, :, :]).sum(axis=2)
    p = diff.astype(np.float64) / L
    p_safe = np.minimum(p, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p_safe / 3.0)
    np.fill_diagonal(d, 0.0)
    d = (d + d.T) * 0.5
    return d

def _build_path_matrix(adj, leaf_nodes, labels, label_to_idx, dist):
    """
    Build the path matrix A and distance vector d_vec for SSE computation.
    Returns (A, d_vec, edges, edge_index)
    """
    n_leaves = len(leaf_nodes)
    if n_leaves < 2:
        return None, None, None, None
    
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
    
    root = None
    for node in adj:
        if len(adj[node]) >= 3 and labels[node] is None:
            root = node
            break
    if root is None:
        for node in adj:
            if labels[node] is None:
                root = node
                break
        if root is None:
            root = leaf_nodes[0]
    
    parent = {root: -1}
    depth = {root: 0}
    order = [root]
    visited = {root}
    for u in order:
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                parent[v] = u
                depth[v] = depth[u] + 1
                order.append(v)
    
    def lca(u, v):
        while depth[u] > depth[v]:
            u = parent[u]
        while depth[v] > depth[u]:
            v = parent[v]
        while u != v:
            u = parent[u]
            v = parent[v]
        return u
    
    pair_idx = 0
    for i in range(n_leaves):
        li = leaf_nodes[i]
        idx_i = label_to_idx[labels[li]]
        
        for j in range(i + 1, n_leaves):
            lj = leaf_nodes[j]
            idx_j = label_to_idx[labels[lj]]
            d_vec[pair_idx] = dist[idx_i, idx_j]
            
            w = lca(li, lj)
            
            path_li = []
            curr = li
            while curr != w:
                path_li.append(curr)
                curr = parent[curr]
            path_li.append(w)
            
            path_lj = []
            curr = lj
            while curr != w:
                path_lj.append(curr)
                curr = parent[curr]
            
            full_path = path_li + list(reversed(path_lj))
            
            for k in range(len(full_path) - 1):
                u, v = full_path[k], full_path[k + 1]
                edge = (u, v) if u < v else (v, u)
                A[pair_idx, edge_index[edge]] = 1.0
            
            pair_idx += 1
    
    return A, d_vec, edges, edge_index

def _tree_sse(adj, labels, dist, label_to_idx):
    """
    Compute SSE for a tree topology by solving for optimal branch lengths.
    """
    leaf_nodes = []
    for node in adj:
        if labels[node] is not None:
            leaf_nodes.append(node)
    
    n_leaves = len(leaf_nodes)
    if n_leaves < 2:
        return 0.0
    
    edges = _get_edges(adj)
    num_edges = len(edges)
    if num_edges == 0:
        return 0.0
    
    edge_index = {e: i for i, e in enumerate(edges)}
    
    root = None
    for node in adj:
        if len(adj[node]) >= 3 and labels[node] is None:
            root = node
            break
    if root is None:
        for node in adj:
            if labels[node] is None:
                root = node
                break
        if root is None:
            root = leaf_nodes[0]
    
    parent = {root: -1}
    depth = {root: 0}
    order = [root]
    visited = {root}
    for u in order:
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                parent[v] = u
                depth[v] = depth[u] + 1
                order.append(v)
    
    def lca(u, v):
        while depth[u] > depth[v]:
            u = parent[u]
        while depth[v] > depth[u]:
            v = parent[v]
        while u != v:
            u = parent[u]
            v = parent[v]
        return u
    
    num_pairs = n_leaves * (n_leaves - 1) // 2
    A = np.zeros((num_pairs, num_edges))
    d_vec = np.zeros(num_pairs)
    
    pair_idx = 0
    for i in range(n_leaves):
        li = leaf_nodes[i]
        idx_i = label_to_idx[labels[li]]
        
        for j in range(i + 1, n_leaves):
            lj = leaf_nodes[j]
            idx_j = label_to_idx[labels[lj]]
            d_vec[pair_idx] = dist[idx_i, idx_j]
            
            w = lca(li, lj)
            
            path_li = []
            curr = li
            while curr != w:
                path_li.append(curr)
                curr = parent[curr]
            path_li.append(w)
            
            path_lj = []
            curr = lj
            while curr != w:
                path_lj.append(curr)
                curr = parent[curr]
            
            full_path = path_li + list(reversed(path_lj))
            
            for k in range(len(full_path) - 1):
                u, v = full_path[k], full_path[k + 1]
                edge = (u, v) if u < v else (v, u)
                A[pair_idx, edge_index[edge]] = 1.0
            
            pair_idx += 1
    
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
    Only considers valid NNI moves where both endpoints have degree exactly 3.
    """
    edges = _get_edges(adj)
    moves = []
    
    for edge in edges:
        u, v = edge
        
        if len(adj[u]) != 3 or len(adj[v]) != 3:
            continue
        
        u_neighbors = [x for x in adj[u] if x != v]
        v_neighbors = [x for x in adj[v] if x != u]
        
        a1, a2 = u_neighbors[0], u_neighbors[1]
        b1, b2 = v_neighbors[0], v_neighbors[1]
        
        # Move 1: (u-a1, v-b1) -> (u-b1, v-a1)
        new_adj1 = {x: list(y) for x, y in adj.items()}
        new_adj1[u].remove(v)
        new_adj1[v].remove(u)
        new_adj1[u].remove(a1)
        new_adj1[v].remove(b1)
        new_adj1[u].append(b1)
        new_adj1[b1].append(u)
        new_adj1[v].append(a1)
        new_adj1[a1].append(v)
        sse1 = _tree_sse(new_adj1, node_labels, dist, label_to_idx)
        moves.append((new_adj1, sse1))
        
        # Move 2: (u-a1, v-b2) -> (u-b2, v-a1)
        new_adj2 = {x: list(y) for x, y in adj.items()}
        new_adj2[u].remove(v)
        new_adj2[v].remove(u)
        new_adj2[u].remove(a1)
        new_adj2[v].remove(b2)
        new_adj2[u].append(b2)
        new_adj2[b2].append(u)
        new_adj2[v].append(a1)
        new_adj2[a1].append(v)
        sse2 = _tree_sse(new_adj2, node_labels, dist, label_to_idx)
        moves.append((new_adj2, sse2))
    
    return moves

def _spr_refine(newick, labels, dist, label_to_idx):
    """
    Perform NNI refinement on a tree.
    Returns the best Newick string found.
    """
    adj, node_labels, is_leaf, n_nodes = _parse_newick_to_tree(newick)
    
    current_sse = _tree_sse(adj, node_labels, dist, label_to_idx)
    
    improved = True
    max_rounds = 20
    round_num = 0
    
    while improved and round_num < max_rounds:
        improved = False
        round_num += 1
        if _time_exceeded():
            break
        
        best_sse = current_sse
        best_adj = None
        
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
    
    return _adj_to_newick(adj, node_labels)

def _newick_to_adj(newick, labels):
    """Parse Newick to adjacency list. Returns adj, node_labels."""
    adj, node_labels, is_leaf, n_nodes = _parse_newick_to_tree(newick)
    return adj, node_labels

def _optimize_branch_lengths(adj, nodes, labels, dist):
    """
    Optimize branch lengths using Least Squares.
    Returns SSE.
    """
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
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    
    nj_newick = _nj_tree(labels, dist)
    
    if not _is_valid_newick(nj_newick, labels):
        return "(" + ",".join(labels) + ");"
    
    newick = nj_newick
    
    if not _time_exceeded():
        try:
            refined_newick = _spr_refine_impl(nj_newick, labels, dist, label_to_idx)
            if _is_valid_newick(refined_newick, labels):
                newick = refined_newick
        except Exception:
            pass
    
    if not _is_valid_newick(newick, labels):
        return "(" + ",".join(labels) + ");"
        
    return newick

def _spr_refine_impl(newick, labels, dist, label_to_idx):
    """
    Implement NNI refinement with multiple random restarts.
    """
    best_newick = newick
    best_sse = float('inf')
    
    adj, node_labels, is_leaf, n_nodes = _parse_newick_to_tree(newick)
    current_sse = _tree_sse(adj, node_labels, dist, label_to_idx)
    best_sse = current_sse
    
    # Perform NNI refinement on NJ tree
    refined = _spr_refine(newick, labels, dist, label_to_idx)
    if _is_valid_newick(refined, labels):
        adj2, node_labels2, is_leaf2, n_nodes2 = _parse_newick_to_tree(refined)
        refined_sse = _tree_sse(adj2, node_labels2, dist, label_to_idx)
        if refined_sse < best_sse:
            best_sse = refined_sse
            best_newick = refined
    
    # If time permits, try multiple random restarts
    if not _time_exceeded() and n_nodes >= 5:
        try:
            import random
            import numpy as np
            
            n_leaves = len(labels)
            num_restarts = 3
            
            for restart in range(num_restarts):
                if _time_exceeded():
                    break
                
                random.seed(42 + restart)
                leaf_labels = list(labels)
                random.shuffle(leaf_labels)
                
                def _random_newick(leaves):
                    if len(leaves) <= 1:
                        return leaves[0] if leaves else "x"
                    if len(leaves) == 2:
                        return "(" + leaves[0] + "," + leaves[1] + ")"
                    mid = len(leaves) // 2
                    left = _random_newick(leaves[:mid])
                    right = _random_newick(leaves[mid:])
                    return "(" + left + "," + right + ")"
                
                random_newick = _random_newick(leaf_labels) + ";"
                
                if _is_valid_newick(random_newick, labels):
                    adj3, node_labels3, is_leaf3, n_nodes3 = _parse_newick_to_tree(random_newick)
                    random_sse = _tree_sse(adj3, node_labels3, dist, label_to_idx)
                    
                    if random_sse < best_sse:
                        best_sse = random_sse
                        best_newick = random_newick
                    
                    if not _time_exceeded():
                        refined_random = _spr_refine(random_newick, labels, dist, label_to_idx)
                        if _is_valid_newick(refined_random, labels):
                            adj4, node_labels4, is_leaf4, n_nodes4 = _parse_newick_to_tree(refined_random)
                            refined_random_sse = _tree_sse(adj4, node_labels4, dist, label_to_idx)
                            if refined_random_sse < best_sse:
                                best_sse = refined_random_sse
                                best_newick = refined_random
        except Exception:
            pass
    
    return best_newick
