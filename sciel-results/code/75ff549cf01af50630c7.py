import math
import time
import numpy as np
import re

_GLOBAL_START = None
_DEADLINE = 200.0  # Total budget for all instances (180s + margin)

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
    """
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    node_counter = [0]
    adj = {}
    labels = []
    
    def create_node(label=None):
        nid = node_counter[0]
        node_counter[0] += 1
        adj[nid] = []
        labels.append(label)
        return nid
    
    def parse(pos):
        if s[pos] == '(':
            pos += 1
            internal_id = create_node(None)
            first_child = None
            while True:
                child_id, pos = parse(pos)
                if first_child is None:
                    first_child = child_id
                else:
                    adj[internal_id].append(child_id)
                    adj[child_id].append(internal_id)
                
                if pos < len(s) and s[pos] == ',':
                    pos += 1
                    continue
                elif pos < len(s) and s[pos] == ')':
                    pos += 1
                    break
                else:
                    break
            
            if first_child is not None:
                adj[internal_id].append(first_child)
                adj[first_child].append(internal_id)
            
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
    return adj, labels

def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    L = len(sequences[labels[0]])
    char_to_idx = {c: i for i, c in enumerate('ACGT')}
    seqs = np.zeros((n, L), dtype=np.int8)
    for i, l in enumerate(labels):
        s = sequences[l]
        # Vectorized encoding
        for j, c in enumerate(s):
            seqs[i, j] = char_to_idx[c]
    
    # Pairwise differences
    diff = (seqs[:, None, :] != seqs[None, :, :]).sum(axis=2)
    p = diff.astype(np.float64) / L
    p_safe = np.minimum(p, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p_safe / 3.0)
    np.fill_diagonal(d, 0.0)
    d = (d + d.T) * 0.5
    return d

def _build_path_matrix(adj, labels):
    """
    Build path matrix A and distance vector d_vec for SSE.
    A[i, j] = 1 if edge j is on path between leaf i and leaf k
    """
    # Identify leaves
    leaf_nodes = []
    leaf_labels = []
    for node in adj:
        if labels[node] is not None:
            leaf_nodes.append(node)
            leaf_labels.append(labels[node])
    
    n_leaves = len(leaf_nodes)
    if n_leaves < 2:
        return None, None, None
        
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
    
    # For each leaf, BFS to find paths to other leaves
    # To be efficient, we can just do BFS from each leaf
    for i in range(n_leaves):
        li = leaf_nodes[i]
        # BFS from li
        parent = {li: -1}
        visited = {li}
        queue = [li]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    queue.append(v)
        
        for j in range(i + 1, n_leaves):
            lj = leaf_nodes[j]
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
                A[num_pairs * 0 + (i * n_leaves - i * (i+1)//2 + j - i - 1), edge_index[edge]] = 1.0
                # Wait, index calculation is tricky. Let's use a simpler counter.
    
    # Let's redo the index calculation properly
    pair_idx = 0
    for i in range(n_leaves):
        li = leaf_nodes[i]
        parent = {li: -1}
        visited = {li}
        queue = [li]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    queue.append(v)
        
        for j in range(i + 1, n_leaves):
            lj = leaf_nodes[j]
            path = []
            curr = lj
            while curr != -1:
                path.append(curr)
                curr = parent[curr]
            path.reverse()
            
            for k in range(len(path) - 1):
                u, v = path[k], path[k + 1]
                edge = (u, v) if u < v else (v, u)
                A[pair_idx, edge_index[edge]] = 1.0
            pair_idx += 1
            
    return A, edges, edge_index, leaf_labels

def _tree_sse(adj, labels, dist, label_to_idx):
    """
    Compute SSE for a tree topology.
    """
    A, edges, edge_index, leaf_labels = _build_path_matrix(adj, labels)
    if A is None:
        return 0.0
        
    # Build d_vec corresponding to A
    d_vec = np.zeros(A.shape[0])
    pair_idx = 0
    for i in range(len(leaf_labels)):
        for j in range(i + 1, len(leaf_labels)):
            li = leaf_labels[i]
            lj = leaf_labels[j]
            idx_i = label_to_idx[li]
            idx_j = label_to_idx[lj]
            d_vec[pair_idx] = dist[idx_i][idx_j]
            pair_idx += 1
            
    try:
        x, residuals, rank, s = np.linalg.lstsq(A, d_vec, rcond=None)
        pred = A.dot(x)
        sse = np.sum((pred - d_vec) ** 2)
    except:
        return float('inf')
        
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

def _adj_to_newick(adj, node_labels):
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

def _generate_spr_moves(adj, node_labels):
    """
    Generate all SPR moves.
    Yields (new_adj, new_labels)
    """
    # Identify edges
    edges = set()
    for u in adj:
        for v in adj[u]:
            if u < v:
                edges.add((u, v))
            else:
                edges.add((v, u))
    edges = list(edges)
    
    for edge in edges:
        u, v = edge
        
        # Prune subtree at u (component containing u when edge u-v is removed)
        # and regraft it on other edges.
        
        # Determine component containing u
        comp_u = set()
        queue = [u]
        visited = {u}
        while queue:
            curr = queue.pop()
            comp_u.add(curr)
            for nb in adj[curr]:
                if nb != v and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
                    
        # The "stump" is the component containing v
        # We prune comp_u and regraft it on an edge in the stump.
        
        # Get edges in the stump
        stump_edges = []
        for e in edges:
            if e[0] in comp_u or e[1] in comp_u:
                # Edge is in comp_u or crosses
                # We want edges entirely in the stump (v side)
                # Actually, we can regraft on any edge in the stump.
                pass
        
        # Find all edges in the stump (not in comp_u)
        stump_nodes = set(adj.keys()) - comp_u
        for e in edges:
            if e[0] in stump_nodes and e[1] in stump_nodes:
                stump_edges.append(e)
                
        # Also, we can regraft on the edge connecting the stump to the pruned part? No, that's the original position.
        # We can regraft on any edge in the stump.
        
        for target_edge in stump_edges:
            tu, tv = target_edge
            
            # New connections:
            # Remove u-v
            # Remove tu-tv
            # Add u-tu (or u-tv, symmetry) and v-tv (or v-tu)
            # Actually, we attach the root of the pruned subtree (u) to one end of the target edge,
            # and the root of the stump (v) to the other? No.
            
            # The pruned subtree is rooted at u. The stump is rooted at v.
            # We cut u-v.
            # We cut tu-tv.
            # We connect u to tu and v to tv? Or u to tv and v to tu?
            # There are 2 ways to connect the two components after cutting 2 edges.
            
            # Case 1: u-tu, v-tv
            new_adj = {x: list(y) for x, y in adj.items()}
            # Remove u-v
            new_adj[u].remove(v)
            new_adj[v].remove(u)
            # Remove tu-tv
            new_adj[tu].remove(tv)
            new_adj[tv].remove(tu)
            # Add u-tu
            new_adj[u].append(tu)
            new_adj[tu].append(u)
            # Add v-tv
            new_adj[v].append(tv)
            new_adj[tv].append(v)
            
            yield new_adj, node_labels
            
            # Case 2: u-tv, v-tu
            new_adj = {x: list(y) for x, y in adj.items()}
            new_adj[u].remove(v)
            new_adj[v].remove(u)
            new_adj[tu].remove(tv)
            new_adj[tv].remove(tu)
            new_adj[u].append(tv)
            new_adj[tv].append(u)
            new_adj[v].append(tu)
            new_adj[tu].append(v)
            
            yield new_adj, node_labels

def _spr_refine(newick, labels, dist, label_to_idx):
    adj, node_labels = _parse_newick_to_tree(newick)
    current_sse = _tree_sse(adj, node_labels, dist, label_to_idx)
    
    improved = True
    max_rounds = 10
    round_num = 0
    
    while improved and round_num < max_rounds:
        improved = False
        round_num += 1
        if _time_exceeded():
            break
            
        best_sse = current_sse
        best_adj = None
        
        # Generate moves
        moves = list(_generate_spr_moves(adj, node_labels))
        
        for new_adj, new_labels in moves:
            if _time_exceeded():
                break
            sse = _tree_sse(new_adj, new_labels, dist, label_to_idx)
            if sse < best_sse - 1e-10:
                best_sse = sse
                best_adj = new_adj
                improved = True
                
        if best_adj is not None:
            adj = best_adj
            current_sse = best_sse
            
    return _adj_to_newick(adj, node_labels)

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
            refined_newick = _spr_refine(nj_newick, labels, dist, label_to_idx)
            if _is_valid_newick(refined_newick, labels):
                newick = refined_newick
        except Exception:
            pass
            
    if not _is_valid_newick(newick, labels):
        return "(" + ",".join(labels) + ");"
        
    return newick
