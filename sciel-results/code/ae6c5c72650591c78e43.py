import numpy as np
import time
import math
from collections import defaultdict

def _jc_distance_matrix(sequences):
    labels = sorted(sequences.keys(), key=lambda x: int(x[1:]))
    n = len(labels)
    if n == 0:
        return np.array([]), labels
    
    L = len(sequences[labels[0]])
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mat = np.zeros((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        for j, c in enumerate(s):
            mat[i, j] = char_map[c]
            
    diff = (mat[:, np.newaxis, :] != mat[np.newaxis, :, :])
    mismatches = np.sum(diff, axis=2)
    p = mismatches / L
    
    D = np.zeros((n, n))
    mask = p < 0.7499
    D[mask] = -0.75 * np.log(1.0 - 4.0 / 3.0 * p[mask])
    D[~mask] = 10.0
    np.fill_diagonal(D, 0.0)
    return D, labels

def _nj_tree(D):
    n = D.shape[0]
    if n == 1:
        return "0;"
    if n == 2:
        return "(0,1);"
    if n == 3:
        return "(0,1,2);"
        
    active = list(range(n))
    dist = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[(i, j)] = D[i, j]
                dist[(j, i)] = D[i, j]
                
    newicks = {i: str(i) for i in range(n)}
    
    while len(active) > 3:
        m = len(active)
        r = {u: 0.0 for u in active}
        for u in active:
            for v in active:
                if u != v:
                    r[u] += dist[(u, v)]
                    
        best_val = float('inf')
        best_u, best_v = -1, -1
        for i_idx, u in enumerate(active):
            for v in active[i_idx+1:]:
                val = (m - 2) * dist[(u, v)] - r[u] - r[v]
                if val < best_val:
                    best_val = val
                    best_u = u
                    best_v = v
                    
        new_node = max(active) + 1
        newicks[new_node] = f"({newicks[best_u]},{newicks[best_v]})"
        
        new_active = [x for x in active if x not in (best_u, best_v)] + [new_node]
        
        for w in new_active:
            if w == new_node:
                continue
            d_uw = dist.get((best_u, w), 0.0)
            d_vw = dist.get((best_v, w), 0.0)
            d_uv = dist[(best_u, best_v)]
            d_nw = 0.5 * (d_uw + d_vw - d_uv)
            dist[(new_node, w)] = d_nw
            dist[(w, new_node)] = d_nw
            
        active = new_active
        
    a, b, c = active
    return f"({newicks[a]},{newicks[b]},{newicks[c]})"

def _newick_to_adj(newick, labels):
    if newick.endswith(';'):
        newick = newick[:-1]
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    nodes = {}
    adj = defaultdict(list)
    
    def parse(s, pos):
        if s[pos] == '(':
            pos += 1
            child1, pos = parse(s, pos)
            if s[pos] == ',':
                pos += 1
                child2, pos = parse(s, pos)
            else:
                child2 = -1
            if s[pos] == ')':
                pos += 1
            node_id = len(nodes)
            nodes[node_id] = None
            adj[node_id].append(child1)
            adj[child1].append(node_id)
            if child2 != -1:
                adj[node_id].append(child2)
                adj[child2].append(node_id)
            return node_id, pos
        else:
            end = pos
            while end < len(s) and s[end] not in ',)':
                end += 1
            label = s[pos:end]
            pos = end
            node_id = label_to_id.get(label, -1)
            if node_id == -1:
                node_id = len(nodes)
                nodes[node_id] = label
            else:
                nodes[node_id] = label
            return node_id, pos

    root, _ = parse(newick, 0)
    return adj, nodes

def _calc_ssr(adj, D_obs, n):
    # Build X matrix
    # Edges:
    # We need to identify edges.
    # In an unrooted tree, edges are undirected.
    # We need to assign an index to each edge.
    # And for each pair (i, j), which edges are on the path?
    
    # This is expensive to build X from scratch every time.
    # But n is small.
    
    # 1. Get list of edges.
    # Edges are pairs (u, v) with u < v to avoid duplicates.
    # We need to know the branch length variable for each edge.
    
    # Let's find all edges.
    edges = set()
    for u in adj:
        for v in adj[u]:
            if u < v:
                edges.add((u, v))
            else:
                edges.add((v, u))
                
    edge_list = list(edges)
    m = len(edge_list)
    edge_idx = {e: i for i, e in enumerate(edge_list)}
    
    # 2. For each pair of leaves (i, j), find the path.
    # Use BFS/DFS.
    # X is (n_pairs, m)
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    n_pairs = len(pairs)
    X = np.zeros((n_pairs, m))
    D_vec = np.zeros(n_pairs)
    
    for k, (i, j) in enumerate(pairs):
        # Find path from i to j in adj
        # BFS
        visited = {i}
        queue = [(i, [i])]
        path = None
        while queue:
            node, path_so_far = queue.pop(0)
            if node == j:
                path = path_so_far
                break
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path_so_far + [neighbor]))
                    
        if path:
            D_vec[k] = D_obs[i, j]
            for e in range(len(path) - 1):
                u, v = path[e], path[e+1]
                edge = (u, v) if u < v else (v, u)
                X[k, edge_idx[edge]] = 1.0
        else:
            # Should not happen
            D_vec[k] = 0.0
            
    # Solve LLS: L = (X^T X)^-1 X^T D
    # Add small regularization to handle singular matrices
    XtX = X.T @ X + 1e-6 * np.eye(m)
    XtD = X.T @ D_vec
    try:
        L = np.linalg.solve(XtX, XtD)
    except:
        L = np.linalg.lstsq(X, D_vec, rcond=None)[0]
        
    # SSR = sum((D - X L)^2)
    pred = X @ L
    ssr = np.sum((D_vec - pred) ** 2)
    return ssr

def _nni_refine(adj, D_obs, n, max_passes=5):
    best_ssr = _calc_ssr(adj, D_obs, n)
    
    for _ in range(max_passes):
        improved = False
        # Get internal edges
        # Internal node: degree > 2? No, in binary tree, internal nodes have degree 3.
        # Leaves have degree 1.
        # So internal edges connect two nodes of degree 3.
        
        internal_edges = []
        for u in adj:
            if len(adj[u]) == 3:
                for v in adj[u]:
                    if u < v and len(adj[v]) == 3:
                        internal_edges.append((u, v))
                        
        for u, v in internal_edges:
            # Neighbors of u (excluding v)
            nu = [x for x in adj[u] if x != v]
            # Neighbors of v (excluding u)
            nv = [x for x in adj[v] if x != u]
            
            if len(nu) != 2 or len(nv) != 2:
                continue
                
            a, b = nu
            c, d = nv
            
            # Original: u connects a,b; v connects c,d; u-v connects them.
            # Alt 1: u connects a,c; v connects b,d; u-v.
            # Alt 2: u connects a,d; v connects b,c; u-v.
            
            # Try Alt 1
            # Modify adj
            # Remove u-v, u-a, u-b, v-c, v-d
            # Add u-a, u-c, v-b, v-d, u-v (u-v stays)
            # Wait, u-v stays.
            # Old: u-a, u-b, v-c, v-d, u-v
            # New: u-a, u-c, v-b, v-d, u-v
            
            # Save original state
            orig_u = list(adj[u])
            orig_v = list(adj[v])
            
            # Apply Alt 1
            adj[u].remove(b)
            adj[b].remove(u)
            adj[v].remove(c)
            adj[c].remove(v)
            
            adj[u].append(c)
            adj[c].append(u)
            adj[v].append(b)
            adj[b].append(v)
            
            ssr1 = _calc_ssr(adj, D_obs, n)
            
            if ssr1 < best_ssr:
                best_ssr = ssr1
                improved = True
            else:
                # Revert
                adj[u].remove(c)
                adj[c].remove(u)
                adj[v].remove(b)
                adj[b].remove(v)
                
                adj[u].append(b)
                adj[b].append(u)
                adj[v].append(c)
                adj[c].append(v)
                
            # Try Alt 2
            # Old: u-a, u-b, v-c, v-d
            # New: u-a, u-d, v-b, v-c
            
            adj[u].remove(b)
            adj[b].remove(u)
            adj[v].remove(d)
            adj[d].remove(v)
            
            adj[u].append(d)
            adj[d].append(u)
            adj[v].append(b)
            adj[b].append(v)
            
            ssr2 = _calc_ssr(adj, D_obs, n)
            
            if ssr2 < best_ssr:
                best_ssr = ssr2
                improved = True
            else:
                # Revert
                adj[u].remove(d)
                adj[d].remove(u)
                adj[v].remove(b)
                adj[b].remove(v)
                
                adj[u].append(b)
                adj[b].append(u)
                adj[v].append(d)
                adj[d].append(v)
                
        if not improved:
            break
            
    return adj, best_ssr

def _adj_to_newick(adj, labels, n):
    # Root the tree arbitrarily at node 0
    # Do DFS
    visited = set()
    def dfs(node, parent):
        if node in visited:
            return ""
        visited.add(node)
        
        # If leaf
        if len(adj[node]) == 1 and node < n:
            return labels[node]
            
        children = []
        for neighbor in adj[node]:
            if neighbor == parent:
                continue
            child_str = dfs(neighbor, node)
            if child_str:
                children.append(child_str)
                
        if not children:
            return labels[node] if node < n else ""
            
        return "(" + ",".join(children) + ")"
        
    # Start from 0
    res = dfs(0, -1)
    return res + ";"

def reconstruct(sequences):
    start_time = time.time()
    D, labels = _jc_distance_matrix(sequences)
    n = len(labels)
    
    if n <= 3:
        return "(" + ",".join(labels) + ");"
        
    newick = _nj_tree(D)
    adj, _ = _newick_to_adj(newick, labels)
    
    # Time check
    if time.time() - start_time < 260:
        adj, _ = _nni_refine(adj, D, n)
        
    final_newick = _adj_to_newick(adj, labels, n)
    return final_newick
