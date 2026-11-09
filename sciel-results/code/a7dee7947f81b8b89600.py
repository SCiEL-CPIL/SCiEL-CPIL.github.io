import numpy as np
import math
import time

def reconstruct(sequences):
    start_time = time.time()
    DEADLINE = 265.0  # Safety margin for 270s limit
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 0:
        return ""
    if n == 1:
        return f"({labels[0]});"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # 1. Convert sequences to numpy array
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = []
    for lab in labels:
        s = sequences[lab]
        arr = np.array([seq_map[c] for c in s], dtype=np.int8)
        seqs.append(arr)
    seqs = np.vstack(seqs)
    L = seqs.shape[1]
    
    # 2. Compute pairwise JC distances
    diff = (seqs[:, None, :] != seqs[None, :, :]).sum(axis=2)
    p = diff.astype(np.float64) / L
    p = np.clip(p, 0.0, 0.7499)
    dist = -0.75 * np.log(1.0 - (4.0/3.0) * p)
    np.fill_diagonal(dist, 0.0)
    
    # 3. Neighbor Joining Algorithm
    max_nodes = 2 * n
    D = np.zeros((max_nodes, max_nodes))
    D[:n, :n] = dist
    
    active = list(range(n))
    next_id = n
    
    edges = []
    
    while len(active) > 2:
        if time.time() > DEADLINE:
            break
            
        m = len(active)
        idx = np.array(active)
        D_sub = D[np.ix_(idx, idx)]
        
        r = D_sub.sum(axis=1)
        Q = (m - 2) * D_sub - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i_idx, j_idx = min_idx
        
        node_i = active[i_idx]
        node_j = active[j_idx]
        
        new_node = next_id
        next_id += 1
        
        D_ij = D_sub[i_idx, j_idx]
        l_iu = 0.5 * D_ij + 0.5 * (r[i_idx] - r[j_idx]) / (m - 2)
        l_ju = D_ij - l_iu
        
        if l_iu < 0: l_iu = 0.0
        if l_ju < 0: l_ju = 0.0
        
        edges.append((node_i, new_node, l_iu))
        edges.append((node_j, new_node, l_ju))
        
        other_active = [x for x in active if x != node_i and x != node_j]
        for k in other_active:
            d_ik = D[node_i, k]
            d_jk = D[node_j, k]
            D[new_node, k] = 0.5 * (d_ik + d_jk - D_ij)
            D[k, new_node] = D[new_node, k]
            
        active.remove(node_i)
        active.remove(node_j)
        active.append(new_node)
        
    if len(active) == 2:
        node_i, node_j = active
        D_ij = D[node_i, node_j]
        edges.append((node_i, node_j, D_ij))
        
    # 4. Build Tree Structure
    adj = {i: [] for i in range(next_id)}
    for u, v, l in edges:
        adj[u].append((v, l))
        adj[v].append((u, l))
        
    root = None
    for node in range(n, next_id):
        if node in adj and len(adj[node]) > 0:
            root = node
            break
    if root is None:
        root = 0
        
    parent = {root: -1}
    children = {i: [] for i in range(next_id)}
    
    stack = [root]
    visited = {root}
    while stack:
        u = stack.pop()
        for v, l in adj[u]:
            if v not in visited:
                visited.add(v)
                parent[v] = u
                children[u].append((v, l))
                stack.append(v)
                
    def to_newick(node, parent_id):
        if node < n:
            return f"{labels[node]}"
        
        child_strs = []
        for v, l in children[node]:
            if v == parent_id:
                continue
            child_strs.append(to_newick(v, node))
            
        if not child_strs:
            return f"{labels[node]}"
            
        return f"({','.join(child_strs)})"
        
    newick = to_newick(root, -1)
    return f"{newick};"
