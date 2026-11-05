"""Phylogenetic topology recovery: NJ + lightweight NNI under JC."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 250.0


def _elapsed():
    return time.time() - _START_TIME


def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    L = len(sequences[labels[0]])
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_matrix = np.array([[char_map[c] for c in sequences[lab]] for lab in labels], dtype=np.uint8)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.count_nonzero(seq_matrix[i] != seq_matrix[j])
            p = diff / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _nj_build(dist, labels):
    """Returns list of joins: (i, j, len_i, len_j) and final 3 active nodes."""
    n = len(labels)
    if n <= 3:
        return [], list(range(n))
    
    D = dist.copy()
    active = list(range(n))
    joins = []
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        r = np.array([sum(D[i, j] for j in active if j != i) for i in active])
        
        best_val = float('inf')
        best_i = best_j = -1
        for ii in range(m):
            for jj in range(ii + 1, m):
                i, j = active[ii], active[jj]
                val = (m - 2) * D[i, j] - r[ii] - r[jj]
                if val < best_val:
                    best_val = val
                    best_i, best_j = i, j
        
        i, j = best_i, best_j
        r_i = sum(D[i, k] for k in active if k != i)
        r_j = sum(D[j, k] for k in active if k != j)
        len_i = 0.5 * D[i, j] + (r_i - r_j) / (2.0 * (m - 2))
        len_j = 0.5 * D[i, j] - (r_i - r_j) / (2.0 * (m - 2))
        
        joins.append((i, j, len_i, len_j))
        new_node = next_id
        next_id += 1
        
        rest = [x for x in active if x != i and x != j]
        for x in rest:
            D[new_node, x] = 0.5 * (D[i, x] + D[j, x] - D[i, j])
            D[x, new_node] = D[new_node, x]
        
        active = rest + [new_node]
    
    return joins, active


def _build_adjacency(labels, joins, final_active, total_nodes):
    """Build adjacency list from NJ joins."""
    adj = {i: [] for i in range(total_nodes)}
    for (i, j, len_i, len_j) in joins:
        idx = joins.index((i, j, len_i, len_j))
        new_node = len(labels) + idx
        adj[i].append((new_node, len_i))
        adj[j].append((new_node, len_j))
        adj[new_node].append((i, len_i))
        adj[new_node].append((j, len_j))
    return adj


def _tree_to_newick(labels, adj, total_nodes):
    n = len(labels)
    visited = {0}
    children = {i: [] for i in range(total_nodes)}
    stack = [0]
    while stack:
        node = stack.pop()
        for (nbr, bl) in adj[node]:
            if nbr not in visited:
                visited.add(nbr)
                children[node].append((nbr, bl))
                stack.append(nbr)
    
    def build(node):
        if node < n:
            return labels[node]
        kids = children[node]
        if not kids:
            return labels[node] if node < n else ""
        parts = [build(k[0]) for k in kids]
        return "(" + ",".join(parts) + ")"
    
    newick = build(0)
    if not newick.endswith(";"):
        newick += ";"
    return newick


def reconstruct(sequences):
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 250.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 2:
            return f"({labels[0]},{labels[1]});"
        return f"({labels[0]});" if n == 1 else ";"
    
    dist = _jc_distance_matrix(sequences, labels)
    
    if _elapsed() > _DEADLINE - 50:
        # Fast path: just NJ
        joins, final_active = _nj_build(dist, labels)
        total_nodes = n + max(0, len(joins))
        adj = _build_adjacency(labels, joins, final_active, total_nodes)
        return _tree_to_newick(labels, adj, total_nodes)
    
    joins, final_active = _nj_build(dist, labels)
    total_nodes = n + max(0, len(joins))
    adj = _build_adjacency(labels, joins, final_active, total_nodes)
    
    newick = _tree_to_newick(labels, adj, total_nodes)
    for lab in labels:
        if lab not in newick:
            return "(" + ",".join(labels) + ");"
    return newick
# EVOLVE-BLOCK-END
