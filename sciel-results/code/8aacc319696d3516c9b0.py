import math
import time
import numpy as np

def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    
    L = len(sequences[labels[0]])
    lookup = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    seq_matrix = np.zeros((n, L), dtype=np.int8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j, c in enumerate(seq):
            seq_matrix[i, j] = lookup[c]
            
    dist_mat = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.sum(seq_matrix[i] != seq_matrix[j])
            p = diff / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist_mat[i, j] = d
            dist_mat[j, i] = d
            
    return dist_mat

def _nj_tree(dist_mat, labels):
    n = len(labels)
    if n <= 2:
        if n == 1:
            return labels[0]
        else:
            return f"({labels[0]},{labels[1]})"
    
    max_nodes = 2 * n
    D = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            D[i, j] = dist_mat[i, j]
    
    active = list(range(n))
    tree_edges = {}
    next_id = n
    
    while len(active) > 3:
        k = len(active)
        
        r = np.zeros(k)
        for i, node in enumerate(active):
            r[i] = np.sum(D[node, active]) - D[node, node]
        
        best_q = float('inf')
        best_i = -1
        best_j = -1
        
        for i in range(k):
            for j in range(i + 1, k):
                node_i = active[i]
                node_j = active[j]
                q = (k - 2) * D[node_i, node_j] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i = i
                    best_j = j
        
        node_a = active[best_i]
        node_b = active[best_j]
        
        delta = (r[best_i] - r[best_j]) / (k - 2)
        len_a = 0.5 * (D[node_a, node_b] + delta)
        len_b = 0.5 * (D[node_a, node_b] - delta)
        
        if len_a < 0: len_a = 0.0
        if len_b < 0: len_b = 0.0
        
        new_node = next_id
        next_id += 1
        
        tree_edges[new_node] = (node_a, node_b, len_a, len_b)
        
        for x in active:
            if x != node_a and x != node_b:
                D[new_node, x] = 0.5 * (D[node_a, x] + D[node_b, x] - D[node_a, node_b])
                D[x, new_node] = D[new_node, x]
        
        active = [x for x in active if x != node_a and x != node_b]
        active.append(new_node)
        
    if len(active) == 3:
        u, v, w = active
        len_u = 0.5 * (D[u, v] + D[u, w] - D[v, w])
        len_v = 0.5 * (D[u, v] + D[v, w] - D[u, w])
        len_w = 0.5 * (D[u, w] + D[v, w] - D[u, v])
        
        if len_u < 0: len_u = 0.0
        if len_v < 0: len_v = 0.0
        if len_w < 0: len_w = 0.0
        
        new_node_1 = next_id
        next_id += 1
        tree_edges[new_node_1] = (u, v, len_u, len_v)
        
        new_node_2 = next_id
        next_id += 1
        tree_edges[new_node_2] = (new_node_1, w, 0.0, len_w)
        
        root = new_node_2
    else:
        root = -1
        
    def get_newick(node):
        if node < n:
            return labels[node]
        else:
            if node not in tree_edges:
                return labels[0] if n > 0 else ""
            child_a, child_b, len_a, len_b = tree_edges[node]
            return f"({get_newick(child_a)}:{len_a:.4f},{get_newick(child_b)}:{len_b:.4f})"
    
    if root != -1:
        return get_newick(root)
    else:
        return "(" + ",".join(labels) + ")"

def reconstruct(sequences):
    start_time = time.time()
    time_limit = 265.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 1:
            return f"{labels[0]};"
        else:
            return f"({labels[0]},{labels[1]})"

    dist_mat = _jc_distance_matrix(sequences, labels)
    newick = _nj_tree(dist_mat, labels)
    
    if time.time() - start_time > time_limit:
        return newick

    return newick
