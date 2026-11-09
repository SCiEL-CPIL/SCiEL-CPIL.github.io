import numpy as np
import time
import math

_START_TIME = None
_DEADLINE = 260.0

def _now():
    return time.time()

def _time_left():
    return _DEADLINE - (_now() - _START_TIME)

def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    if n == 0:
        return np.zeros((0, 0))
    
    seq_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(sequences[labels[0]])
    num_seqs = np.zeros((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        for j, c in enumerate(s):
            num_seqs[i, j] = seq_dict[c]
            
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diffs = np.count_nonzero(num_seqs[i] != num_seqs[j])
            p = diffs / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist[i, j] = d
            dist[j, i] = d
    return dist

def _neighbor_joining(dist, labels):
    n = len(labels)
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}))"

    active = list(range(n))
    next_id = n
    structure = {}
    for i in range(n):
        structure[i] = None
        
    dist_dict = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_dict[(i, j)] = dist[i, j]
                
    while len(active) > 3:
        m = len(active)
        sums = {u: 0.0 for u in active}
        for u in active:
            for v in active:
                if u != v:
                    sums[u] += dist_dict.get((u, v), 0.0)
                    
        best_c = float('inf')
        best_u, best_v = -1, -1
        for i, u in enumerate(active):
            for v in active[i + 1:]:
                c = (m - 2) * dist_dict[(u, v)] - sums[u] - sums[v]
                if c < best_c:
                    best_c = c
                    best_u, best_v = u, v
                    
        u, v = best_u, best_v
        d_uv = dist_dict[(u, v)]
        
        new_node = next_id
        next_id += 1
        structure[new_node] = (u, v)
        
        for w in active:
            if w == u or w == v:
                continue
            d_uw = dist_dict[(u, w)]
            d_vw = dist_dict[(v, w)]
            d_new_w = 0.5 * (d_uw + d_vw - d_uv)
            dist_dict[(new_node, w)] = d_new_w
            dist_dict[(w, new_node)] = d_new_w
            
        active.remove(u)
        active.remove(v)
        active.append(new_node)
        
    a, b, c = active[0], active[1], active[2]
    
    def get_newick(node_id):
        if node_id < n:
            return labels[node_id]
        if node_id in structure:
            children = structure[node_id]
            if isinstance(children, tuple):
                left, right = children
                return f"({get_newick(left)},{get_newick(right)})"
            else:
                return str(node_id)
        return str(node_id)

    return f"({get_newick(a)},({get_newick(b)},{get_newick(c)}))"

def reconstruct(sequences):
    global _START_TIME
    _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"

    dist = _jc_distance_matrix(sequences, labels)
    newick = _neighbor_joining(dist, labels)
    
    return newick + ";"
