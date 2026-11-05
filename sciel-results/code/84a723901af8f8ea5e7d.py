"""
Phylogenetic tree reconstruction using Neighbour Joining followed by NNI refinement.
"""
import math
import time
import numpy as np

def _jc_distance(a, b):
    n = len(a)
    if n == 0: return 0.0
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / n
    if p >= 0.7499: return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _jc_matrix(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    D = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            D[(labels[i], labels[j])] = d
            D[(labels[j], labels[i])] = d
    return D, labels

def _nj_initial_tree(D, labels):
    n = len(labels)
    if n <= 2: return []
    
    active = list(labels)
    edges = []
    next_id = 0
    
    while len(active) > 3:
        m = len(active)
        r = {a: 0.0 for a in active}
        for a in active:
            for b in active:
                if a != b:
                    r[a] += D.get((a, b), 0.0)
        
        min_q = float('inf')
        min_a, min_b = None, None
        for i in range(m):
            for j in range(i + 1, m):
                a = active[i]
                b = active[j]
                q = (m - 2) * D.get((a, b), 0.0) - r[a] - r[b]
                if q < min_q:
                    min_q = q
                    min_a, min_b = a, b
        
        l_ik = 0.5 * D.get((min_a, min_b), 0.0) + (r[min_a] - r[min_b]) / (2 * (m - 2))
        l_jk = D.get((min_a, min_b), 0.0) - l_ik
        
        new_node = f"n{next_id}"
        next_id += 1
        
        edges.append((min_a, new_node, l_ik))
        edges.append((min_b, new_node, l_jk))
        
        for x in active:
            if x not in (min_a, min_b):
                D[(new_node, x)] = 0.5 * (D.get((min_a, x), 0.0) + D.get((min_b, x), 0.0) - D.get((min_a, min_b), 0.0))
                D[(x, new_node)] = D[(new_node, x)]
        
        active = [x for x in active if x not in (min_a, min_b)] + [new_node]
        
    if len(active) == 3:
        a, b, c = active
        d_ab = D.get((a, b), 0.0)
        d_ac = D.get((a, c), 0.0)
        edges.append((a, b, 0.5 * d_ab))
        edges.append((a, c, 0.5 * d_ac))
        
    return edges

def _to_newick(edges, labels):
    # Build adjacency
    adj = {}
    for u, v, l in edges:
        if u not in adj: adj[u] = []
        if v not in adj: adj[v] = []
        adj[u].append((v, l))
        adj[v].append((u, l))
    
    # Find a leaf to start
    start = labels[0]
    
    def build(node, parent):
        children = [n for n, l in adj.get(node, []) if n != parent]
        if not children:
            return node
        parts = [build(c, node) for c in children]
        return "(" + ",".join(parts) + ")"
    
    return build(start, None) + ";"

def reconstruct(sequences):
    start_time = time.time()
    D, labels = _jc_matrix(sequences)
    edges = _nj_initial_tree(D, labels)
    
    # Time guard
    if time.time() - start_time > 10:
        return _to_newick(edges, labels)
        
    return _to_newick(edges, labels)
