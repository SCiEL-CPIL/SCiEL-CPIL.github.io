"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def _count_c4(adj):
    """Count 4-cycles in a graph given its adjacency matrix (vectorized)."""
    common = adj @ adj
    c = common - adj
    iu = np.triu_indices(common.shape[0], k=1)
    vals = c[iu]
    return int(np.sum(vals * (vals - 1) // 2))

def _build_polarity_graph():
    """
    Construct the polarity graph of PG(2,5) with 90 edges.
    For q=5 (odd), the orthogonal polarity graph of PG(2,q) has:
    - n = q^2 + q + 1 = 31 vertices
    - 90 edges (C4-free)
    """
    q = 5
    points = []
    seen = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0 and b == 0 and c == 0:
                    continue
                vec = [a, b, c]
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], q - 2, q)
                        vec = [(v * inv) % q for v in vec]
                        break
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(vec)
    
    n = len(points)
    assert n == 31, f"Expected 31 points, got {n}"
    
    M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 4]], dtype=int)
    
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            x = np.array(points[i], dtype=int)
            y = np.array(points[j], dtype=int)
            val = (x @ M @ y) % q
            if val == 0:
                adj[i, j] = 1
                adj[j, i] = 1
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                edges.append([i, j])
    
    return edges, adj

def _delta_c4_add(adj, i, j):
    """
    Compute change in C4 count when adding edge (i,j) to graph.
    In a C4-free graph, this equals C(common_neighbors(i,j), 2).
    For general graphs, it's the number of length-2 paths between i and j
    not already counted, which equals common_neighbors(i,j) choose 2
    minus any existing C4s... Actually for general case:
    New C4s = number of pairs (a,b) with a~i, a~j, b~i, b~j, a<b, a!=b
    that don't already form a C4 with edge (i,j). Since (i,j) is not an edge,
    no C4 currently uses it, so new C4s = C(|N(i)∩N(j)|, 2).
    """
    common = np.sum(adj[i] & adj[j])
    return common * (common - 1) // 2

def _delta_c4_remove(adj, i, j):
    """
    Compute change in C4 count when removing edge (i,j).
    Returns the number of C4s destroyed (positive).
    C4s containing edge (i,j): for each pair (a,b) with a~i, a~j, b~i, b~j,
    a<b, a!=b, a!=j, b!=i, the C4 i-a-j-b-i exists.
    Count = C(|N(i)∩N(j)|, 2) but we need to exclude a=i or b=j which
    can't happen since i not in N(j) if we consider open neighborhoods...
    Actually N(i) includes j if (i,j) is an edge. So we need N(i)\\{j} ∩ N(j)\\{i}.
    """
    ni = adj[i].copy()
    nj = adj[j].copy()
    ni[j] = 0
    nj[i] = 0
    common = np.sum(ni & nj)
    return common * (common - 1) // 2

def _fast_local_search(adj, n, max_iters=500, deadline=None):
    """
    Fast local search using delta computations.
    Try edge swaps: remove edge e1, add non-edge e2.
    Net delta = delta_remove(e1) - delta_add(e2) + interaction.
    For simplicity, first do greedy: find best single swap by evaluating
    a limited set of candidates using fast delta computations.
    """
    current_c4 = _count_c4(adj)
    best_c4 = current_c4
    best_adj = adj.copy()
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        best_delta = 0
        best_swap = None
        
        # Build edge and non-edge lists
        edges = []
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    edges.append((i, j))
                else:
                    non_edges.append((i, j))
        
        # Precompute delta for each edge removal and non-edge addition
        # For speed, only consider top candidates
        remove_deltas = {}
        for e in edges:
            i, j = e
            remove_deltas[e] = _delta_c4_remove(adj, i, j)
        
        add_deltas = {}
        for e in non_edges:
            i, j = e
            add_deltas[e] = _delta_c4_add(adj, i, j)
        
        # Sort and take top candidates
        top_remove = sorted(remove_deltas.items(), key=lambda x: -x[1])[:20]
        top_add = sorted(add_deltas.items(), key=lambda x: x[1])[:20]
        
        # Try all combinations of top candidates
        for e1, rd in top_remove:
            i1, j1 = e1
            for e2, ad in top_add:
                i2, j2 = e2
                if e1 == e2:
                    continue
                # Compute actual delta by modifying and counting
                adj[i1, j1] = 0
                adj[j1, i1] = 0
                adj[i2, j2] = 1
                adj[j2, i2] = 1
                new_c4 = _count_c4(adj)
                adj[i1, j1] = 1
                adj[j1, i1] = 1
                adj[i2, j2] = 0
                adj[j2, i2] = 0
                
                delta = new_c4 - current_c4
                if delta < best_delta:
                    best_delta = delta
                    best_swap = (e1, e2)
                    improved = True
        
        if not improved:
            # Try broader search: all top-50 removes × all top-50 adds
            top_remove = sorted(remove_deltas.items(), key=lambda x: -x[1])[:50]
            top_add = sorted(add_deltas.items(), key=lambda x: x[1])[:50]
            
            for e1, rd in top_remove:
                i1, j1 = e1
                for e2, ad in top_add:
                    i2, j2 = e2
                    if e1 == e2:
                        continue
                    adj[i1, j1] = 0
                    adj[j1, i1] = 0
                    adj[i2, j2] = 1
                    adj[j2, i2] = 1
                    new_c4 = _count_c4(adj)
                    adj[i1, j1] = 1
                    adj[j1, i1] = 1
                    adj[i2, j2] = 0
                    adj[j2, i2] = 0
                    
                    delta = new_c4 - current_c4
                    if delta < best_delta:
                        best_delta = delta
                        best_swap = (e1, e2)
                        improved = True
                    if deadline is not None and time.time() > deadline:
                        break
                if deadline is not None and time.time() > deadline:
                    break
            
            if not improved:
                break
        
        # Apply best swap
        e1, e2 = best_swap
        i1, j1 = e1
        i2, j2 = e2
        adj[i1, j1] = 0
        adj[j1, i1] = 0
        adj[i2, j2] = 1
        adj[j2, i2] = 1
        current_c4 += best_delta
        if current_c4 < best_c4:
            best_c4 = current_c4
            best_adj = adj.copy()
    
    return best_adj, best_c4

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Build the polarity graph (C4-free, 90 edges)
    edges_90, adj = _build_polarity_graph()
    
    n = 31
    c4_base = _count_c4(adj)
    assert c4_base == 0, f"Expected 0 C4s in polarity graph, got {c4_base}"
    
    # Find the non-edge with minimum C4 creation (min common neighbors)
    best_single = None
    best_single_c4 = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                delta = _delta_c4_add(adj, i, j)
                if delta < best_single_c4:
                    best_single_c4 = delta
                    best_single = [i, j]
    
    # Add the best single edge
    if best_single is not None:
        i, j = best_single
        adj[i, j] = 1
        adj[j, i] = 1
    
    current_c4 = _count_c4(adj)
    
    # Run fast greedy local search to minimize C4 count
    adj_opt, c4_opt = _fast_local_search(adj, n, max_iters=300, deadline=deadline)
    
    # Extract edges from optimized adjacency matrix
    edges_final = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_opt[i, j] == 1:
                edges_final.append([i, j])
    
    # Verify we have exactly 91 edges
    assert len(edges_final) == 91, f"Expected 91 edges, got {len(edges_final)}"
    
    return edges_final

# EVOLVE-BLOCK-END
