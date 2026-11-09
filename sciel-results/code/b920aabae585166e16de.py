"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def _compute_triangles(adj, n):
    """Return list of triangles as tuples (u,v,w) with u<v<w."""
    triangles = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                for w in range(v + 1, n):
                    if adj[v][w] and adj[u][w]:
                        triangles.append((u, v, w))
    return triangles

def _compute_max_tri_sum(adj, n, triangles):
    """Compute max degree sum over triangles. Returns (max_sum, tri_list_with_sums)."""
    if not triangles:
        return 0, []
    deg = [sum(adj[i]) for i in range(n)]
    tri_sums = [(u, v, w, deg[u] + deg[v] + deg[w]) for (u, v, w) in triangles]
    max_sum = max(s for _, _, _, s in tri_sums)
    return max_sum, tri_sums

def _compute_qubo_score(adj, n, triangles):
    """Compute sum of squared triangle degree sums (QUBO surrogate)."""
    if not triangles:
        return 0
    deg = [sum(adj[i]) for i in range(n)]
    return sum((deg[u] + deg[v] + deg[w]) ** 2 for (u, v, w) in triangles)

def _make_base_graph(n=16):
    """Create K_{8,8} + one intra-part edge."""
    adj = [[0] * n for _ in range(n)]
    for u in range(8):
        for v in range(8, 16):
            adj[u][v] = adj[v][u] = 1
    # Add one edge within left part
    adj[0][1] = adj[1][0] = 1
    return adj

def _edges_from_adj(adj, n):
    """Convert adjacency matrix to edge list."""
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                edges.append([u, v])
    return edges

def _edge_count(adj, n):
    return sum(sum(row) for row in adj) // 2

def construct():
    n = 16
    target_edges = 65
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Initialize with base graph
    adj = _make_base_graph(n)
    triangles = _compute_triangles(adj, n)
    max_sum, tri_sums = _compute_max_tri_sum(adj, n, triangles)
    best_adj = [row[:] for row in adj]
    best_max_sum = max_sum
    best_qubo = _compute_qubo_score(adj, n, triangles)
    
    # Simulated annealing
    T_start = 5.0
    T_end = 0.01
    alpha = 0.9995
    
    T = T_start
    step = 0
    max_steps = 200000
    
    while time.time() < deadline and step < max_steps:
        step += 1
        
        # Generate a random edge swap: remove one edge, add another
        # Collect current edges
        current_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u][v]]
        
        if len(current_edges) < 2:
            T = T_end
            break
        
        # Pick a random existing edge to remove
        rem_u, rem_v = random.choice(current_edges)
        
        # Pick a random non-edge to add (not the same edge)
        # Try a few times to find a valid non-edge
        valid_add = False
        for _attempt in range(50):
            add_u = random.randint(0, n - 1)
            add_v = random.randint(0, n - 1)
            if add_u == add_v:
                continue
            if add_u > add_v:
                add_u, add_v = add_v, add_u
            if add_u == rem_u and add_v == rem_v:
                continue
            if not adj[add_u][add_v]:
                valid_add = True
                break
        
        if not valid_add:
            continue
        
        # Apply swap
        adj[rem_u][rem_v] = 0
        adj[rem_v][rem_u] = 0
        adj[add_u][add_v] = 1
        adj[add_v][add_u] = 1
        
        # Recompute triangles and scores
        triangles = _compute_triangles(adj, n)
        new_max_sum, new_tri_sums = _compute_max_tri_sum(adj, n, triangles)
        new_qubo = _compute_qubo_score(adj, n, triangles)
        
        # Acceptance: use combined score (QUBO surrogate + max penalty)
        old_combined = best_qubo + 10 * best_max_sum
        new_combined = new_qubo + 10 * new_max_sum
        
        delta = new_combined - old_combined
        
        if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            # Accept
            best_adj = [row[:] for row in adj]
            best_max_sum = new_max_sum
            best_qubo = new_qubo
        else:
            # Reject: undo swap
            adj[rem_u][rem_v] = 1
            adj[rem_v][rem_u] = 1
            adj[add_u][add_v] = 0
            adj[add_v][add_u] = 0
        
        T = max(T_end, T * alpha)
    
    # Final: verify edge count and return
    final_adj = best_adj
    final_edges = _edges_from_adj(final_adj, n)
    
    # If edge count is wrong, try to fix
    ec = len(final_edges)
    if ec != target_edges:
        # Fallback to base graph
        final_adj = _make_base_graph(n)
        final_edges = _edges_from_adj(final_adj, n)
    
    return {'n': n, 'edges': final_edges}
# EVOLVE-BLOCK-END
