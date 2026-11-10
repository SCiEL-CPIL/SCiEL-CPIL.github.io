"""
Combines the Polarity Graph construction of PG(2,5) (structural insight) 
with the fast numpy-based local search (refinement machinery).
"""
import time
import random
import numpy as np

N = 31
M = 91

def count_c4(adj):
    """Count C4s from numpy adjacency matrix."""
    common = adj @ adj
    c4 = 0
    for u in range(N):
        for v in range(u + 1, N):
            c = common[u, v]
            if c >= 2:
                c4 += c * (c - 1) // 2
    return c4

def make_adj(edges):
    adj = np.zeros((N, N), dtype=np.int32)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    return adj

def delta_add_fast(adj, u, v):
    """C4s created by adding edge (u,v) assuming it's not currently present."""
    common = adj[u] & adj[v]
    cn = np.where(common)[0]
    if len(cn) < 2:
        return 0
    sub = adj[np.ix_(cn, cn)]
    return sub.sum() // 2

def local_search(edges, time_budget):
    """Hill-climbing with 1-swap moves and fast delta computation."""
    adj = make_adj(edges)
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))

    current_c4 = count_c4(adj)
    best_c4 = current_c4
    best_edges = [list(e) for e in edges]

    start_time = time.time()
    deadline = start_time + time_budget
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    while time.time() < deadline:
        improved = False
        for i in range(len(edges)):
            if time.time() > deadline:
                return best_edges
            u1, v1 = edges[i]
            e1 = (min(u1, v1), max(u1, v1))

            remove_delta = delta_add_fast(adj, u1, v1)

            adj[u1, v1] = 0
            adj[v1, u1] = 0
            edge_set.discard(e1)

            best_move = None
            best_delta = 0

            for u2, v2 in all_pairs:
                e2 = (min(u2, v2), max(u2, v2))
                if e2 in edge_set:
                    continue
                add_delta = delta_add_fast(adj, u2, v2)
                delta = add_delta - remove_delta
                if delta < best_delta:
                    best_delta = delta
                    best_move = (u2, v2)

            if best_move is not None:
                u2, v2 = best_move
                e2 = (min(u2, v2), max(u2, v2))
                adj[u2, v2] = 1
                adj[v2, u2] = 1
                edge_set.add(e2)
                edges[i] = [u2, v2]
                current_c4 += best_delta
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_edges = [list(e) for e in edges]
                improved = True
            else:
                adj[u1, v1] = 1
                adj[v1, u1] = 1
                edge_set.add(e1)

        if not improved:
            break

    return best_edges

def construct_polarity_graph():
    """
    Construct the polarity graph of PG(2,5) with an orthogonal polarity.
    This graph has 31 vertices and 90 edges.
    """
    # Generate all 31 points in PG(2,5)
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: first non-zero component is 1
                if x != 0:
                    rep = (1, y * pow(x, 4, 5) % 5, z * pow(x, 4, 5) % 5)
                elif y != 0:
                    rep = (0, 1, z * pow(y, 4, 5) % 5)
                else:
                    rep = (0, 0, 1)
                
                if rep not in seen:
                    seen.add(rep)
                    points.append(rep)
    
    # Map points to indices 0..30
    # The order doesn't strictly matter, but consistency does
    idx_map = {p: i for i, p in enumerate(points)}
    
    # Define symmetric bilinear form B(u, v) = u0*v1 + u1*v0 + u2*v2 mod 5
    # Edge exists if B(u, v) == 0
    edges = []
    for i in range(31):
        u = points[i]
        for j in range(i + 1, 31):
            v = points[j]
            # u0*v1 + u1*v0 + u2*v2
            val = (u[0] * v[1] + u[1] * v[0] + u[2] * v[2]) % 5
            if val == 0:
                edges.append([i, j])
                
    # Verify edge count
    # Should be 90
    return edges

def random_near_regular_init(seed):
    """Random graph with degree sequence close to regular, exactly M edges."""
    random.seed(seed)
    target = [6]*27 + [5]*4
    random.shuffle(target)
    
    edges = []
    edge_set = set()
    deg = [0]*N
    
    candidates = [(i, j) for i in range(N) for j in range(i+1, N)]
    random.shuffle(candidates)
    
    for u, v in candidates:
        if len(edges) >= M:
            break
        if deg[u] < target[u] and deg[v] < target[v]:
            edges.append([u, v])
            edge_set.add((u, v))
            deg[u] += 1
            deg[v] += 1
    
    if len(edges) < M:
        for u, v in candidates:
            if len(edges) >= M:
                break
            if (u, v) not in edge_set:
                edges.append([u, v])
                edge_set.add((u, v))
                deg[u] += 1
                deg[v] += 1
    
    return edges

def construct():
    """Main entry: Polarity graph init + greedy add + local search."""
    start_time = time.time()
    deadline = start_time + 115
    
    best_edges = None
    best_c4 = float('inf')
    
    def update_best(edges):
        nonlocal best_edges, best_c4
        if edges is None or len(edges) != M:
            return
        adj = make_adj(edges)
        c4 = count_c4(adj)
        if c4 < best_c4:
            best_c4 = c4
            best_edges = [list(e) for e in edges]

    # Strategy 1: Polarity Graph (90 edges) + Best 91st Edge
    pol_edges = construct_polarity_graph()
    if len(pol_edges) == 90:
        adj_pol = make_adj(pol_edges)
        c4_pol = count_c4(adj_pol)
        
        # Find the non-edge that adds the fewest C4s
        best_add_edge = None
        min_delta = float('inf')
        
        # We need to check all non-edges
        edge_set_pol = set((min(u, v), max(u, v)) for u, v in pol_edges)
        
        for i in range(N):
            for j in range(i + 1, N):
                e = (i, j)
                if e in edge_set_pol:
                    continue
                d = delta_add_fast(adj_pol, i, j)
                if d < min_delta:
                    min_delta = d
                    best_add_edge = [i, j]
        
        if best_add_edge is not None:
            candidate_edges = pol_edges + [best_add_edge]
            update_best(candidate_edges)
            
            # Run local search on this promising candidate
            if time.time() < deadline - 5:
                ls_edges = local_search([list(e) for e in candidate_edges], time_budget=60)
                update_best(ls_edges)

    # Strategy 2: Random restarts if time remains
    if time.time() < deadline - 10:
        for seed in range(100):
            if time.time() > deadline - 5:
                break
            edges = random_near_regular_init(seed)
            if len(edges) != M:
                continue
            update_best(edges)
            
            if time.time() < deadline - 5:
                # Only search if close to best
                adj = make_adj(edges)
                c4 = count_c4(adj)
                if c4 < best_c4 + 10:
                    ls_edges = local_search([list(e) for e in edges], time_budget=10)
                    update_best(ls_edges)

    if best_edges is None or len(best_edges) != M:
        # Fallback
        pairs = [[i, j] for i in range(N) for j in range(i+1, N)]
        best_edges = pairs[:M]
    
    return best_edges
