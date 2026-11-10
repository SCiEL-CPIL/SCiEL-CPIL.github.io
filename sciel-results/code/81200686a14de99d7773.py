"""K3,3-free graph on 20 vertices - polarized construction."""
import time
import random
from itertools import combinations

def _has_k33(adj, n=20):
    """Check if graph has a K3,3 subgraph."""
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                common = [x for x in range(n) if x != a and x != b and x != c
                          and adj[a][x] and adj[b][x] and adj[c][x]]
                if len(common) >= 3:
                    return True
    return False

def _edge_count(adj, n=20):
    return sum(sum(row) for row in adj) // 2

def _try_add_edge(adj, u, v, n=20):
    """Try to add edge (u,v), return True if K3,3-free maintained."""
    if u == v or adj[u][v]:
        return False
    adj[u][v] = adj[v][u] = 1
    if _has_k33(adj, n):
        adj[u][v] = adj[v][u] = 0
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    n = 20
    best_edges = None
    best_count = 0
    
    # Polarized construction: two parts A and B
    # Within each part: dense (complement of sparse)
    # Cross: sparse
    
    # Strategy: try different splits and within-part structures
    random.seed(42)
    
    for split in range(8, 13):  # A has 'split' vertices, B has 20-split
        a_size = split
        b_size = n - a_size
        
        # Try different within-part densities
        for a_density in [0.85, 0.90, 0.95]:
            for b_density in [0.85, 0.90, 0.95]:
                for cross_density in [0.15, 0.20, 0.25, 0.30]:
                    if time.time() > deadline:
                        break
                        
                    # Build initial graph
                    adj = [[0] * n for _ in range(n)]
                    
                    # Within A: keep edges with probability a_density
                    a_vertices = list(range(a_size))
                    for i in range(len(a_vertices)):
                        for j in range(i + 1, len(a_vertices)):
                            if random.random() < a_density:
                                adj[a_vertices[i]][a_vertices[j]] = 1
                                adj[a_vertices[j]][a_vertices[i]] = 1
                    
                    # Within B: keep edges with probability b_density
                    b_vertices = list(range(a_size, n))
                    for i in range(len(b_vertices)):
                        for j in range(i + 1, len(b_vertices)):
                            if random.random() < b_density:
                                adj[b_vertices[i]][b_vertices[j]] = 1
                                adj[b_vertices[j]][b_vertices[i]] = 1
                    
                    # Cross edges: keep with probability cross_density
                    for u in a_vertices:
                        for v in b_vertices:
                            if random.random() < cross_density:
                                adj[u][v] = adj[v][u] = 1
                    
                    # Check K3,3 and fix
                    if _has_k33(adj, n):
                        continue
                    
                    # Try to add more edges greedily
                    improved = True
                    while improved and time.time() < deadline:
                        improved = False
                        # Try adding random missing edges
                        for _ in range(200):
                            if time.time() > deadline:
                                break
                            u = random.randint(0, n - 1)
                            v = random.randint(0, n - 1)
                            if u != v and not adj[u][v]:
                                if _try_add_edge(adj, u, v, n):
                                    improved = True
                    
                    count = _edge_count(adj, n)
                    if count > best_count:
                        best_count = count
                        best_edges = [[u, v] for u in range(n) for v in range(u + 1, n) if adj[u][v]]
    
    # Also try the simple K2,18 as fallback
    if best_count < 36:
        best_edges = [[a, b] for a in range(2) for b in range(2, 20)]
        best_count = 36
    
    # If we found something good, do a final greedy improvement pass
    if best_edges is not None and time.time() < deadline - 5:
        adj = [[0] * n for _ in range(n)]
        for u, v in best_edges:
            adj[u][v] = adj[v][u] = 1
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not adj[u][v]:
                        if _try_add_edge(adj, u, v, n):
                            improved = True
        
        final_count = _edge_count(adj, n)
        if final_count > best_count:
            best_count = final_count
            best_edges = [[u, v] for u in range(n) for v in range(u + 1, n) if adj[u][v]]
    
    return best_edges
