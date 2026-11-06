import time
import numpy as np

def _count_c4s(adj, n):
    """Count C4s using common neighbors."""
    total = 0
    for u in range(n):
        for v in range(u + 1, n):
            common = len(adj[u] & adj[v])
            if common >= 2:
                total += common * (common - 1) // 2
    return total

def _count_c4s_fast(adj_matrix):
    """Fast C4 count using numpy matrix multiplication."""
    n = adj_matrix.shape[0]
    A2 = adj_matrix @ adj_matrix
    total = 0
    for u in range(n):
        for v in range(u + 1, n):
            c = int(A2[u, v])
            if c >= 2:
                total += c * (c - 1) // 2
    return total // 2

def construct():
    """
    Returns a list of 91 edges [u, v] on vertices 0..30 minimizing the number of C4s.
    Uses the polarity graph of order 5 as a seed (90 edges, 0 C4s) and adds
    the single edge that creates the fewest C4s, then performs local search.
    """
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    q = 5
    # Generate vertices of PG(2, q)
    vertices = []
    for i in range(q):
        for j in range(q):
            vertices.append((1, i, j))
    for k in range(q):
        vertices.append((0, 1, k))
    vertices.append((0, 0, 1))
    
    n = len(vertices)  # 31
    
    # Build adjacency list for the polarity graph
    adj = [set() for _ in range(n)]
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            u = vertices[i]
            v = vertices[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
    
    # We have 90 edges. Find the non-edge that adds the fewest C4s.
    best_cost = float('inf')
    best_edge = None
    
    for u in range(n):
        for v in range(u + 1, n):
            if v in adj[u]:
                continue
            A = adj[u]
            B = adj[v]
            cost = 0
            for x in A:
                cost += len(adj[x] & B)
            if cost < best_cost:
                best_cost = cost
                best_edge = [u, v]
    
    edges.append(best_edge)
    adj[best_edge[0]].add(best_edge[1])
    adj[best_edge[1]].add(best_edge[0])
    
    # Build numpy adjacency matrix
    def build_adj_matrix(edge_list):
        mat = np.zeros((n, n), dtype=np.int64)
        for u, v in edge_list:
            mat[u, v] = 1
            mat[v, u] = 1
        return mat
    
    adj_mat = build_adj_matrix(edges)
    current_c4 = _count_c4s_fast(adj_mat)
    best_c4 = current_c4
    best_edges = list(edges)
    
    # Build set of current edges for O(1) lookup
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    # Precompute all non-edges
    all_non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in edge_set]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try all edge swaps: for each current edge (a,b), for each non-edge (c,d),
        # check if removing (a,b) and adding (c,d) reduces C4 count
        best_delta = 0  # We want negative delta (reduction)
        best_swap = None
        
        # Incremental C4 counting for speed
        # For edge swap (remove a-b, add c-d):
        # Delta = C4s_gained(c,d) - C4s_lost(a,b) + interaction_terms
        
        for ei, (a, b) in enumerate(edges):
            if time.time() > deadline:
                break
            
            # Count C4s lost by removing (a,b):
            # C4s containing edge (a,b): paths a-x-y-b where x~a, y~b, x~y, x!=y, x!=b, y!=a
            lost = 0
            for x in adj[a]:
                if x == b:
                    continue
                for y in adj[b]:
                    if y == a or y == x:
                        continue
                    if x in adj[y]:
                        lost += 1
            
            # Try each non-edge (c, d)
            for c, d in all_non_edges:
                if c == a or c == b or d == a or d == b:
                    continue
                
                # Count C4s gained by adding (c,d):
                # C4s containing edge (c,d): paths c-x-y-d where x~c, y~d, x~y, x!=y, x!=d, y!=c
                # But we need to account for the fact that (a,b) is removed
                gained = 0
                for x in adj[c]:
                    if x == d:
                        continue
                    for y in adj[d]:
                        if y == c or y == x:
                            continue
                        if x in adj[y]:
                            # Check if this C4 uses edge (a,b)
                            # The C4 is c-x-y-d. It uses (a,b) only if {a,b} ⊆ {c,x,y,d}
                            # If it uses (a,b), the actual gain is less
                            c4_verts = {c, x, y, d}
                            if a in c4_verts and b in c4_verts:
                                # This C4 would be destroyed by removing (a,b), so it's not really gained
                                continue
                            gained += 1
                
                delta = gained - lost
                if delta < best_delta:
                    best_delta = delta
                    best_swap = (ei, c, d)
        
        if best_swap is not None and best_delta < 0:
            ei, c, d = best_swap
            a, b = edges[ei]
            # Remove edge (a,b)
            edges[ei] = [c, d]
            adj[a].remove(b)
            adj[b].remove(a)
            adj[c].add(d)
            adj[d].add(c)
            edge_set.discard((a, b) if a < b else (b, a))
            edge_set.add((c, d) if c < d else (d, c))
            
            # Update numpy matrix
            adj_mat[a, b] = 0
            adj_mat[b, a] = 0
            adj_mat[c, d] = 1
            adj_mat[d, c] = 1
            
            # Update non-edges list
            all_non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in edge_set]
            
            current_c4 = _count_c4s_fast(adj_mat)
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edges = list(edges)
                improved = True
    
    # Ensure exactly 91 edges
    assert len(best_edges) == 91, f"Expected 91 edges, got {len(best_edges)}"
    assert len(set(tuple(sorted(e)) for e in best_edges)) == 91, "Edges are not distinct"
    
    return best_edges
