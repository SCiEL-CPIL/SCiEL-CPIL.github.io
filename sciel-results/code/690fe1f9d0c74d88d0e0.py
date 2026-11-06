"""K2,3-free graph construction on 20 vertices; maximize edge count."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n = 20
    total_edges_possible = n * (n - 1) // 2  # 190

    # Precompute all edge pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {}
    for idx, (u, v) in enumerate(all_pairs):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx

    def common_neighbors_count(adj, u, v):
        """Count common neighbors of u and v (excluding each other if adjacent)."""
        # Common neighbors are w such that adj[u,w] and adj[v,w] and w != u and w != v
        # Since adj[u,u] = 0 and adj[v,v] = 0, and u != v:
        # If u and v are adjacent, u is not in N(u) but v is in N(u), and u is in N(v) but not in N(v)
        # Actually N(u) = {w : adj[u,w]=1, w != u}. Since adj[u,u]=0, N(u) = {w : adj[u,w]=1}
        # Common neighbors: w where adj[u,w]=1 and adj[v,w]=1
        # But we need w != u and w != v.
        # Since adj[u,u]=0, u won't be counted as neighbor of u. Similarly for v.
        # But if adj[u,v]=1, then v is in N(u) and u is in N(v).
        # However, v is not in N(v) (since adj[v,v]=0), so v won't be a common neighbor.
        # Similarly u won't be a common neighbor.
        # So just np.sum(adj[u] & adj[v]) works.
        return np.sum(adj[u] & adj[v])

    def is_k23_free(adj):
        """Check if graph is K2,3-free: no two vertices have >= 3 common neighbors."""
        for i in range(n):
            for j in range(i + 1, n):
                if np.sum(adj[i] & adj[j]) >= 3:
                    return False
        return True

    def count_common_neighbors_matrix(adj):
        """Return matrix C where C[i,j] = |N(i) ∩ N(j)| for i != j."""
        C = adj @ adj
        # C[i,j] for i!=j is the number of common neighbors
        # For i==j, C[i,i] = degree(i), which we don't care about
        return C

    def get_violation_count(C):
        """Count pairs (i,j) with i < j and C[i,j] >= 3."""
        # Only look at upper triangle
        viol = 0
        for i in range(n):
            for j in range(i + 1, n):
                if C[i, j] >= 3:
                    viol += 1
        return viol

    def can_add_edge(adj, C, u, v):
        """Check if adding edge (u,v) maintains K2,3-freeness."""
        if adj[u, v]:
            return True, C  # Already present

        # Check 1: common neighbors of u and v must be <= 2 (after adding, it stays the same for w != u,v)
        # Actually, adding (u,v) doesn't change N(u) ∩ N(v) in terms of other vertices.
        # But u becomes a neighbor of v, and v becomes a neighbor of u.
        # So the new common neighbors of u and v are the old common neighbors (w != u, v).
        # So we need old common neighbors <= 2.
        cuv = C[u, v]
        if cuv >= 3:
            return False, C

        # Check 2: for each w in N(v), w != u, the new common neighbors of u and w
        # After adding (u,v), v becomes a common neighbor of u and w (since w is in N(v) and now v is in N(u)).
        # So new C[u,w] = old C[u,w] + 1 (if v was not already a common neighbor of u and w, which it wasn't since adj[u,v]=0)
        # We need new C[u,w] <= 2, so old C[u,w] <= 1.
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v, w]:  # w is in N(v)
                if C[u, w] >= 2:
                    return False, C

        # Check 3: for each w in N(u), w != v, the new common neighbors of v and w
        # Similarly, u becomes a common neighbor of v and w.
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[u, w]:  # w is in N(u)
                if C[v, w] >= 2:
                    return False, C

        return True, C

    def greedy_construct(adj, C, max_time):
        """Deterministic greedy: repeatedly add the edge with minimum common neighbors that is valid."""
        while time.time() < max_time:
            best_edge = None
            best_cost = float('inf')
            
            for u, v in all_pairs:
                if adj[u, v]:
                    continue
                valid, _ = can_add_edge(adj, C, u, v)
                if valid:
                    cost = C[u, v]  # Current common neighbors
                    if cost < best_cost or (cost == best_cost and (u, v) < best_edge):
                        best_cost = cost
                        best_edge = (u, v)
            
            if best_edge is None:
                break
            
            u, v = best_edge
            adj[u, v] = True
            adj[v, u] = True
            C = count_common_neighbors_matrix(adj)
        
        return adj, C

    def local_search_2swap(adj, C, max_time):
        """Try removing one edge and adding another (or two) to improve edge count."""
        best_adj = adj.copy()
        best_C = C.copy()
        best_count = np.sum(adj) // 2
        
        while time.time() < max_time:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    if time.time() > max_time:
                        break
                    if not adj[i, j]:
                        continue
                    
                    # Try removing edge (i, j)
                    temp_adj = adj.copy()
                    temp_adj[i, j] = False
                    temp_adj[j, i] = False
                    temp_C = count_common_neighbors_matrix(temp_adj)
                    
                    # Now try to greedily add edges to temp_adj
                    # First, try to add back (i, j) if it helps? No, we want net gain.
                    # Try to add any valid edges to temp_adj
                    for u, v in all_pairs:
                        if time.time() > max_time:
                            break
                        if temp_adj[u, v]:
                            continue
                        valid, _ = can_add_edge(temp_adj, temp_C, u, v)
                        if valid:
                            temp_adj[u, v] = True
                            temp_adj[v, u] = True
                            temp_C = count_common_neighbors_matrix(temp_adj)
                    
                    temp_count = np.sum(temp_adj) // 2
                    if temp_count > best_count:
                        best_adj = temp_adj
                        best_C = temp_C
                        best_count = temp_count
                        improved = True
                        break
                if improved or time.time() > max_time:
                    break
            
            if not improved:
                break
        
        return best_adj, best_C, best_count

    # Main construction
    adj = np.zeros((n, n), dtype=bool)
    C = count_common_neighbors_matrix(adj)
    
    # Phase 1: Greedy construction
    adj, C = greedy_construct(adj, C, deadline - 10)
    
    # Phase 2: Local search with 1-edge removal and refill
    adj, C, count = local_search_2swap(adj, C, deadline - 2)
    
    # Extract edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append([i, j])
    
    return edges

# EVOLVE-BLOCK-END
