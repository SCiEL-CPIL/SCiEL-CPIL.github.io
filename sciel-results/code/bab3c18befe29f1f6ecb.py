"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    q = 5
    n = 31
    target_edges = 91
    
    # Enumerate points of PG(2,5)
    points = []
    for x in range(q):
        for y in range(q):
            points.append([x, y, 1])
    for t in range(q):
        points.append([1, t, 0])
    points.append([0, 1, 0])
    
    # Build polarity graph: edge if dot product is 0 mod q
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            dot = (points[i][0]*points[j][0] + points[i][1]*points[j][1] + points[i][2]*points[j][2]) % q
            if dot == 0:
                edges.append([i, j])
    
    # Build adjacency as numpy boolean matrix for fast C4 counting
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    def count_c4(adj_matrix):
        """Count C4 subgraphs using the formula: 1/2 * sum_{u<v} C(common(u,v), 2)"""
        comm = adj_matrix @ adj_matrix
        total = 0
        for u in range(n - 1):
            for v in range(u + 1, n):
                c = comm[u, v]
                if c >= 2:
                    total += c * (c - 1) // 2
        return total // 2
    
    def c4_change_add(adj_matrix, u, v):
        """Number of new C4s when adding edge (u,v)."""
        common = int(np.dot(adj_matrix[u], adj_matrix[v]))
        return common * (common - 1) // 2
    
    def c4_change_remove(adj_matrix, u, v):
        """Number of C4s destroyed when removing edge (u,v)."""
        # Temporarily remove the edge to count common neighbors without u,v
        adj_matrix[u, v] = 0
        adj_matrix[v, u] = 0
        common = int(np.dot(adj_matrix[u], adj_matrix[v]))
        adj_matrix[u, v] = 1
        adj_matrix[v, u] = 1
        return common * (common - 1) // 2
    
    # Adjust to exactly 91 edges
    if len(edges) > 91:
        excess = len(edges) - 91
        # Remove edges that contribute most to C4 count
        for _ in range(excess):
            if time.time() > deadline:
                break
            best_idx = -1
            best_cost = -1
            for idx in range(len(edges)):
                u, v = edges[idx]
                cost = c4_change_remove(adj, u, v)
                if best_idx == -1 or cost > best_cost:
                    best_cost = cost
                    best_idx = idx
            if best_idx >= 0:
                u, v = edges.pop(best_idx)
                adj[u, v] = 0
                adj[v, u] = 0
    
    elif len(edges) < 91:
        deficit = 91 - len(edges)
        random.seed(42)
        for _ in range(deficit):
            if time.time() > deadline:
                break
            best_add = None
            best_cost = float('inf')
            for attempt in range(5000):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u < v and adj[u, v] == 0:
                    cost = c4_change_add(adj, u, v)
                    if cost < best_cost:
                        best_cost = cost
                        best_add = (u, v)
            if best_add:
                u, v = best_add
                edges.append([u, v])
                adj[u, v] = 1
                adj[v, u] = 1
    
    # Local search: try edge swaps to reduce C4
    current_c4 = count_c4(adj)
    random.seed(123)
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        
        # Try multiple random swap attempts
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            # Pick a random edge to remove
            if not edges:
                break
            idx = random.randint(0, len(edges) - 1)
            u, v = edges[idx]
            
            # Calculate C4s destroyed by removing (u,v)
            c4_removed = c4_change_remove(adj, u, v)
            
            # Temporarily remove
            adj[u, v] = 0
            adj[v, u] = 0
            
            # Find best edge to add
            best_add = None
            best_net_change = float('inf')
            
            # Sample candidate edges to add
            for _ in range(300):
                a = random.randint(0, n - 1)
                b = random.randint(0, n - 1)
                if a < b and adj[a, b] == 0:
                    c4_added = c4_change_add(adj, a, b)
                    net_change = c4_added - c4_removed
                    if net_change < best_net_change:
                        best_net_change = net_change
                        best_add = (a, b)
            
            if best_add and best_net_change < 0:
                a, b = best_add
                edges.append([a, b])
                adj[a, b] = 1
                adj[b, a] = 1
                current_c4 += best_net_change
                improved = True
                break
            else:
                # Restore removed edge
                adj[u, v] = 1
                adj[v, u] = 1
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
    
    return edges
# EVOLVE-BLOCK-END
