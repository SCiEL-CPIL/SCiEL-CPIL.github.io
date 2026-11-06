"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s out of 150s limit
    
    # Build the polarity graph of PG(2, 5)
    # Points in PG(2, 5)
    points = []
    # z = 1: (x, y, 1) for x, y in F_5
    for x in range(5):
        for y in range(5):
            points.append((x, y, 1))
    # z = 0: (x, y, 0) with (x, y) != (0, 0), normalized
    # x != 0: (1, y, 0) for y in F_5
    for y in range(5):
        points.append((1, y, 0))
    # x = 0: (0, 1, 0)
    points.append((0, 1, 0))
    
    n = len(points)  # should be 31
    assert n == 31, f"Expected 31 points, got {n}"
    
    # Polarity matrix M
    # u^T M v = x1*z2 + y1*y2 + z1*x2 (mod 5)
    def dot_product(u, v):
        x1, y1, z1 = u
        x2, y2, z2 = v
        return (x1 * z2 + y1 * y2 + z1 * x2) % 5
    
    # Build adjacency matrix
    adj = [[0]*n for _ in range(n)]
    edges = []
    
    for i in range(n):
        for j in range(i+1, n):
            if dot_product(points[i], points[j]) == 0:
                adj[i][j] = 1
                adj[j][i] = 1
                edges.append([i, j])
    
    base_edge_count = len(edges)
    # The polarity graph of PG(2, q) for odd q has q^2+q+1 vertices
    # and (q^2+q+1)(q+1)/2 edges... wait, let me think about this.
    # Actually, for a polarity graph of PG(2, q), each vertex is adjacent to q+1 vertices
    # (the absolute points on its polar line). So total edges = n*(q+1)/2 = 31*6/2 = 93.
    # But wait, that doesn't match. Let me reconsider.
    # Actually, in a polarity graph, vertex v is adjacent to w iff v is on the polar line of w.
    # The number of absolute points on a line is q+1 = 6.
    # So each vertex has degree q+1 = 6.
    # Total edges = 31 * 6 / 2 = 93.
    # Hmm, but ex(31; C4) = 90, so we need 91 edges.
    # Let me just compute what we get.
    
    if base_edge_count >= 91:
        # We have too many edges, need to remove some
        # But first, let's see if we can find a subset of 91 edges with few C4s
        # For now, just return 91 edges and hope for the best
        # Actually, we need to be more careful
        # Let's try to find 91 edges with minimum C4s
        # Start with all edges, try removing edges to reduce C4 count
        pass
    
    # For the polarity graph, let's just take the first 91 edges
    # But we need to minimize C4s
    # Let me try a different approach: start with the polarity graph and if it has
    # more than 91 edges, remove edges to minimize C4s
    
    if base_edge_count > 91:
        # Need to remove (base_edge_count - 91) edges
        # Remove edges that are in the most C4s
        def count_c4s(adj_matrix, n_vertices):
            """Count the number of 4-cycles in the graph"""
            c4_count = 0
            for a in range(n_vertices):
                for b in range(a+1, n_vertices):
                    if adj_matrix[a][b] == 0:
                        continue
                    common = 0
                    for c in range(b+1, n_vertices):
                        if adj_matrix[b][c] == 0:
                            continue
                        for d in range(c+1, n_vertices):
                            if adj_matrix[c][d] == 0:
                                continue
                            if adj_matrix[d][a] == 1:
                                common += 1
                    c4_count += common
            return c4_count // 4  # Each C4 counted 4 times
        
        # Greedy edge removal
        current_adj = [row[:] for row in adj]
        current_edges = list(edges)
        current_c4 = count_c4s(current_adj, n)
        
        edges_to_remove = base_edge_count - 91
        for _ in range(edges_to_remove):
            if time.time() > deadline - 10:
                break
            
            # Find edge with maximum participation in C4s
            best_edge = None
            best_c4_increase = -1
            for edge_idx, (u, v) in enumerate(current_edges):
                # Remove edge and count C4s
                current_adj[u][v] = 0
                current_adj[v][u] = 0
                new_c4 = count_c4s(current_adj, n)
                c4_decrease = current_c4 - new_c4
                current_adj[u][v] = 1
                current_adj[v][u] = 1
                
                if c4_decrease > best_c4_increase:
                    best_c4_increase = c4_decrease
                    best_edge = edge_idx
            
            if best_edge is not None:
                u, v = current_edges[best_edge]
                current_adj[u][v] = 0
                current_adj[v][u] = 0
                current_edges.pop(best_edge)
                current_c4 -= best_c4_increase
    
    elif base_edge_count < 91:
        # Need to add edges
        current_adj = [row[:] for row in adj]
        current_edges = list(edges)
        current_c4 = count_c4s(current_adj, n)
        
        edges_to_add = 91 - base_edge_count
        for _ in range(edges_to_add):
            if time.time() > deadline - 10:
                break
            
            # Find non-edge that adds fewest C4s
            best_edge = None
            best_c4_increase = float('inf')
            for u in range(n):
                for v in range(u+1, n):
                    if current_adj[u][v] == 1:
                        continue
                    # Add edge and count C4s
                    current_adj[u][v] = 1
                    current_adj[v][u] = 1
                    new_c4 = count_c4s(current_adj, n)
                    c4_increase = new_c4 - current_c4
                    current_adj[u][v] = 0
                    current_adj[v][u] = 0
                    
                    if c4_increase < best_c4_increase:
                        best_c4_increase = c4_increase
                        best_edge = (u, v)
            
            if best_edge is not None:
                u, v = best_edge
                current_adj[u][v] = 1
                current_adj[v][u] = 1
                current_edges.append([u, v])
                current_c4 += best_c4_increase
    
    else:
        current_adj = [row[:] for row in adj]
        current_edges = list(edges)
    
    return current_edges[:91]
# EVOLVE-BLOCK-END
