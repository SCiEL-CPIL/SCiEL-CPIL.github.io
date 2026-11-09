import time
import random

def _is_triangle_free(adj_matrix, n):
    """
    Check if the graph defined by adj_matrix is triangle-free.
    adj_matrix is a list of lists of ints (0 or 1).
    """
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i][j]:
                for k in range(j + 1, n):
                    if adj_matrix[i][k] and adj_matrix[j][k]:
                        return False
    return True

def _is_3_colorable(adj_matrix, n):
    """
    Check if the graph is 3-colorable using backtracking.
    """
    colors = [-1] * n
    
    def backtrack(v):
        if v == n:
            return True
        for c in range(3):
            if all(not (adj_matrix[v][u] and colors[u] == c) for u in range(v)):
                colors[v] = c
                if backtrack(v + 1):
                    return True
                colors[v] = -1
        return False
    
    # Order vertices by degree (descending) for better pruning
    # For simplicity in this fast check, we stick to natural order or a static heuristic
    # Given n=16, even naive backtracking is fast if we prune well.
    
    # Heuristic: try vertices with highest degree first?
    # Let's just use the standard backtrack. 16 vertices is small.
    return backtrack(0)

def _count_edges(adj_matrix, n):
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i][j]:
                count += 1
    return count

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Start with Clebsch graph (40 edges, chi=4, triangle-free)
    # Vertices 0..15
    # Adjacency: u xor v in {1, 2, 4, 8, 15}
    S = {1, 2, 4, 8, 15}
    adj = [[0] * n for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if (u ^ v) in S:
                adj[u][v] = 1
                adj[v][u] = 1
                
    current_edges = 40
    
    # Try to add edges
    # Generate all potential edges not currently in the graph
    potential_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u][v]:
                potential_edges.append((u, v))
                
    # Shuffle to allow multiple restarts if needed, but let's try greedy first
    random.shuffle(potential_edges)
    
    # Greedy addition
    for u, v in potential_edges:
        if time.time() > deadline:
            break
        # Check if adding (u, v) creates a triangle
        # Triangle exists if there is a w such that (u,w) and (v,w) are edges
        creates_triangle = False
        for w in range(n):
            if w != u and w != v:
                if adj[u][w] and adj[v][w]:
                    creates_triangle = True
                    break
        
        if not creates_triangle:
            # Check if adding (u, v) makes it 3-colorable
            # Temporarily add edge
            adj[u][v] = 1
            adj[v][u] = 1
            
            if _is_3_colorable(adj, n):
                # Revert
                adj[u][v] = 0
                adj[v][u] = 0
            else:
                # Keep edge
                current_edges += 1
                
    # Local search: try to swap edges to increase count?
    # Or just try to add more if we missed some?
    # The greedy approach might get stuck.
    # Let's try a few random restarts if time permits.
    
    best_edges = current_edges
    best_adj = [row[:] for row in adj]
    
    # Try to improve with random restarts of the greedy process
    # Since n=16 is small, we can do many restarts.
    while time.time() < deadline - 5:
        # Reset to Clebsch
        adj = [[0] * n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if (u ^ v) in S:
                    adj[u][v] = 1
                    adj[v][u] = 1
        
        current_edges = 40
        
        # Random order of potential edges
        pe = []
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u][v]:
                    pe.append((u, v))
        random.shuffle(pe)
        
        for u, v in pe:
            if time.time() > deadline - 5:
                break
            creates_triangle = False
            for w in range(n):
                if w != u and w != v:
                    if adj[u][w] and adj[v][w]:
                        creates_triangle = True
                        break
            
            if not creates_triangle:
                adj[u][v] = 1
                adj[v][u] = 1
                if _is_3_colorable(adj, n):
                    adj[u][v] = 0
                    adj[v][u] = 0
                else:
                    current_edges += 1
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [row[:] for row in adj]
            
    # Convert best_adj to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u][v]:
                edges.append([u, v])
                
    return edges
