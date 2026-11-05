import time
import random
import sys

def construct():
    n = 16
    deadline = time.time() + 100  # Safety margin before 120s limit
    
    # Helper: check if graph is triangle-free given adjacency matrix
    def is_triangle_free(adj):
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    for k in range(j + 1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        return True
    
    # Helper: check if graph is 3-colorable using backtracking
    def is_3_colorable(adj):
        color = [-1] * n
        
        def backtrack(v):
            if v == n:
                return True
            for c in range(3):
                ok = True
                for u in range(n):
                    if adj[v][u] and color[u] == c:
                        ok = False
                        break
                if ok:
                    color[v] = c
                    if backtrack(v + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)
    
    # Start with Clebsch graph
    def clebsch_adj():
        vertices = []
        for i in range(32):
            if bin(i).count('1') % 2 == 0:
                vertices.append(i)
        adj = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                xor_val = vertices[i] ^ vertices[j]
                if bin(xor_val).count('1') == 4:
                    adj[i][j] = 1
                    adj[j][i] = 1
        return adj
    
    adj = clebsch_adj()
    
    # Count edges
    def count_edges(adj):
        return sum(sum(row) for row in adj) // 2
    
    best_adj = [row[:] for row in adj]
    best_edges = count_edges(adj)
    
    # Verify initial
    if not is_triangle_free(adj) or is_3_colorable(adj):
        # Fallback: return Clebsch edges directly
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges
    
    # Simulated annealing
    temp = 10.0
    cooling = 0.995
    min_temp = 0.1
    
    random.seed(42)
    
    while time.time() < deadline and temp > min_temp:
        # Pick a random non-edge to try adding, or random edge to try removing
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        i, j = random.choice(all_pairs)
        
        if adj[i][j] == 0:
            # Try adding edge (i, j)
            # Check if it creates a triangle
            creates_triangle = False
            for k in range(n):
                if k != i and k != j and adj[i][k] and adj[j][k]:
                    creates_triangle = True
                    break
            
            if not creates_triangle:
                # Add edge
                adj[i][j] = 1
                adj[j][i] = 1
                new_edges = count_edges(adj)
                
                # Check 3-colorability
                colorable = is_3_colorable(adj)
                
                if not colorable:
                    # Good move: triangle-free and not 3-colorable
                    if new_edges > best_edges:
                        best_edges = new_edges
                        best_adj = [row[:] for row in adj]
                    # Accept (always accept improving or equal)
                else:
                    # Becomes 3-colorable: accept with probability
                    delta = -new_edges  # We lose the "not 3-colorable" property
                    # Use a penalty: if it becomes 3-colorable, reject unless we can recover
                    # Actually, let's reject this move since we need non-3-colorable
                    adj[i][j] = 0
                    adj[j][i] = 0
            # If creates triangle, reject
        else:
            # Try removing edge (i, j)
            adj[i][j] = 0
            adj[j][i] = 0
            new_edges = count_edges(adj)
            
            # Check if still triangle-free (should be) and 3-colorable status
            colorable = is_3_colorable(adj)
            
            if not colorable:
                # Still valid but fewer edges
                delta = new_edges - best_edges
                if delta >= 0 or random.random() < pow(2.718281828, delta / temp):
                    # Accept
                    pass
                else:
                    # Reject
                    adj[i][j] = 1
                    adj[j][i] = 1
            else:
                # Now 3-colorable, reject
                adj[i][j] = 1
                adj[j][i] = 1
        
        temp *= cooling
    
    # Extract best edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    
    return edges
