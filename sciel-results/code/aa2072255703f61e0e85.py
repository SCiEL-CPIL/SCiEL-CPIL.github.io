import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    def is_valid(n, edges):
        # Build adjacency matrix
        A = np.zeros((n, n), dtype=int)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
            
        # Check for triangles (C3)
        # If there is a triangle, we can't use the trace(A^5) trick easily.
        # But we can just check for C5 directly if we have to.
        # For now, let's assume we want C3-free graphs to use the trace trick.
        # If we find a triangle, we return False (invalid for our search heuristic).
        # This restricts the search space, but might be necessary for speed.
        
        # Check for triangles
        for u in range(n):
            for v in range(u+1, n):
                if A[u, v] == 1:
                    # Check common neighbors
                    common = np.dot(A[u], A[v])
                    if common > 0:
                        return False
        
        # Check for C5 using trace(A^5)
        A2 = A @ A
        A4 = A2 @ A2
        A5 = A4 @ A
        if np.trace(A5) > 0:
            return False
            
        # Check for Independent Set of size 9
        # Greedy check
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining = set(range(n))
        alpha = 0
        while remaining:
            min_deg_vertex = min(remaining, key=lambda v: len(adj[v] & remaining))
            alpha += 1
            remaining -= adj[min_deg_vertex]
            remaining.discard(min_deg_vertex)
            if alpha >= 9:
                return False
                
        return True

    best_n = 32
    best_edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    
    for target_n in [35, 34, 33]:
        if time.time() > deadline:
            break
            
        found = False
        while time.time() < deadline and not found:
            # Generate candidate
            edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
            
            for v in range(32, target_n):
                # Connect to random vertices
                # To avoid triangles, connect to independent sets?
                # Just random for now.
                for u in range(v):
                    if random.random() < 0.2:
                        edges.append([v, u])
                        
            if is_valid(target_n, edges):
                best_n = target_n
                best_edges = edges
                found = True
                break
                
        if found:
            break
            
    return {'n': best_n, 'edges': best_edges}
