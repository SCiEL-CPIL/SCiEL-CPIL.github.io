import numpy as np
import time
import random

def construct():
    n = 16
    target_edges = 65
    
    # Precompute all possible triangles (triples of vertices)
    triangles = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triangles.append((i, j, k))
                
    def get_score(A, deg):
        max_sum = 0
        # A is numpy boolean array n x n
        # deg is numpy array n
        for i, j, k in triangles:
            if A[i, j] and A[j, k] and A[k, i]:
                s = deg[i] + deg[j] + deg[k]
                if s > max_sum:
                    max_sum = s
        return max_sum

    def random_graph():
        A = np.zeros((n, n), dtype=bool)
        edges = []
        # Generate 65 random edges
        while len(edges) < target_edges:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and not A[u, v]:
                A[u, v] = True
                A[v, u] = True
                edges.append((u, v))
        return A, edges

    def initial_construction():
        # K_{8,8} + 1 edge
        A = np.zeros((n, n), dtype=bool)
        edges = []
        for u in range(8):
            for v in range(8, 16):
                A[u, v] = True
                A[v, u] = True
                edges.append((u, v))
        # Add one edge in part A
        u, v = 0, 1
        A[u, v] = True
        A[v, u] = True
        edges.append((u, v))
        return A, edges

    best_score = float('inf')
    best_A = None
    best_edges = None
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    # Run multiple SA instances
    # We will run SA for a certain number of steps or time
    # Let's do 5 restarts
    
    for restart in range(5):
        if restart == 0:
            A, edges = initial_construction()
        else:
            A, edges = random_graph()
            
        deg = A.sum(axis=1).astype(int)
        current_score = get_score(A, deg)
        
        if current_score < best_score:
            best_score = current_score
            best_A = A.copy()
            best_edges = edges.copy()
            
        T = 5.0
        T_min = 0.1
        cooling = 0.999
        
        steps = 0
        while time.time() - start_time < time_limit:
            # Pick a random edge to remove
            if not edges: break
            idx_remove = random.randint(0, len(edges)-1)
            u, v = edges[idx_remove]
            
            # Pick a random non-edge to add
            # We need to find a pair (x, y) not in edges
            # Simple strategy: pick random x, y until valid
            while True:
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y: continue
                if x > y: x, y = y, x
                # Check if (x, y) is in edges
                # Using a set for O(1) lookup is faster than scanning list
                # But edges list is small (65). Scanning is O(65).
                # Let's maintain a set of edges for lookup.
                # For simplicity in this thought block, I'll assume we use a set in the actual code.
                pass
            
            # ... Implementation details in final code ...
            pass
            
    # Return best found
    if best_edges is None:
        # Fallback to initial
        A, edges = initial_construction()
        best_edges = edges
    return {'n': n, 'edges': [[u, v] for u, v in best_edges]}

