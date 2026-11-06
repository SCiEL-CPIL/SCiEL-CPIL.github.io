import numpy as np
import time
import random

def construct():
    n = 40
    m = 401
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initialize random graph
    A = np.zeros((n, n), dtype=np.int8)
    edges = set()
    
    # Generate random edges
    while len(edges) < m:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u != v:
            if u > v: u, v = v, u
            if (u, v) not in edges:
                edges.add((u, v))
                A[u, v] = 1
                A[v, u] = 1
                
    all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
    edge_set = set(edges)
    non_edges = [e for e in all_possible if e not in edge_set]
    
    def get_score(A, edge_set):
        if not edge_set: return 0
        M = A @ A.T
        # Create mask
        # Instead of building mask from set every time, we can just iterate?
        # No, vectorized is better.
        mask = np.zeros((n, n), dtype=bool)
        for u, v in edge_set:
            mask[u, v] = True
        vals = M[mask]
        return int(np.max(vals)) if len(vals) > 0 else 0

    current_score = get_score(A, edge_set)
    best_score = current_score
    best_edges = set(edge_set)
    
    # Local Search
    while time.time() < deadline:
        if not edges or not non_edges:
            break
            
        u1, v1 = random.choice(list(edge_set))
        u2, v2 = random.choice(non_edges)
        
        # Perform Swap
        # Remove (u1, v1)
        A[u1, v1] = 0
        A[v1, u1] = 0
        edge_set.remove((u1, v1))
        non_edges.append((u1, v1))
        
        # Add (u2, v2)
        A[u2, v2] = 1
        A[v2, u2] = 1
        edge_set.add((u2, v2))
        # Remove (u2, v2) from non_edges
        # Since non_edges is a list, finding and removing is O(N). N=379.
        # This is acceptable.
        idx = non_edges.index((u2, v2))
        non_edges[idx] = non_edges[-1]
        non_edges.pop()
        
        new_score = get_score(A, edge_set)
        
        if new_score <= current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_edges = set(edge_set)
        else:
            # Revert
            A[u1, v1] = 1
            A[v1, u1] = 1
            edge_set.add((u1, v1))
            non_edges.remove((u1, v1))
            
            A[u2, v2] = 0
            A[v2, u2] = 0
            edge_set.remove((u2, v2))
            non_edges.append((u2, v2))

    return [[u, v] for u, v in best_edges]
