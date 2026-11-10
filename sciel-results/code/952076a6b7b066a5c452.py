import time
import numpy as np

_START = time.time()
_DEADLINE = 110.0          # safety margin before the 150 s wall-clock kill

def construct():
    N = 40
    best_edges = 0
    best_A = None
    
    # List of all possible edges
    all_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            all_edges.append((u, v))
            
    # Use a fixed seed for reproducibility in case the evaluator runs multiple times?
    # Or random? 
    # "randomized local search" suggests randomness is good.
    # I will not fix the seed to allow exploring different landscapes in different runs if needed, 
    # but typically a single run should be robust.
    
    while time.time() < _DEADLINE:
        A = np.zeros((N, N), dtype=np.int64)
        A2 = np.zeros((N, N), dtype=np.int64)
        m = 0
        
        # Shuffle edges
        np.random.shuffle(all_edges)
        
        for u, v in all_edges:
            # Check if adding (u,v) creates a C6
            # This happens if there is a path of length 4 between u and v in current graph
            # Number of such paths is dot(A2[u, :], A2[v, :])
            if np.dot(A2[u, :], A2[v, :]) == 0:
                # Add edge
                A[u, v] = 1
                A[v, u] = 1
                m += 1
                
                # Update A2
                # Only rows/cols u and v change
                # Recompute them using current A
                A2[u, :] = A[u, :] @ A
                A2[v, :] = A[v, :] @ A
                A2[:, u] = A @ A[:, u]
                A2[:, v] = A @ A[:, v]
                
        if m > best_edges:
            best_edges = m
            best_A = A.copy()
            
    if best_A is None:
        return []
        
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_A[u, v]:
                edges.append([u, v])
    return edges

# Self-check (never executed by the evaluator; keeps the module importable).
if __name__ == "__main__":
    e = construct()
    print("edges:", len(e))
