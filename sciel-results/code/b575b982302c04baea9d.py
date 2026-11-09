import numpy as np
import time

def priority(adjacency):
    """
    Calculates priority scores for vertices to maximize the size of the independent set.
    Heuristic: 'Core Number' based.
    Vertices with lower core numbers are in the sparser periphery of the graph.
    Picking them first is analogous to the Smallest Degree First heuristic,
    which is known to perform well for greedy Maximum Independent Set.
    Score is primarily -core_number, with tie-breaking by -degree.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    A = np.array(adjacency, dtype=bool)
    n = A.shape[0]
    
    if n == 0:
        return np.array([])
        
    degrees = A.sum(axis=1).astype(int)
    max_deg = int(degrees.max())
    
    if max_deg == 0:
        # All isolated vertices
        return np.zeros(n)

    # Calculate Core Numbers
    # Algorithm: Peeling
    # k goes from 1 to max_deg
    # At step k, remove all vertices with current degree < k.
    # The core number of removed vertices is k-1.
    
    removed = np.zeros(n, dtype=bool)
    current_deg = degrees.copy()
    core_num = np.zeros(n, dtype=int)
    
    # Precompute neighbor lists for speed
    neighbors = [np.where(A[i])[0] for i in range(n)]
    
    for k in range(1, max_deg + 1):
        if time.time() > deadline:
            break
            
        # We need to remove all vertices with current_deg < k
        # This might cascade: removing a vertex reduces neighbor degrees,
        # which might drop them below k.
        # So we loop until no more vertices are removed.
        
        changed = True
        while changed:
            changed = False
            for v in range(n):
                if not removed[v] and current_deg[v] < k:
                    removed[v] = True
                    core_num[v] = k - 1
                    changed = True
                    # Update neighbors
                    for u in neighbors[v]:
                        if not removed[u]:
                            current_deg[u] -= 1
                            # Optimization: if current_deg[u] < k, it will be caught in next iteration of 'for v' or 'while changed'
                            # We don't need to check immediately, the 'while changed' loop handles stability.
        
    # Vertices not removed have core number max_deg (or potentially higher, but bounded by max_deg)
    # Actually, if they survive the max_deg step, their core number is max_deg.
    for v in range(n):
        if not removed[v]:
            core_num[v] = max_deg
            
    # Score: Lower core number is better.
    # Tie-break: Lower degree is better.
    scores = -core_num.astype(float) * 1000.0 - degrees.astype(float)
    
    return scores
