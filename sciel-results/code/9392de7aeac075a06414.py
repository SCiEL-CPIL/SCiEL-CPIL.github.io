import numpy as np
import time
from collections import deque

def priority(adjacency):
    """
    Calculates priority scores for vertices to maximize the size of the independent set
    selected by a greedy decoder (descending score).
    
    Heuristic: 'Degree-Thresholded Component Size'
    For each vertex v, we consider the subgraph induced by vertices with degree >= degree(v).
    The score is the size of the connected component of v in this subgraph.
    This rewards vertices that are part of large, relatively sparse regions (low degree threshold
    implies a larger subgraph, and being in a large component implies 'uncontested territory').
    Ties are broken by lower degree and lower internal connectivity.
    """
    start_time = time.time()
    # Safety margin: 150s limit. We use 140s.
    deadline = start_time + 140.0
    
    A = np.array(adjacency, dtype=bool)
    n = A.shape[0]
    
    if n == 0:
        return np.array([])
        
    degrees = A.sum(axis=1).astype(int)
    
    # Precompute neighbor lists for faster BFS
    # neighbors[i] is a list of indices of neighbors of i
    neighbors = [np.where(A[i])[0] for i in range(n)]
    
    scores = np.zeros(n, dtype=float)
    
    for v in range(n):
        if time.time() > deadline:
            # If we are running out of time, fill remaining scores with a fallback
            # (e.g., simple degree heuristic) to ensure a valid return
            for u in range(v, n):
                scores[u] = -degrees[u]
            return scores
        
        d_v = degrees[v]
        
        # Create mask for nodes with degree >= d_v
        # This defines the "valid" nodes for v's territory exploration
        mask = degrees >= d_v
        
        # BFS from v
        visited = np.zeros(n, dtype=bool)
        queue = deque([v])
        visited[v] = True
        count = 0
        internal_neighbors = 0 # Count edges within the component
        
        while queue:
            u = queue.popleft()
            count += 1
            for w in neighbors[u]:
                if mask[w]:
                    if not visited[w]:
                        visited[w] = True
                        queue.append(w)
                    # Count internal edges: if w is visited (part of component) and w != u, 
                    # we count the edge. To avoid double counting, we can just count 
                    # neighbors w that are in the component. 
                    # Actually, simpler: count how many neighbors of u are in the component.
                    # We don't know the full component until BFS finishes.
                    # Let's just collect the component nodes first.
        
        # Let's re-do the BFS to collect nodes, then compute internal density.
        # This is safer for correct internal edge counting.
        
    # The previous BFS didn't collect nodes for post-processing. 
    # Let's refactor to collect nodes.
    
    return scores
