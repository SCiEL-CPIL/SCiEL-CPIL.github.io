import time
import random
import numpy as np
from itertools import combinations

def _is_valid_graph(adj, n):
    """Check if no subset of size >= 6 induces a regular subgraph.
    Uses vectorized numpy for speed."""
    # Precompute all subset masks of size 6 to n
    # For n=20, C(20,6) = 38760, C(20,7) = 77520, etc.
    # Total subsets: sum(C(20,k) for k=6..20) ~ 100k+
    # For each subset, check if induced subgraph is regular
    
    # Strategy: For each subset S of size k >= 6, compute degrees of vertices in S
    # within the induced subgraph, and check if all equal.
    
    # Vectorized approach:
    # For each subset, we need adj[S][:, S] and check if row sums are all equal.
    
    # Precompute all subsets as index arrays
    all_subsets = []
    for k in range(6, n + 1):
        for subset in combinations(range(n), k):
            all_subsets.append(np.array(subset))
    
    if not all_subsets:
        return True
    
    # For each subset, check regularity
    # adj is n x n boolean matrix
    for subset in all_subsets:
        k = len(subset)
        # Get induced subgraph
        sub_adj = adj[np.ix_(subset, subset)]
        # Compute degrees (row sums, since simple graph, no self-loops)
        degrees = sub_adj.sum(axis=1)
        # Check if all degrees are equal
        if np.all(degrees == degrees[0]):
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    best_n = 5
    best_edges = []
    
    # Try to find valid graphs for n from 22 down to 6
    for n in range(22, 5, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        max_attempts = 500
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate a random total order (permutation)
            order = list(range(n))
            random.shuffle(order)
            
            # Build comparability graph: edge (i,j) if |pos(i) - pos(j)| > threshold
            # This creates a graph where vertices have very different degrees
            # based on their position in the order
            threshold = n // 3  # tune this
            
            adj = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i + 1, n):
                    pos_i = order.index(i)
                    pos_j = order.index(j)
                    if abs(pos_i - pos_j) > threshold:
                        adj[i, j] = True
                        adj[j, i] = True
            
            # Check validity
            if _is_valid_graph(adj, n):
                # Convert to edge list
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i, j]:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            # Also try random graphs as fallback
            if attempt % 10 == 0:
                adj = np.zeros((n, n), dtype=bool)
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            adj[i, j] = True
                            adj[j, i] = True
                
                if _is_valid_graph(adj, n):
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i, j]:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
    
    # Fallback: return empty graph on 5 vertices
    return {'n': 5, 'edges': []}
