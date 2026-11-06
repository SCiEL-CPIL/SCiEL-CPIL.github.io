import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin

    def is_regular_subgraph(vertices, adj_np):
        """Check if the induced subgraph on 'vertices' is regular using NumPy."""
        if len(vertices) < 6:
            return False
        # Extract submatrix
        sub_adj = adj_np[np.ix_(vertices, vertices)]
        # Compute degrees
        degrees = np.sum(sub_adj, axis=1)
        # Check if all degrees are equal
        return np.all(degrees == degrees[0])

    def has_regular_induced_subgraph(adj_np, n):
        """Check if any subset of 6+ vertices induces a regular subgraph."""
        # We must check all sizes from 6 to n
        for size in range(6, n + 1):
            for subset in combinations(range(n), size):
                if is_regular_subgraph(subset, adj_np):
                    return True
        return False

    def check_graph(adj_np, n):
        """Check if graph has no regular induced subgraph on 6+ vertices."""
        return not has_regular_induced_subgraph(adj_np, n)

    def try_construct(n, max_attempts=500):
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None

            # Build random graph
            # Probability 0.5 is standard, but maybe we can tune?
            # Let's stick to 0.5 for now but use vectorized generation
            mask = np.random.rand(n, n) < 0.5
            # Ensure symmetry
            adj = (mask | mask.T).astype(np.int8)
            # Remove self-loops
            np.fill_diagonal(adj, 0)
            
            # Check validity
            if check_graph(adj, n):
                # Convert to edge list
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i, j]:
                            edges.append([i, j])
                return edges
        
        return None

    # Try larger graphs first
    # The current code tries 20 down to 6.
    # With faster checking, we might be able to afford more attempts for n=20.
    for n in [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() > deadline:
            break
        
        # Use more attempts for smaller n since they're easier, 
        # but with NumPy, even n=20 might be feasible for a few attempts.
        # Let's allocate time budget per n.
        # If n=20 takes ~1s per check (optimistic), 500 attempts = 500s. Too long.
        # Let's estimate check time.
        # 1 million subsets * (extract + sum + all)
        # Extract is O(k^2). Sum is O(k^2).
        # For k=6, it's small.
        # Python loop overhead is the killer. 1 million iterations in Python is ~1s-10s?
        # Let's try 50 attempts for n=20, 100 for n=19, etc.
        
        max_attempts = 50 if n >= 18 else (100 if n >= 15 else 500)

        edges = try_construct(n, max_attempts)
        if edges is not None:
            return {'n': n, 'edges': edges}
    
    # Fallback: return a valid small graph
    # Empty graph on 5 vertices
    return {'n': 5, 'edges': []}
# EVOLVE-BLOCK-END
