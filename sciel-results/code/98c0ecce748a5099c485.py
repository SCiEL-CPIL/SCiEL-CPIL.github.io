import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    def check_valid(n, adj):
        """Check if no subset of size >= 6 induces a regular subgraph.
        adj is an n x n boolean numpy array (adjacency matrix, no self-loops).
        Returns True if valid (no regular subset of size >= 6)."""
        # Only need to check size-6 subsets: if a larger subset S is regular,
        # then any 6-subset of S is NOT necessarily regular, so we must check all sizes.
        # However, for efficiency, we check size 6 first (most numerous), then larger.
        
        # Check size 6 subsets using vectorized approach
        # For n=20, C(20,6) = 38760 subsets
        # For n=21, C(21,6) = 54264 subsets
        
        for k in range(6, n + 1):
            for subset in combinations(range(n), k):
                # Compute degrees within subset
                sub_adj = adj[np.ix_(subset, subset)]
                degrees = sub_adj.sum(axis=1)
                if len(set(degrees)) == 1:
                    return False  # Found a regular subset
        return True
    
    def check_valid_fast(n, adj):
        """Faster check: only check size-6 subsets first, then larger if needed.
        Uses early termination."""
        # Size 6: most common bad case
        # For efficiency, check all size-6 subsets
        for subset in combinations(range(n), 6):
            sub_adj = adj[np.ix_(subset, subset)]
            degrees = sub_adj.sum(axis=1)
            if np.all(degrees == degrees[0]):
                return False
        
        # If no size-6 regular subset, check size 7+
        for k in range(7, n + 1):
            for subset in combinations(range(n), k):
                sub_adj = adj[np.ix_(subset, subset)]
                degrees = sub_adj.sum(axis=1)
                if np.all(degrees == degrees[0]):
                    return False
        return True
    
    # Try n=21 first (would be a breakthrough)
    for n_target in [21, 20, 19, 18]:
        if time.time() > deadline - 10:
            break
        
        max_attempts = 500 if n_target <= 20 else 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate random graph with p=0.5
            np.random.seed(attempt * 1000 + n_target)
            # Create upper triangle random matrix
            upper = np.random.random((n_target, n_target)) < 0.5
            # Make symmetric, no self-loops
            adj = np.triu(upper, 1) | np.triu(upper, 1).T
            np.fill_diagonal(adj, False)
            
            # Check validity
            if check_valid_fast(n_target, adj):
                # Convert to edge list
                edges = []
                for i in range(n_target):
                    for j in range(i + 1, n_target):
                        if adj[i, j]:
                            edges.append([i, j])
                return {'n': n_target, 'edges': edges}
    
    # Fallback: return a known small valid graph
    return {'n': 5, 'edges': []}
