"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit
    
    def is_regular(subset, adj):
        """Check if induced subgraph on subset is regular."""
        k = len(subset)
        if k < 6:
            return False
        # Compute degrees within the induced subgraph
        degs = [0] * k
        idx_map = {v: i for i, v in enumerate(subset)}
        for i, u in enumerate(subset):
            for v in subset[i+1:]:
                if adj[u][v]:
                    degs[i] += 1
                    degs[idx_map[v]] += 1
        # Check if all degrees are equal
        first = degs[0]
        for d in degs[1:]:
            if d != first:
                return False
        return True
    
    def check_graph(n, adj):
        """Check if graph has any regular induced subgraph on 6+ vertices."""
        for k in range(6, n+1):
            for subset in combinations(range(n), k):
                if is_regular(subset, adj):
                    return False
        return True
    
    def try_random_graph(n, trials=50):
        """Try to find a valid random graph on n vertices."""
        for t in range(trials):
            if time.time() > deadline:
                return None
            adj = [[False]*n for _ in range(n)]
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i][j] = adj[j][i] = True
                        edges.append([i, j])
            if check_graph(n, adj):
                return {'n': n, 'edges': edges}
        return None
    
    # Try from large n down to small
    # Known: 20-vertex graphs exist. Try 21 first, then 20, etc.
    for n in [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() > deadline:
            break
        result = try_random_graph(n, trials=20)
        if result is not None:
            return result
    
    # Fallback: empty graph on 5 vertices
    return {'n': 5, 'edges': []}
# EVOLVE-BLOCK-END
