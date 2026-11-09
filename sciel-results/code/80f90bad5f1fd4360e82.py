"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def construct():
    """
    Find a graph with no regular induced subgraph on 6+ vertices.
    Uses local search: start random, fix violations by flipping edges.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Helper: check if induced subgraph on subset is regular
    def is_regular_induced(subset, adj):
        """Check if all vertices in subset have same degree within subset."""
        if len(subset) < 2:
            return True
        degrees = []
        for v in subset:
            deg = sum(1 for u in subset if u != v and adj[v][u])
            degrees.append(deg)
        return all(d == degrees[0] for d in degrees)
    
    # Helper: find all regular induced subgraphs of size >= 6
    def find_violations(n, adj):
        violations = []
        for size in range(6, n + 1):
            for subset in combinations(range(n), size):
                if is_regular_induced(subset, adj):
                    violations.append(subset)
        return violations
    
    # Helper: flip an edge in adjacency matrix
    def flip_edge(adj, u, v):
        adj[u][v] = 1 - adj[u][v]
        adj[v][u] = adj[u][v]
    
    # Try to find a valid graph for a given n
    def try_find_graph(n, max_iters=5000):
        # Start with random graph
        adj = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        # Local search
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            violations = find_violations(n, adj)
            if not violations:
                # Build edge list
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i][j]:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            # Pick a random violation and try to fix it
            subset = random.choice(violations)
            subset_set = set(subset)
            
            # Try flipping an edge within the subset or connecting subset to outside
            candidates = []
            # Edges within subset
            for i, u in enumerate(subset):
                for v in subset[i+1:]:
                    candidates.append((u, v))
            # Edges from subset to outside
            outside = [v for v in range(n) if v not in subset_set]
            for u in subset:
                for v in outside:
                    candidates.append((u, v))
            
            if candidates:
                u, v = random.choice(candidates)
                flip_edge(adj, u, v)
        
        return None
    
    # Try increasing n from high to low
    best_result = {'n': 1, 'edges': []}
    
    for n in range(22, 5, -1):
        if time.time() > deadline - 10:
            break
        
        # Multiple restarts for each n
        max_restarts = 3 if n <= 15 else 2
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            result = try_find_graph(n, max_iters=2000 if n > 12 else 5000)
            if result is not None:
                best_result = result
                break  # Found valid graph for this n, try smaller n? No, we want max n
                # Actually we want the largest n, so if we found it, we're done
        
        if best_result['n'] == n:
            break  # Found valid graph for this n
    
    return best_result
# EVOLVE-BLOCK-END
