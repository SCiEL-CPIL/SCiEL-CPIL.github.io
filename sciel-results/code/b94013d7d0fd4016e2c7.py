"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = 85.0
    
    def time_up():
        return time.time() - start_time > deadline
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    def is_clique(vertices):
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if (vertices[i], vertices[j]) not in edge_set:
                    return False
        return True
    
    def find_max_clique():
        """Find maximum clique using Bron-Kerbosch with pivot."""
        best = []
        
        def bronk(R, P_set, X):
            nonlocal best
            if time_up():
                return
            if not P_set and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            
            # Choose pivot
            pivot_candidates = P_set | X
            if pivot_candidates:
                # Pick vertex with max degree in P_set
                pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P_set))
                candidates = P_set - adj[pivot]
            else:
                candidates = P_set
            
            for v in list(candidates):
                bronk(R | {v}, P_set & adj[v], X & adj[v])
                P_set.discard(v)
                X.add(v)
        
        bronk(set(), set(range(n)), set())
        return best
    
    # Find maximum clique
    max_clique = find_max_clique()
    
    if time_up() or not max_clique:
        # Fallback: find any triangle
        for combo in combinations(range(n), 3):
            if is_clique(list(combo)):
                max_clique = list(combo)
                break
        if not max_clique:
            u, v = edges[0]
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    # Try to extend the clique by adding vertices that can connect via paths
    # For a clique subdivision, we need internally disjoint paths between all pairs
    # If we use direct edges, the clique itself works. Let's verify and try to extend.
    
    def verify_clique_subdivision(branches):
        """Verify that branches form a valid clique subdivision using direct edges."""
        branch_set = set(branches)
        paths = {}
        for u, v in combinations(branches, 2):
            if (u, v) in edge_set:
                paths[(u, v)] = [u, v]
            else:
                return None
        return paths
    
    # Start with the max clique and try to add more vertices
    current_branches = list(max_clique)
    
    # Try to add vertices one by one
    remaining = set(range(n)) - set(current_branches)
    
    # Sort remaining by degree (higher degree first, more likely to connect)
    remaining_sorted = sorted(remaining, key=lambda v: -len(adj[v]))
    
    for v in remaining_sorted:
        if time_up():
            break
        
        new_branches = current_branches + [v]
        paths = verify_clique_subdivision(new_branches)
        if paths is not None:
            current_branches = new_branches
    
    # If the direct-edge clique doesn't work for all pairs, try path-based approach
    paths = verify_clique_subdivision(current_branches)
    
    if paths is None:
        # Fall back to just the clique with direct edges where possible
        # Find the largest subset that forms a clique with direct edges
        # This shouldn't happen since max_clique is a clique
        current_branches = list(max_clique)
        paths = verify_clique_subdivision(current_branches)
        if paths is None:
            # Ultimate fallback
            u, v = edges[0]
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    # Build the result
    result_paths = []
    for u, v in combinations(current_branches, 2):
        result_paths.append(paths[(u, v)])
    
    # Convert to proper types
    branches_out = [int(b) for b in current_branches]
    paths_out = [[int(x) for x in p] for p in result_paths]
    
    return {'branches': branches_out, 'paths': paths_out}

# EVOLVE-BLOCK-END
