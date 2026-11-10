"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

P = {'n': 15}

def construct():
    n = P['n']
    # Start with no edges
    edges = set()
    
    # Try to add all possible edges in a greedy manner
    # We'll check constraint after adding each edge
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def get_degrees(edge_set):
        deg = [0] * n
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def is_valid(edge_set):
        if not edge_set:
            return True
        deg = get_degrees(edge_set)
        # Build adjacency
        adj = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check every 3-edge path s-u-v-t
        for (u, v) in edge_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True
    
    # Greedy: try to add edges, prefer those that don't break constraint
    # Sort edges to try - maybe by some heuristic
    for u, v in all_possible:
        new_edges = edge_set | {(u, v)}
        if is_valid(new_edges):
            edge_set = new_edges
    
    # Convert to list of lists with proper types
    result = []
    for u, v in sorted(edge_set):
        result.append([int(u), int(v)])
    return result
# EVOLVE-BLOCK-END
