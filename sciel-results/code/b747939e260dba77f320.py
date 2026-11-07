import random
import itertools

def construct():
    N = 15
    
    # Start from complete bipartite K_{7,8} with 56 edges
    # Vertices 0..6 in A, 7..14 in B
    edges = set()
    for i in range(7):
        for j in range(7, 15):
            edges.add((i, j))
    
    # Try to add chords (edges within A or within B) if they don't violate the constraint
    # For each potential chord, check if adding it violates the constraint
    
    def is_valid(edge_set):
        """Check if the graph satisfies the constraint."""
        # Build adjacency and degree
        adj = {i: set() for i in range(N)}
        for (u, v) in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        deg = {i: len(adj[i]) for i in range(N)}
        
        # Check every path s-u-v-t where s != t
        # For each edge (u, v), for each s in adj[u] \ {v}, for each t in adj[v] \ {u}:
        #   if s != t and deg[s] == deg[t], invalid
        for (u, v) in edge_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u:
                        continue
                    if s != t and deg[s] == deg[t]:
                        return False
        return True
    
    # Try adding all possible chords within A and within B
    candidate_chords = []
    # Within A: vertices 0..6
    for i in range(7):
        for j in range(i + 1, 7):
            candidate_chords.append((i, j))
    # Within B: vertices 7..14
    for i in range(7, 15):
        for j in range(i + 1, 15):
            candidate_chords.append((i, j))
    
    # Try adding chords one by one, keeping only those that maintain validity
    # Use a greedy approach: try all chords, keep the set that maximizes edge count
    best_edges = set(edges)
    best_count = len(edges)
    
    # Try each individual chord
    for chord in candidate_chords:
        test_edges = best_edges | {chord}
        if is_valid(test_edges):
            best_edges = test_edges
            best_count += 1
    
    # Try to add more chords greedily
    changed = True
    while changed:
        changed = False
        for chord in candidate_chords:
            if chord not in best_edges:
                test_edges = best_edges | {chord}
                if is_valid(test_edges):
                    best_edges = test_edges
                    best_count += 1
                    changed = True
    
    # Convert to list of lists with proper types
    result = []
    for (u, v) in best_edges:
        result.append([int(u), int(v)])
    
    return result
