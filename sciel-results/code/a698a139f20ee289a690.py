import time
import itertools

def construct():
    n = 17
    sets = [
        {1, 2, 4, 8},
        {9, 13, 15, 16},
        {3, 5, 6, 7},
        {10, 11, 12, 14}
    ]
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            for c in range(4):
                if d in sets[c]:
                    colors.append(c)
                    break
    
    # Verify
    if not is_valid(colors, n):
        # Fallback to seed
        n = 9
        colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        
    return {'n': n, 'colors': colors}

def is_valid(colors, n):
    # Build adjacency lists for each color
    adj = [[] for _ in range(4)]
    for c in range(4):
        adj[c] = [[] for _ in range(n)]
        
    for k, c in enumerate(colors):
        # Decode k to (i, j)
        # This is slow. Let's just rebuild from pairs.
        pass
        
    # Better verification:
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for c in range(4):
        adj_c = [[] for _ in range(n)]
        for k, (i, j) in enumerate(pairs):
            if colors[k] == c:
                adj_c[i].append(j)
                adj_c[j].append(i)
                
        if has_c6(adj_c, n):
            return False
            
    return True

def has_c6(adj, n):
    # Check for C6 in graph adj
    # Use the 6-subset method
    for subset in itertools.combinations(range(n), 6):
        # Check if induced subgraph has C6
        # Build local adjacency
        # Or just check if there is a cycle of length 6
        # For 6 vertices, we can check if there is a Hamiltonian cycle
        # in the induced subgraph.
        
        # Get the 6 vertices
        v = subset
        # Check all permutations? No, just check if the subgraph has a cycle of length 6.
        # A subgraph on 6 vertices has a C6 iff it has a Hamiltonian cycle.
        
        # Simple check:
        # Count edges in induced subgraph
        # If < 6, no C6.
        # If >= 6, check for HC.
        
        # To speed up, we can just check if the subgraph is 2-regular? No.
        
        # Let's just do a DFS for a cycle of length 6 starting from v[0].
        # This is fast enough for 12376 subsets.
        
        def dfs(start, current, visited, path):
            if len(path) == 6:
                if start in adj[current]:
                    return True
                return False
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    if dfs(start, neighbor, visited, path):
                        return True
                    path.pop()
                    visited.remove(neighbor)
            return False
            
        # We only care about cycles within the subset
        # So we should restrict neighbors to those in the subset
        subset_set = set(subset)
        
        def dfs_restricted(start, current, visited, path):
            if len(path) == 6:
                if start in adj[current] and start in subset_set:
                    return True
                return False
            for neighbor in adj[current]:
                if neighbor in subset_set and neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    if dfs_restricted(start, neighbor, visited, path):
                        return True
                    path.pop()
                    visited.remove(neighbor)
            return False
            
        # Try starting from each vertex in subset?
        # No, a cycle is a cycle. We can just fix the start vertex.
        # But the cycle might not include the start vertex?
        # No, we are checking the induced subgraph on the subset.
        # If there is a C6, it uses all 6 vertices.
        # So we can fix the start vertex to be subset[0].
        
        if dfs_restricted(subset[0], subset[0], {subset[0]}, [subset[0]]):
            return True
            
    return False
