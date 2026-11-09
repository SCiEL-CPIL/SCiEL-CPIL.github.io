def construct():
    from itertools import combinations, permutations
    from collections import deque
    
    n = 12
    # We want a cubic graph (degree 3) on 12 vertices, triangle-free
    # Search for graph with minimum distinct cycle lengths
    
    def get_cycles(graph, n):
        """Find all simple cycles in an undirected graph."""
        cycles = set()
        adj = {i: set() for i in range(n)}
        for u, v in graph:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find all simple cycles
        def find_cycles(start, path, visited):
            if len(path) > 1 and path[-1] == start:
                cycle_len = len(path) - 1
                if cycle_len >= 3:
                    # Normalize the cycle
                    cycle = tuple(sorted(path[:-1]))
                    cycles.add(cycle_len)
                return
            
            for neighbor in adj[path[-1]]:
                if neighbor == start and len(path) > 3:
                    find_cycles(start, path + [start], visited)
                elif neighbor not in visited:
                    find_cycles(start, path + [neighbor], visited | {neighbor})
        
        for i in range(n):
            find_cycles(i, [i], {i})
        
        return cycles
    
    def is_triangle_free(graph, n):
        adj = {i: set() for i in range(n)}
        for u, v in graph:
            adj[u].add(v)
            adj[v].add(u)
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    for k in adj[j]:
                        if k > j and k in adj[i]:
                            return False
        return True
    
    def min_degree(graph, n):
        deg = [0] * n
        for u, v in graph:
            deg[u] += 1
            deg[v] += 1
        return min(deg)
    
    # Try all possible cubic graphs on 12 vertices (this is too many, so use heuristic)
    # Instead, try known good structures
    
    # Try: two K_{3,3} components (current best: 2 distinct lengths)
    edges1 = []
    for a in (0,1,2):
        for b in (3,4,5):
            edges1.append(tuple(sorted((a,b))))
    for a in (6,7,8):
        for b in (9,10,11):
            edges1.append(tuple(sorted((a,b))))
    
    cycles1 = get_cycles(edges1, n)
    len1 = len(cycles1)
    
    # Try: a single connected cubic graph
    # Use the Petersen graph + 2 vertices? Petersen is 10 vertices.
    # Let's try the dodecahedron? No, that's 20 vertices.
    
    # Try: cube graph Q_3 (8 vertices) + 4 vertices
    # Q_3 edges:
    q3_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    # Add 4 vertices (8,9,10,11) and connect to make cubic
    # This is complex, let's try a different approach
    
    # Try: a bipartite graph with parts {0,1,2,3,4,5} and {6,7,8,9,10,11}
    # Each vertex degree 3, so 18 edges total
    # Try to make it so all cycles are length 4
    
    # One approach: use a "grid-like" structure
    # Actually, let's try the complete bipartite K_{3,3} + K_{3,3} but check if we can do better
    
    # For now, return the known best
    return [list(e) for e in sorted(set(edges1))]
