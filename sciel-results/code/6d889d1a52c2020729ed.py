from itertools import combinations
from collections import deque

def construct():
    n = 16
    
    # Construction: K_{3,13} plus a triangle on vertices 0,1,2
    edges = set()
    # Triangle on 0,1,2
    edges.add((0, 1))
    edges.add((0, 2))
    edges.add((1, 2))
    # All edges between {0,1,2} and {3,...,15}
    for a in range(3):
        for b in range(3, 16):
            edges.add((a, b))
    
    # Verify no Q3
    all_8subsets = list(combinations(range(n), 8))
    
    def is_q3(vertices, edge_set):
        vset = set(vertices)
        induced_edges = []
        for i, u in enumerate(vertices):
            for v in vertices[i+1:]:
                if (u, v) in edge_set or (v, u) in edge_set:
                    induced_edges.append((u, v))
        if len(induced_edges) != 12:
            return False
        
        deg = {v: 0 for v in vertices}
        for (u, v) in induced_edges:
            deg[u] += 1
            deg[v] += 1
        for v in vertices:
            if deg[v] != 3:
                return False
        
        # Check bipartite
        adj = {v: set() for v in vertices}
        for (u, v) in induced_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        color = {}
        for start in vertices:
            if start not in color:
                color[start] = 0
                queue = deque([start])
                while queue:
                    node = queue.popleft()
                    for neighbor in adj[node]:
                        if neighbor not in color:
                            color[neighbor] = color[node] ^ 1
                            queue.append(neighbor)
                        elif color[neighbor] == color[node]:
                            return False
        
        # Check no triangles
        for u in vertices:
            neighbors = list(adj[u])
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    if neighbors[j] in adj[neighbors[i]]:
                        return False
        
        return True
    
    for vertices in all_8subsets:
        if is_q3(vertices, edges):
            # Found Q3, this construction is invalid
            # Fall back to greedy removal from K16
            break
    else:
        # No Q3 found, return this construction
        return [list(e) for e in edges]
    
    # Fallback: greedy removal from K16
    edges = set()
    for u in range(n):
        for v in range(u+1, n):
            edges.add((u, v))
    
    for _ in range(10000):
        found = False
        for vertices in all_8subsets:
            if is_q3(vertices, edges):
                vset = set(vertices)
                q3_edges = []
                for i, u in enumerate(vertices):
                    for v in vertices[i+1:]:
                        if (u, v) in edges:
                            q3_edges.append((u, v))
                if q3_edges:
                    edges.remove(q3_edges[0])
                    found = True
                    break
        if not found:
            break
    
    return [list(e) for e in edges]
