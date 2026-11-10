from itertools import combinations

def construct():
    n = 12
    
    # First, try to find a graph with only one distinct cycle length
    # We'll do a targeted search
    
    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append([i, j])
        return edges
    
    def is_triangle_free(adj):
        for i in range(n):
            for j in adj[i]:
                if i >= j:
                    continue
                for k in adj[i]:
                    if k > j and k in adj[j]:
                        return False
        return True
    
    def min_degree(adj):
        return min(len(adj[i]) for i in range(n))
    
    def has_cycle(adj):
        # BFS/DFS to check for cycle
        visited = [False] * n
        for start in range(n):
            if not visited[start]:
                stack = [start]
                visited[start] = True
                while stack:
                    node = stack.pop()
                    for nei in adj[node]:
                        if not visited[nei]:
                            visited[nei] = True
                            stack.append(nei)
                        elif nei != start:
                            # This simple check isn't perfect, but for our purposes
                            # if we can reach a visited node that's not the parent, there's a cycle
                            pass
        # Better: just check if edges >= vertices for any component
        # Actually, let's just enumerate cycles
        return True  # We'll verify with cycle enumeration
    
    def enumerate_cycle_lengths(adj):
        """Find all simple cycle lengths in the graph."""
        cycle_lengths = set()
        
        def dfs(start, current, path, visited):
            for nei in adj[current]:
                if nei == start and len(path) >= 3:
                    cycle_lengths.add(len(path))
                elif not visited[nei] and nei > start:
                    visited[nei] = True
                    path.append(nei)
                    dfs(start, nei, path, visited)
                    path.pop()
                    visited[nei] = False
        
        for start in range(n):
            visited = [False] * n
            visited[start] = True
            dfs(start, start, [start], visited)
        
        return cycle_lengths
    
    # Try a specific construction: three 4-cycles connected in a way that only creates 4-cycles
    # 4-cycle 1: 0-1-2-3-0
    # 4-cycle 2: 4-5-6-7-4  
    # 4-cycle 3: 8-9-10-11-8
    
    # Connect them with cross edges
    # Try: 0-4, 1-5, 2-8, 3-9, 6-10, 7-11
    # Let me check degrees:
    # 0: 1,3,4 -> deg 3
    # 1: 0,2,5 -> deg 3
    # 2: 1,3,8 -> deg 3
    # 3: 2,0,9 -> deg 3
    # 4: 5,7,0 -> deg 3
    # 5: 4,6,1 -> deg 3
    # 6: 5,7,10 -> deg 3
    # 7: 6,4,11 -> deg 3
    # 8: 9,11,2 -> deg 3
    # 9: 8,10,3 -> deg 3
    # 10: 9,11,6 -> deg 3
    # 11: 10,8,7 -> deg 3
    # All degree 3!
    
    edges_candidate = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 4-cycle 1
        [4, 5], [5, 6], [6, 7], [7, 4],  # 4-cycle 2
        [8, 9], [9, 10], [10, 11], [11, 8],  # 4-cycle 3
        [0, 4], [1, 5], [2, 8], [3, 9], [6, 10], [7, 11]  # cross edges
    ]
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges_candidate:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check triangle-free
    triangle_free = True
    for i in range(n):
        for j in adj[i]:
            if i >= j:
                continue
            for k in adj[i]:
                if k > j and k in adj[j]:
                    triangle_free = False
                    break
            if not triangle_free:
                break
        if not triangle_free:
            break
    
    if triangle_free and min(len(adj[i]) for i in range(n)) >= 3:
        cycle_lens = enumerate_cycle_lengths(adj)
        if len(cycle_lens) == 1:
            return [[u, v] for u, v in edges_candidate]
    
    # Fallback: two K3,3 components
    edges = []
    for u in range(3):
        for v in range(3, 6):
            edges.append([u, v])
    for u in range(6, 9):
        for v in range(9, 12):
            edges.append([u, v])
    
    return edges
