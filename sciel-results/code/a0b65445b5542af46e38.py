import time
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((min(u, v), max(u, v)))
    
    def is_valid(added_edges):
        if not added_edges:
            return False # Path diameter is 23
        
        for u, v in added_edges:
            if u == v:
                return False
            if (min(u, v), max(u, v)) in base_edges:
                return False
        
        n = P['n']
        adj = [[] for _ in range(n)]
        for u, v in base_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        added_set = set()
        for u, v in added_edges:
            if (min(u, v), max(u, v)) in added_set:
                return False
            added_set.add((min(u, v), max(u, v)))
            adj[u].append(v)
            adj[v].append(u)
            
        # Triangle check
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    if set(adj[u]) & set(adj[v]):
                        return False
        
        # Connected
        visited = [False] * n
        stack = [0]
        visited[0] = True
        while stack:
            node = stack.pop()
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        if not all(visited):
            return False
            
        # Diameter
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                node = queue[head]
                head += 1
                for neighbor in adj[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        queue.append(neighbor)
            if max(dist) > 4:
                return False
        return True

    # Initial solution
    initial = [
        [11, 0], [11, 2], [11, 4], [11, 6], [11, 8],
        [11, 14], [11, 16], [11, 18], [11, 20], [11, 22]
    ]
    
    best_edges = initial[:]
    
    # Local Search: Try to remove edges
    improved = True
    edges = best_edges[:]
    while improved:
        improved = False
        for i in range(len(edges)):
            test = edges[:i] + edges[i+1:]
            if is_valid(test):
                edges = test
                improved = True
                break
                
    if len(edges) < len(best_edges):
        best_edges = edges[:]
        
    # Try random swaps/additions if we have time?
    # If we are at 10, try to find 9.
    # Generate random candidates with 9 edges?
    # Too many.
    
    # Try 2-hub specific construction
    # We know 1 hub gives 10.
    # Let's try to manually construct a 9-edge 2-hub solution.
    # Hub 5, Hub 18.
    # L=5, R=18.
    # S_L: 0, 2, 4? (5-4 is path? No. 5-4 is path. So 4 is neighbor. Can't add 5-4? No, 4 is neighbor of 5. So 5-4 is base edge. Can't add.
    # Wait, 5 is connected to 4 and 6 in path.
    # So S_L cannot contain 4, 6.
    # S_L can contain 0, 2, 8, 10?
    # 5-2 (dist 3 in path). Add 5-2.
    # 5-0 (dist 5). Add 5-0.
    # 5-8 (dist 3). Add 5-8.
    # 5-10 (dist 5). Add 5-10.
    # Check triangles:
    # 5-2-1-0-5? No.
    # 2 and 0 are not adjacent.
    # 8 and 10 are not adjacent.
    # 2 and 8 not adjacent.
    # So S_L = {0, 2, 8, 10} is independent?
    # 0-1-2. 0 and 2 not adjacent.
    # 8-9-10. 8 and 10 not adjacent.
    # Yes.
    # S_R for 18.
    # Neighbors 17, 19.
    # S_R can contain 20, 22, 23?
    # 18-20 (dist 2). Add.
    # 18-22 (dist 4). Add.
    # 18-23 (dist 5). Add.
    # 20, 22, 23.
    # 20-21-22. 20, 22 not adjacent.
    # 22-23 adjacent. So can't have both 22 and 23.
    # Pick 20, 22.
    # S_R = {20, 22}.
    # Total edges: 1 (5-18) + 4 (S_L) + 2 (S_R) = 7?
    # Let's check diameter.
    # Dist(5, 18) = 1.
    # Dist(0, 23) = dist(0, 5) + dist(5, 18) + dist(18, 23)
    # dist(0, 5) = 1 (edge 0-5).
    # dist(18, 23) = dist(18, 22) + 1 = 1 + 1 = 2.
    # Total 1 + 1 + 2 = 4.
    # Dist(10, 20) = dist(10, 5) + dist(5, 18) + dist(18, 20)
    # dist(10, 5) = 1 (edge 10-5).
    # dist(18, 20) = 1 (edge 18-20).
    # Total 1 + 1 + 1 = 3.
    # Dist(1, 21) = dist(1, 0) + dist(0, 5) + dist(5, 18) + dist(18, 20) + dist(20, 21)
    # 1 + 1 + 1 + 1 + 1 = 5. Fail.
    # We need to cover 1 and 21 better.
    # Add 1 to S_L?
    # 5-1 (dist 4).
    # 1 and 0 adjacent. Can't have both.
    # If we swap 0 for 1?
    # S_L = {1, 2, 8, 10}.
    # 1 and 2 adjacent. Fail.
    # S_L = {0, 2, 8, 10} seems good for left.
    # Maybe add 17 to S_R?
    # 18-17 is base edge. Can't add.
    # Add 16 to S_R?
    # 18-16 (dist 2).
    # S_R = {16, 20, 22}.
    # 16, 20, 22 independent?
    # 16-17-18-19-20. 16, 20 not adjacent.
    # 20, 22 not adjacent.
    # Yes.
    # Edges: 1 + 4 + 3 = 8.
    # Check diameter.
    # Dist(1, 21) = dist(1, 0) + dist(0, 5) + dist(5, 18) + dist(18, 20) + dist(20, 21)
    # 1 + 1 + 1 + 1 + 1 = 5. Still 5.
    # We need to shorten the path between 1 and 21.
    # Maybe connect 5 to 1?
    # If S_L = {1, 2, 8, 10}, 1 and 2 adjacent.
    # If S_L = {0, 2, 8, 10}, 1 is dist 2 from 5 (1-0-5).
    # 21 is dist 2 from 18 (21-20-18).
    # So dist(1, 21) = 2 + 1 + 2 = 5.
    # We need one of these to be 1.
    # Add 1 to S_L? Can't (adjacent to 2).
    # Add 21 to S_R? Can't (adjacent to 20).
    # What if we change hubs?
    
    # This manual search is slow.
    # I will return the best found by greedy removal.
    
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    return result
