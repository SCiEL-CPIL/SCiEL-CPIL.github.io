"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import heapq
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    # Build adjacency list
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((min(u,v), max(u,v)))
    
    def bfs_distances(start):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)
        return dist
    
    def get_diameter():
        max_d = 0
        for i in range(n):
            dists = bfs_distances(i)
            for d in dists:
                if d > max_d:
                    max_d = d
        return max_d
    
    def would_create_triangle(u, v):
        # Check if adding edge (u,v) creates a triangle
        # A triangle exists if there's a common neighbor w of u and v
        neighbors_u = set(adj[u])
        for w in adj[v]:
            if w in neighbors_u:
                return True
        return False
    
    # Greedy: add edges until diameter <= 4
    added_edges = []
    
    while True:
        # Compute current diameter and find worst pair
        dist_matrix = []
        for i in range(n):
            dist_matrix.append(bfs_distances(i))
        
        max_dist = 0
        worst_pair = None
        for i in range(n):
            for j in range(i+1, n):
                if dist_matrix[i][j] > max_dist:
                    max_dist = dist_matrix[i][j]
                    worst_pair = (i, j)
        
        if max_dist <= 4:
            break
        
        # Find best edge to add
        # Try all valid (even-odd) edges not in edge_set
        best_edge = None
        best_score = -1
        
        for u in range(n):
            for v in range(u+1, n):
                if (u, v) in edge_set:
                    continue
                if (u - v) % 2 == 0:  # same parity, skip (must be even-odd for bipartite)
                    continue
                if would_create_triangle(u, v):
                    continue
                
                # Score: how much does this edge reduce the max distance?
                # Approximate: count pairs where this edge helps
                # Better: simulate adding edge, compute new max dist
                # For efficiency, just use a heuristic: reduce distance between worst_pair
                new_dist_worst = min(dist_matrix[worst_pair[0]][worst_pair[1]], 
                                     dist_matrix[worst_pair[0]][u] + 1 + dist_matrix[v][worst_pair[1]],
                                     dist_matrix[worst_pair[0]][v] + 1 + dist_matrix[u][worst_pair[1]])
                
                # Also check global improvement: count how many pairs improve
                improvements = 0
                for i in range(n):
                    for j in range(i+1, n):
                        old_d = dist_matrix[i][j]
                        new_d = min(old_d,
                                     dist_matrix[i][u] + 1 + dist_matrix[v][j],
                                     dist_matrix[i][v] + 1 + dist_matrix[u][j])
                        if new_d < old_d:
                            improvements += 1
                
                score = improvements * 1000 - max_dist  # prioritize improvements
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)
        
        if best_edge is None:
            # Should not happen, but break to avoid infinite loop
            break
        
        u, v = best_edge
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((u, v))
        added_edges.append([u, v])
    
    return added_edges
# EVOLVE-BLOCK-END
