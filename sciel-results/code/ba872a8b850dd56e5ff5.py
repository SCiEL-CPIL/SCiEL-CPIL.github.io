"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        u, v = e
        base_edges.add((min(u,v), max(u,v)))
    
    # Build adjacency list for base graph
    adj = [[] for _ in range(n)]
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Candidate edges: only between even and odd vertices (to maintain triangle-freeness)
    # and not already in base
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u - v) % 2 == 1:  # different parity
                if (min(u,v), max(u,v)) not in base_edges:
                    candidates.append((u, v))
    
    # Greedy approach: repeatedly add the edge that most reduces the maximum distance
    added_edges = []
    added_set = set()
    current_adj = [list(a) for a in adj]
    
    def compute_diameter(adj_list):
        """Compute diameter using BFS from each vertex."""
        max_dist = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for neighbor in adj_list[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        queue.append(neighbor)
                        max_dist = max(max_dist, dist[neighbor])
        return max_dist
    
    def bfs_distances(adj_list, start):
        """Compute distances from start using BFS."""
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adj_list[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)
        return dist
    
    # Check initial diameter
    initial_diam = compute_diameter(current_adj)
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # Greedy: keep adding edges until diameter <= 4
    while compute_diameter(current_adj) > 4:
        if time.time() > deadline:
            break
        
        best_edge = None
        best_new_diam = float('inf')
        
        # Get current distances from each vertex to identify far pairs
        all_dists = [bfs_distances(current_adj, s) for s in range(n)]
        
        # Find the pair with maximum distance
        max_dist_pair = (0, 0)
        max_dist_val = 0
        for i in range(n):
            for j in range(i+1, n):
                d = max(all_dists[i][j], all_dists[j][i])
                if d > max_dist_val:
                    max_dist_val = d
                    max_dist_pair = (i, j)
        
        # Prioritize candidates that are "useful" - connect vertices that can help reduce max distance
        # Sort candidates by how central they are (closer to endpoints of longest paths)
        sorted_candidates = []
        for u, v in candidates:
            if (min(u,v), max(u,v)) in added_set:
                continue
            # Score: prefer edges that are far from each other and reduce eccentricity
            # Simple heuristic: prefer edges where both endpoints have high eccentricity
            ecc_u = max(all_dists[u]) if all_dists[u][0] != -1 else 0
            ecc_v = max(all_dists[v]) if all_dists[v][0] != -1 else 0
            sorted_candidates.append((-(ecc_u + ecc_v), u, v))
        
        sorted_candidates.sort()
        
        # Try candidates in priority order, but limit number of trials per iteration
        max_trials = min(len(sorted_candidates), 200)
        for _, u, v in sorted_candidates[:max_trials]:
            if (min(u,v), max(u,v)) in added_set:
                continue
            
            # Try adding this edge
            temp_adj = [list(a) for a in current_adj]
            temp_adj[u].append(v)
            temp_adj[v].append(u)
            
            new_diam = compute_diameter(temp_adj)
            if new_diam < best_new_diam:
                best_new_diam = new_diam
                best_edge = (u, v)
            
            # If we already achieved diameter <= 4, stop early
            if best_new_diam <= 4:
                break
        
        if best_edge is None:
            break
        
        u, v = best_edge
        current_adj[u].append(v)
        current_adj[v].append(u)
        added_set.add((min(u,v), max(u,v)))
        added_edges.append([u, v])
    
    # Verify triangle-freeness
    def is_triangle_free(adj_list):
        adj_set = [set(a) for a in adj_list]
        for i in range(n):
            for j in adj_set[i]:
                if j > i:
                    for k in adj_set[j]:
                        if k > j and k in adj_set[i]:
                            return False
        return True
    
    if not is_triangle_free(current_adj):
        # Fallback: return empty if something went wrong
        return []
    
    # Convert to proper Python int lists
    result = []
    for e in added_edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END
