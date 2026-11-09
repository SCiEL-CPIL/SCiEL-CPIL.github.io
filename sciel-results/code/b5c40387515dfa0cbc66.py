import random
import time
from collections import deque

def construct():
    n_vertices = 40
    best_edges_count = 0
    best_result = []
    
    def creates_c6(u, v, adj):
        # BFS from u to depth 3
        dist_u = [-1] * n_vertices
        dist_u[u] = 0
        q = deque([u])
        while q:
            node = q.popleft()
            if dist_u[node] >= 3:
                continue
            for nei in adj[node]:
                if dist_u[nei] == -1:
                    dist_u[nei] = dist_u[node] + 1
                    q.append(nei)
                    
        # BFS from v to depth 3
        dist_v = [-1] * n_vertices
        dist_v[v] = 0
        q = deque([v])
        while q:
            node = q.popleft()
            if dist_v[node] >= 3:
                continue
            for nei in adj[node]:
                if dist_v[nei] == -1:
                    dist_v[nei] = dist_v[node] + 1
                    q.append(nei)
        
        # Check for path of length 5
        # We look for edges (x, y) that bridge the distance gap
        
        # Helper to check if path u->x and v->y are disjoint
        # This is hard without storing the actual paths.
        # Given the rarity, let's just do a DFS if a candidate is found.
        
        found_candidate = False
        
        # Case 1: dist_u[x] == 2, dist_v[y] == 2
        for x in range(n_vertices):
            if dist_u[x] == 2:
                for y in range(n_vertices):
                    if dist_v[y] == 2:
                        if (x, y) in adj[x]: # Check if edge exists
                            found_candidate = True
                            break
                if found_candidate: break
        if found_candidate:
            # Verify with DFS
            return dfs_c6(u, v, adj)
            
        # Case 2: dist_u[x] == 1, dist_v[y] == 3
        for x in range(n_vertices):
            if dist_u[x] == 1:
                for y in range(n_vertices):
                    if dist_v[y] == 3:
                        if (x, y) in adj[x]:
                            found_candidate = True
                            break
                if found_candidate: break
        if found_candidate:
            return dfs_c6(u, v, adj)
            
        # Case 3: dist_u[x] == 3, dist_v[y] == 1
        for x in range(n_vertices):
            if dist_u[x] == 3:
                for y in range(n_vertices):
                    if dist_v[y] == 1:
                        if (x, y) in adj[x]:
                            found_candidate = True
                            break
                if found_candidate: break
        if found_candidate:
            return dfs_c6(u, v, adj)
            
        return False

    def dfs_c6(u, v, adj):
        # Check if simple path of length 5 exists
        def dfs(node, depth, visited):
            if depth == 5:
                return node == v
            for nei in adj[node]:
                if nei not in visited and nei != u:
                    visited.add(nei)
                    if dfs(nei, depth + 1, visited):
                        return True
                    visited.remove(nei)
            return False
        return dfs(u, 1, {u})

    # ... rest of the code using this check
