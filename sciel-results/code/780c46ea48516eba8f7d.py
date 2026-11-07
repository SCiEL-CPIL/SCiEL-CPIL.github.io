"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from collections import deque
import itertools

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    
    # Build base distance matrix
    def get_base_dist():
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        dist = [[n]*n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[i][v] == n:
                        dist[i][v] = dist[i][u] + 1
                        q.append(v)
        return dist

    base_dist = get_base_dist()
    
    def get_bad_pairs_from_dist(dist):
        bad = set()
        for i in range(n):
            for j in range(i+1, n):
                if dist[i][j] > 4:
                    bad.add(frozenset((i, j)))
        return bad

    bad_pairs_base = get_bad_pairs_from_dist(base_dist)
    
    # Candidates
    candidates = []
    for u in range(n):
        for v in range(u + 3, n):
            candidates.append((u, v))

    def update_dist(dist, u, v):
        # Add edge (u, v) to graph represented by dist
        # Returns new dist matrix
        new_dist = [row[:] for row in dist]
        # The new edge allows paths u-v and v-u
        # dist'(i, j) = min(dist(i, j), dist(i, u) + 1 + dist(v, j), dist(i, v) + 1 + dist(u, j))
        # We can do this in O(N^2)
        for i in range(n):
            for j in range(n):
                # Path via u -> v
                d1 = dist[i][u] + 1 + dist[v][j]
                # Path via v -> u
                d2 = dist[i][v] + 1 + dist[u][j]
                if d1 < new_dist[i][j]:
                    new_dist[i][j] = d1
                if d2 < new_dist[i][j]:
                    new_dist[i][j] = d2
        return new_dist

    def dfs(current_edges, current_dist, current_bad, remaining, candidates_list):
        if not current_bad:
            return current_edges
        if remaining == 0:
            return None
            
        potentials = []
        union_fixes = set()
        
        for e in candidates_list:
            u, v = e
            # Check triangle: if dist[u][v] == 2, adding edge creates triangle
            if current_dist[u][v] == 2:
                continue
                
            new_dist = update_dist(current_dist, u, v)
            
            # Check if new_dist is valid (no negative, etc, not needed for simple update)
            # Compute new bad pairs
            new_bad = get_bad_pairs_from_dist(new_dist)
            fixes = current_bad - new_bad
            
            if not fixes:
                continue
            
            potentials.append((len(fixes), e, new_dist, new_bad))
            union_fixes.update(fixes)
            
        if not potentials:
            return None
            
        if union_fixes != current_bad:
            return None
            
        potentials.sort(key=lambda x: x[0], reverse=True)
        
        for _, e, new_dist, new_bad in potentials:
            new_candidates = [c for c in candidates_list if c != e]
            res = dfs(current_edges + [e], new_dist, new_bad, remaining - 1, new_candidates)
            if res is not None:
                return res
                
        return None

    for k in range(1, n):
        res = dfs([], base_dist, bad_pairs_base, k, candidates)
        if res is not None:
            return [[u, v] for u, v in res]
            
    return []
# EVOLVE-BLOCK-END
