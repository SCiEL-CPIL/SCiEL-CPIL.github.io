import time
import itertools
import sys
from collections import deque

# Global deadline
START_TIME = time.time()
TIME_LIMIT = 85.0 # Safety margin for 90s limit

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

# Build adjacency structure
n = P['n']
edges = P['edges']
adj = [[] for _ in range(n)]
adj_set = [set() for _ in range(n)]
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)
    adj_set[u].add(v)
    adj_set[v].add(u)

# Sort neighbors for deterministic behavior
for i in range(n):
    adj[i].sort()

# Sort vertices by degree (descending) for heuristic ordering
sorted_vertices = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)

def find_max_clique_size():
    # Simple backtracking
    max_clique = []
    
    def bronk(R, P, X):
        nonlocal max_clique
        if not P and not X:
            if len(R) > len(max_clique):
                max_clique = R[:]
            return
        
        if P or X:
            u = max(P | X, key=lambda v: len(adj_set[v] & P))
        else:
            return
            
        P_minus_u = P - adj_set[u]
        
        for v in P_minus_u:
            bronk(R + [v], P & adj_set[v], X & adj_set[v])
            P.remove(v)
            X.add(v)

    V = set(range(n))
    bronk([], V, set())
    return len(max_clique)

def check_subdivision(B):
    """
    Returns (True, paths_dict) if a K_|B| subdivision exists with branch set B.
    """
    if len(B) < 2:
        return False, None
    
    B_set = set(B)
    available = [v for v in range(n) if v not in B_set]
    pairs = list(itertools.combinations(B, 2))
    
    paths = {}
    
    def dfs(idx, blocked_set):
        if idx == len(pairs):
            return True
        
        u, v = pairs[idx]
        
        # Check reachability first
        # BFS from u to v in G - (blocked_set ∪ (B \ {u,v}))
        # Nodes allowed: u, v, and (available \ blocked_set)
        
        allowed_nodes = set(available) - blocked_set
        allowed_nodes.add(u)
        allowed_nodes.add(v)
        
        # Quick BFS
        visited = {u}
        q = deque([u])
        found = False
        while q:
            node = q.popleft()
            if node == v:
                found = True
                break
            for nb in adj[node]:
                if nb in visited:
                    continue
                if nb not in allowed_nodes:
                    continue
                visited.add(nb)
                q.append(nb)
                
        if not found:
            return False
            
        # Find simple paths
        def get_paths():
            res = []
            if time.time() - START_TIME > TIME_LIMIT:
                raise TimeoutError
                
            def dfs_p(curr, path):
                if time.time() - START_TIME > TIME_LIMIT:
                    raise TimeoutError
                if len(res) > 20: # Limit candidates
                    return
                if curr == v:
                    res.append(path[:])
                    return
                
                for nb in adj[curr]:
                    if nb in path:
                        continue
                    if nb not in allowed_nodes:
                        continue
                    path.append(nb)
                    dfs_p(nb, path)
                    path.pop()
            
            dfs_p(u, [u])
            res.sort(key=lambda x: len(x))
            return res
        
        try:
            candidate_paths = get_paths()
        except TimeoutError:
            return False
        
        if not candidate_paths:
            return False
        
        for path in candidate_paths:
            interiors = set(path[1:-1])
            # Interiors should be disjoint from blocked_set (ensured by allowed_nodes)
            # and disjoint from previous interiors (ensured by blocked_set update)
            
            paths[(u, v)] = path
            new_blocked = blocked_set | interiors
            
            if dfs(idx + 1, new_blocked):
                return True
            
            del paths[(u, v)]
            
        return False

    try:
        if dfs(0, set()):
            return True, paths
    except TimeoutError:
        return False, None
        
    return False, None

def construct():
    max_clique_size = find_max_clique_size()
    
    # Try k from max_clique_size down to 2
    # If max_clique_size is small, we might miss non-clique branches?
    # But usually max subdivision size <= max clique size + something?
    # Actually, in a subdivision, the branch graph is a clique.
    # So the max number of branches is bounded by the size of the largest clique in the "branch graph", which is K_k.
    # This doesn't bound k by the clique number of G directly, but it's a good heuristic.
    
    # To be safe, if max_clique_size is 3, we should still check k=4 non-cliques?
    # C(24,4) is 10k. It's fast enough.
    
    max_k_to_try = max(max_clique_size, 4) # At least try 4
    
    for k in range(max_k_to_try, 1, -1):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # Use sorted_vertices for heuristic ordering
        # itertools.combinations(sorted_vertices, k)
        
        # To avoid re-checking, we could store failed sets? Not needed for small N.
        
        for B in itertools.combinations(sorted_vertices, k):
            if time.time() - START_TIME > TIME_LIMIT:
                break
                
            # Optional: Quick check if G-B is connected?
            # If G-B is disconnected, and some branches are in different components, no path.
            # But branches are in B, so they are removed.
            # Paths must be in G-B.
            # So if G-B is disconnected, we need all pairs to be in the same component?
            # No, u and v are in B. The path is in G-B.
            # So u and v must be connected in G-B?
            # No, u and v are NOT in G-B.
            # The path is from u to v.
            # u is connected to neighbors in G-B. v is connected to neighbors in G-B.
            # So u must be able to reach v via G-B.
            # This implies u and v must be in the same "super-component" if we consider G-B + u + v.
            
            # The BFS in check_subdivision handles this.
            
            success, paths_dict = check_subdivision(list(B))
            if success:
                branches = list(B)
                paths = []
                # Ensure consistent ordering in paths list?
                # Task says "one simple path for each unordered pair".
                # Order in list doesn't matter, but let's be consistent.
                for u, v in itertools.combinations(branches, 2):
                    if (u, v) in paths_dict:
                        paths.append(list(paths_dict[(u, v)]))
                    else:
                        # Should not happen
                        paths.append([u, v])
                
                return {
                    'branches': branches,
                    'paths': paths
                }
    
    # Fallback: Find a triangle
    for i in range(n):
        for j in range(i+1, n):
            if j not in adj_set[i]:
                continue
            for l in range(j+1, n):
                if l in adj_set[i] and l in adj_set[j]:
                    return {
                        'branches': [i, j, l],
                        'paths': [[i, j], [i, l], [j, l]]
                    }
    
    # Fallback: Edge
    if edges:
        u, v = edges[0]
        return {
            'branches': [u, v],
            'paths': [[u, v]]
        }
    
    return {
        'branches': [0, 1],
        'paths': [[0, 1]]
    }
