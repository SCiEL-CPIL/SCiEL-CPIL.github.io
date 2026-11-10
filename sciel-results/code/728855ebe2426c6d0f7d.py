"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    n = P['n']
    edges = P['edges']
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # 1. Find Maximum Clique
    max_clique = []
    
    def expand_clique(r, p, x):
        nonlocal max_clique
        if time.time() > deadline:
            return
        if not p and not x:
            if len(r) > len(max_clique):
                max_clique = list(r)
            return
        if p:
            u = next(iter(p))
            candidates = list(p - adj[u])
        else:
            candidates = list(p)
        
        for v in candidates:
            if time.time() > deadline:
                return
            expand_clique(r | {v}, p & adj[v], x & adj[v])
            p.remove(v)
            x.add(v)
            
    expand_clique(set(), set(range(n)), set())
    
    if len(max_clique) > 3:
        branches = sorted(max_clique)
        paths = []
        for u, v in itertools.combinations(branches, 2):
            paths.append([u, v])
        return {'branches': branches, 'paths': paths}

    # 2. Helper to find clique subdivision for a given set of branches
    def find_clique_subdivision(branches):
        pairs = list(itertools.combinations(branches, 2))
        
        def get_shortest_path(u, v, blocked):
            visited = {u}
            parent = {u: None}
            queue = deque([u])
            
            while queue:
                curr = queue.popleft()
                if curr == v:
                    path = []
                    node = v
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        if neighbor == v:
                            visited.add(neighbor)
                            parent[neighbor] = curr
                            queue.append(neighbor)
                        elif neighbor not in blocked:
                            visited.add(neighbor)
                            parent[neighbor] = curr
                            queue.append(neighbor)
            return None

        def backtrack(idx, used_interiors, paths_found):
            if time.time() > deadline:
                return None
            if idx == len(pairs):
                return paths_found
            
            u, v = pairs[idx]
            blocked = set(branches) | used_interiors
            
            path = get_shortest_path(u, v, blocked)
            
            if path:
                interior = set(path[1:-1])
                new_used = used_interiors | interior
                new_paths = paths_found.copy()
                new_paths[(u,v)] = path
                
                result = backtrack(idx + 1, new_used, new_paths)
                if result:
                    return result
            
            return None

        return backtrack(0, set(), {})

    # 3. Search for K_4 Subdivision
    found = None
    for combo in itertools.combinations(range(n), 4):
        if time.time() > deadline:
            break
        branches = list(combo)
        res = find_clique_subdivision(branches)
        if res:
            found = {
                'branches': branches,
                'paths': [res[(u,v)] for u,v in itertools.combinations(branches, 2)]
            }
            break
            
    if found:
        return found

    # 4. Search for K_3 Subdivision
    if time.time() < deadline - 5:
        for combo in itertools.combinations(range(n), 3):
            if time.time() > deadline:
                break
            branches = list(combo)
            res = find_clique_subdivision(branches)
            if res:
                return {
                    'branches': branches,
                    'paths': [res[(u,v)] for u,v in itertools.combinations(branches, 2)]
                }

    # 5. Fallback to Max Clique
    if len(max_clique) >= 2:
        branches = sorted(max_clique)
        paths = []
        for u, v in itertools.combinations(branches, 2):
            paths.append([u, v])
        return {'branches': branches, 'paths': paths}
    else:
        u, v = edges[0]
        return {'branches': [u, v], 'paths': [[u, v]]}

# EVOLVE-BLOCK-END
