"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    adj_list = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    for i in range(n):
        adj_list[i].sort(key=lambda x: -degrees[x])
    
    # Precompute common neighbors
    common_nb = [[set() for _ in range(n)] for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            c = adj[u] & adj[v]
            common_nb[u][v] = c
            common_nb[v][u] = c
    
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    sorted_vertices = sorted(range(n), key=lambda x: -degrees[x])
    
    def find_shortest_paths(src, dst, forbidden, max_paths=6):
        """BFS to find up to max_paths shortest simple paths from src to dst."""
        if src == dst:
            return []
        q = deque()
        q.append((src, [src]))
        visited = {src}
        results = []
        shortest_dist = None
        while q:
            if len(results) >= max_paths:
                break
            curr, path = q.popleft()
            if curr == dst:
                results.append(path)
                if shortest_dist is None:
                    shortest_dist = len(path) - 1
                if len(results) >= max_paths:
                    break
                continue
            if shortest_dist is not None and len(path) - 1 >= shortest_dist:
                continue
            for nb in adj_list[curr]:
                if nb not in visited and (nb not in forbidden or nb == dst):
                    visited.add(nb)
                    q.append((nb, path + [nb]))
        return results
    
    def check_subdivision(branches):
        """Check if branches form a valid clique subdivision. Returns paths or None."""
        k = len(branches)
        branch_set = set(branches)
        
        # Identify pairs and non-adjacent pairs
        all_pairs = []
        non_adj_pairs = []
        for i in range(k):
            for j in range(i + 1, k):
                u, v = branches[i], branches[j]
                all_pairs.append((i, j, u, v))
                if v not in adj[u]:
                    non_adj_pairs.append((i, j, u, v))
        
        # Pigeonhole: need at least len(non_adj_pairs) available vertices for length-3 paths
        available = n - k
        if len(non_adj_pairs) > available:
            return None
        
        # Phase 1: Try bipartite matching with length-3 paths
        pair_to_verts = {}
        for idx, (i, j, u, v) in enumerate(non_adj_pairs):
            valid_ws = [w for w in common_nb[u][v] if w not in branch_set]
            pair_to_verts[idx] = valid_ws
        
        possible_matching = True
        for idx in range(len(non_adj_pairs)):
            if not pair_to_verts[idx]:
                possible_matching = False
                break
        
        if possible_matching and len(non_adj_pairs) > 0:
            # Kuhn's algorithm for bipartite matching
            match_vertex = {v: -1 for v in range(n)}
            
            def try_kuhn(v_idx, seen):
                for u_vert in pair_to_verts[v_idx]:
                    if seen[u_vert]:
                        continue
                    seen[u_vert] = True
                    if match_vertex[u_vert] == -1 or try_kuhn(match_vertex[u_vert], seen):
                        match_vertex[u_vert] = v_idx
                        return True
                return False
            
            found = True
            for i_idx in range(len(non_adj_pairs)):
                seen = {v: False for v in range(n)}
                if not try_kuhn(i_idx, seen):
                    found = False
                    break
            
            if found:
                # Build result
                pair_to_match_vert = {}
                for v, p_idx in match_vertex.items():
                    if p_idx != -1:
                        pair_to_match_vert[p_idx] = v
                
                paths = []
                for idx, (i, j, u, v) in enumerate(all_pairs):
                    if v in adj[u]:
                        paths.append([u, v])
                    else:
                        na_idx = None
                        for na_i, (ni, nj, nu, nv) in enumerate(non_adj_pairs):
                            if (nu, nv) == (u, v) or (nv, nu) == (u, v):
                                na_idx = na_i
                                break
                        w = pair_to_match_vert[na_idx]
                        paths.append([u, w, v])
                
                return paths
        
        # Phase 2: Backtracking with BFS path options
        def pair_difficulty(p):
            i, j, u, v = p
            has_direct = 1 if v in adj[u] else 0
            common = len((adj[u] & adj[v]) - branch_set)
            return (has_direct, -common)
        
        ordered_pairs = sorted(all_pairs, key=pair_difficulty)
        
        used_interiors = set()
        path_map = {}
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == len(ordered_pairs):
                return True
            
            i, j, u, v = ordered_pairs[idx]
            
            # Try direct edge first
            if v in adj[u]:
                path_map[(i, j)] = [u, v]
                if backtrack(idx + 1):
                    return True
                del path_map[(i, j)]
            
            # Try longer paths via BFS
            forbidden = used_interiors | (branch_set - {u, v})
            path_options = find_shortest_paths(u, v, forbidden, max_paths=4)
            
            for path in path_options:
                interior = set(path[1:-1])
                if interior.isdisjoint(branch_set) and interior.isdisjoint(used_interiors):
                    path_map[(i, j)] = path
                    used_interiors.update(interior)
                    if backtrack(idx + 1):
                        return True
                    used_interiors.difference_update(interior)
                    del path_map[(i, j)]
            
            return False
        
        if backtrack(0):
            paths = []
            for i in range(k):
                for j in range(i + 1, k):
                    paths.append(path_map[(i, j)])
            return paths
        
        return None
    
    # Try k from 12 down to 2
    for k in range(12, 1, -1):
        if time.time() > deadline:
            break
        
        # Adjust pool size based on k
        if k >= 10:
            pool_size = n
        elif k >= 8:
            pool_size = max(16, n - 2)
        else:
            pool_size = max(14, n - 4)
        
        candidates = sorted_vertices[:pool_size]
        
        for combo in combinations(candidates, k):
            if time.time() > deadline:
                break
            
            branches = list(combo)
            paths = check_subdivision(branches)
            if paths is not None:
                result_branches = [int(b) for b in branches]
                result_paths = [list(p) for p in paths]
                return {'branches': result_branches, 'paths': result_paths}
    
    # Fallback: max clique
    best_clique = []
    def bron_kerbosch(R, P_set, X):
        nonlocal best_clique
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        if P_set:
            pivot = max(P_set, key=lambda v: len(adj[v] & P_set))
            candidates = P_set - adj[pivot]
        else:
            candidates = P_set
        for v in list(candidates):
            bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v])
            P_set.remove(v)
            X.add(v)
    
    bron_kerbosch(set(), set(range(n)), set())
    
    if len(best_clique) >= 2:
        branches = sorted(best_clique)
        paths = []
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                paths.append([int(branches[i]), int(branches[j])])
        return {'branches': list(branches), 'paths': paths}
    
    u, v = P['edges'][0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

# EVOLVE-BLOCK-END
