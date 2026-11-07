"""Rewritten program with stronger clique subdivision search."""
import time
import itertools
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    # Precompute sorted adjacency lists for deterministic traversal
    adj_sorted = [sorted(adj[i]) for i in range(n)]
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u,v), max(u,v)))
    
    def is_edge(u, v):
        return (min(u,v), max(u,v)) in edge_set
    
    def dfs_path(src, dst, forbidden, deadline):
        """DFS to find a path from src to dst avoiding forbidden (except endpoints).
        Tries direct edge first, then short paths via DFS with backtracking."""
        if src == dst:
            return [src]
        if is_edge(src, dst):
            return [src, dst]
        
        # DFS with backtracking
        path = [src]
        visited = {src}
        
        def dfs(node):
            if time.time() > deadline:
                return False
            for nb in adj_sorted[node]:
                if nb == dst:
                    path.append(dst)
                    return True
                if nb in forbidden or nb in visited:
                    continue
                visited.add(nb)
                path.append(nb)
                if dfs(nb):
                    return True
                path.pop()
                visited.discard(nb)
            return False
        
        if dfs(src):
            return path
        return None
    
    def try_build_subdivision(branches, deadline):
        """Find internally disjoint paths for all pairs in branches.
        Returns dict of (u,v) -> path, or None if impossible."""
        branches = list(branches)
        branch_set = set(branches)
        pairs = list(itertools.combinations(branches, 2))
        
        # Order pairs: direct edges first, then by degree sum (high degree = more options)
        pair_order = []
        for (u, v) in pairs:
            if is_edge(u, v):
                pair_order.append((0, len(adj[u]) + len(adj[v]), u, v))
            else:
                pair_order.append((1, len(adj[u]) + len(adj[v]), u, v))
        pair_order.sort(key=lambda x: (x[0], -x[1]))
        
        paths = {}
        used_internal = set()
        
        for _, _, u, v in pair_order:
            if time.time() > deadline:
                return None
            forbidden = branch_set - {u, v} | used_internal
            path = dfs_path(u, v, forbidden, deadline)
            if path is None:
                return None
            paths[(u, v)] = path
            for x in path[1:-1]:
                used_internal.add(x)
        
        return paths
    
    def find_max_cliques(deadline):
        """Bron-Kerbosch to find maximal cliques."""
        cliques = []
        
        def bronk(R, P, X):
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            if P or X:
                pivot = max(P | X, key=lambda v: len(adj[v] & P))
            else:
                return
            candidates = [v for v in P if v not in adj[pivot]]
            for v in candidates:
                bronk(R | {v}, P & adj[v], X & adj[v])
                P = P - {v}
                X = X | {v}
        
        bronk(frozenset(), set(range(n)), set())
        cliques.sort(key=len, reverse=True)
        return cliques
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    best_result = None
    best_size = 0
    
    # Phase 1: Find largest cliques and try to build subdivisions
    cliques = find_max_cliques(deadline)
    
    for clique in cliques:
        if time.time() > deadline - 5:
            break
        if len(clique) < 2:
            break
        if len(clique) <= best_size:
            continue
        result = try_build_subdivision(clique, deadline)
        if result is not None:
            branches = sorted(clique)
            paths_list = [result[(u, v)] for u, v in itertools.combinations(branches, 2)]
            best_result = {'branches': branches, 'paths': paths_list}
            best_size = len(clique)
    
    # Phase 2: Greedy expansion from best clique
    if best_size >= 2 and time.time() < deadline - 10:
        current_branches = list(best_result['branches'])
        
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            candidates = [v for v in range(n) if v not in set(current_branches)]
            # Sort by degree (higher degree = more likely to have paths)
            candidates.sort(key=lambda v: -len(adj[v]))
            
            added = False
            for v in candidates:
                if time.time() > deadline - 10:
                    break
                new_branches = current_branches + [v]
                result = try_build_subdivision(new_branches, deadline)
                if result is not None:
                    branches = sorted(new_branches)
                    paths_list = [result[(u, w)] for u, w in itertools.combinations(branches, 2)]
                    current_branches = new_branches
                    best_result = {'branches': branches, 'paths': paths_list}
                    best_size = len(branches)
                    added = True
                    break
            
            if not added:
                # Try swap: remove one, add another
                if len(current_branches) >= 3:
                    swapped = False
                    for i in range(len(current_branches)):
                        if time.time() > deadline - 10:
                            break
                        temp = current_branches[:i] + current_branches[i+1:]
                        for v in candidates:
                            if v in set(temp):
                                continue
                            new_branches = temp + [v]
                            result = try_build_subdivision(new_branches, deadline)
                            if result is not None:
                                branches = sorted(new_branches)
                                paths_list = [result[(u, w)] for u, w in itertools.combinations(branches, 2)]
                                current_branches = new_branches
                                best_result = {'branches': branches, 'paths': paths_list}
                                best_size = len(branches)
                                swapped = True
                                break
                        if swapped:
                            break
                    if not swapped:
                        break
    
    # Phase 3: Randomized restarts from different cliques
    if time.time() < deadline - 5:
        for clique in cliques[:30]:
            if time.time() > deadline - 5:
                break
            if len(clique) < 2:
                continue
            rb = list(clique)
            random.shuffle(rb)
            
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                candidates = [v for v in range(n) if v not in set(rb)]
                random.shuffle(candidates)
                added = False
                for v in candidates:
                    new_branches = rb + [v]
                    result = try_build_subdivision(new_branches, deadline)
                    if result is not None:
                        branches = sorted(new_branches)
                        paths_list = [result[(u, w)] for u, w in itertools.combinations(branches, 2)]
                        rb = new_branches
                        if len(rb) > best_size:
                            best_size = len(rb)
                            best_result = {'branches': branches, 'paths': paths_list}
                        added = True
                        break
                if not added:
                    break
    
    if best_result is not None:
        branches = [int(b) for b in best_result['branches']]
        paths = [[int(x) for x in p] for p in best_result['paths']]
        return {'branches': branches, 'paths': paths}
    
    # Final fallback: just an edge
    u, v = P['edges'][0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
