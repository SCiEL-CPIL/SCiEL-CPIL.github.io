"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
    
    adj_list = [sorted(adj[i]) for i in range(n)]
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    def check_time():
        return time.time() < deadline
    
    def bfs_path(src, dst, forbidden):
        """Find shortest path from src to dst avoiding forbidden vertices (except src/dst)."""
        if src == dst:
            return [src]
        visited = {src}
        queue = deque([(src, [src])])
        while queue:
            node, path = queue.popleft()
            for nb in adj_list[node]:
                if nb == dst:
                    return path + [dst]
                if nb not in forbidden and nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        return None
    
    def try_build_subdivision(branches, deadline, path_order=None):
        """Try to find internally disjoint paths for all pairs in branches.
        Returns list of paths or None if impossible."""
        branches = list(branches)
        branch_set = set(branches)
        pairs = list(itertools.combinations(branches, 2))
        
        if path_order is None:
            # Order pairs: direct edges first, then by BFS distance
            pair_info = []
            for (u, v) in pairs:
                if v in adj[u]:
                    pair_info.append((0, u, v))
                else:
                    pair_info.append((1, u, v))
            pair_info.sort()
            ordered_pairs = [(u, v) for _, u, v in pair_info]
        else:
            ordered_pairs = path_order
        
        paths = {}
        used_internal = set()
        
        for u, v in ordered_pairs:
            if time.time() > deadline:
                return None
            forbidden = branch_set - {u, v} | used_internal
            path = bfs_path(u, v, forbidden)
            if path is None:
                return None
            paths[(u, v)] = path
            for x in path[1:-1]:
                used_internal.add(x)
        
        return [paths[(u, v)] for (u, v) in pairs]
    
    def find_cliques(deadline):
        """Find maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        
        def bronk(R, P, X):
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(list(R))
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
        
        bronk(set(), set(range(n)), set())
        cliques.sort(key=len, reverse=True)
        return cliques
    
    def expand_branches(base_branches, deadline):
        """Try to expand a set of branches by adding vertices one at a time."""
        current = list(base_branches)
        current_set = set(current)
        best = list(current)
        
        for _ in range(100):
            if time.time() > deadline:
                break
            
            candidates = [v for v in range(n) if v not in current_set]
            random.shuffle(candidates)
            
            added = False
            for v in candidates:
                if time.time() > deadline:
                    break
                new_branches = current + [v]
                result = try_build_subdivision(new_branches, deadline)
                if result is not None:
                    current = new_branches
                    current_set.add(v)
                    if len(current) > len(best):
                        best = list(current)
                    added = True
                    break
            
            if not added:
                break
        
        return best
    
    def expand_with_backtracking(base_branches, deadline):
        """More thorough expansion trying different orders."""
        current = list(base_branches)
        current_set = set(current)
        best = list(current)
        
        for _ in range(200):
            if time.time() > deadline:
                break
            
            candidates = [v for v in range(n) if v not in current_set]
            
            # Score candidates by how many direct edges they have to current branches
            scored = []
            for v in candidates:
                direct_edges = sum(1 for b in current if b in adj[v])
                scored.append((direct_edges, v))
            scored.sort(key=lambda x: (-x[0], x[1]))
            
            added = False
            for _, v in scored:
                if time.time() > deadline:
                    break
                new_branches = current + [v]
                result = try_build_subdivision(new_branches, deadline)
                if result is not None:
                    current = new_branches
                    current_set.add(v)
                    if len(current) > len(best):
                        best = list(current)
                    added = True
                    break
            
            if not added:
                # Try removing a vertex and adding a different one
                if len(current) >= 3:
                    for i in range(len(current)):
                        if time.time() > deadline:
                            break
                        temp = current[:i] + current[i+1:]
                        temp_set = set(temp)
                        candidates2 = [v for v in range(n) if v not in temp_set]
                        random.shuffle(candidates2)
                        for v in candidates2[:15]:
                            new_branches = temp + [v]
                            result = try_build_subdivision(new_branches, deadline)
                            if result is not None:
                                current = new_branches
                                current_set = set(current)
                                if len(current) > len(best):
                                    best = list(current)
                                break
        
        return best
    
    # Phase 1: Find cliques
    cliques = find_cliques(deadline)
    
    best_branches = None
    best_size = 0
    
    # Try each clique as a starting point
    for clique in cliques:
        if time.time() > deadline - 5:
            break
        if len(clique) < 2:
            break
        if len(clique) <= best_size:
            continue
        
        # Expand from this clique
        expanded = expand_with_backtracking(clique, deadline - 5)
        if len(expanded) > best_size:
            best_size = len(expanded)
            best_branches = expanded
    
    # Phase 2: Random restarts with random initial pairs
    if time.time() < deadline - 10:
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            # Pick a random edge as starting pair
            u, v = random.choice(P['edges'])
            initial = [u, v]
            expanded = expand_with_backtracking(initial, deadline - 10)
            if len(expanded) > best_size:
                best_size = len(expanded)
                best_branches = expanded
    
    # Phase 3: More aggressive search - try all pairs of high-degree vertices
    if time.time() < deadline - 15:
        # Sort vertices by degree
        deg = [len(adj[i]) for i in range(n)]
        high_deg = sorted(range(n), key=lambda x: -deg[x])[:12]
        
        for i in range(len(high_deg)):
            if time.time() > deadline - 15:
                break
            for j in range(i+1, len(high_deg)):
                if time.time() > deadline - 15:
                    break
                u, v = high_deg[i], high_deg[j]
                if v not in adj[u]:
                    continue
                initial = [u, v]
                expanded = expand_with_backtracking(initial, deadline - 15)
                if len(expanded) > best_size:
                    best_size = len(expanded)
                    best_branches = expanded
    
    # Final: build the result
    if best_branches is not None and len(best_branches) >= 2:
        result = try_build_subdivision(best_branches, deadline)
        if result is not None:
            branches = [int(b) for b in best_branches]
            paths = [[int(x) for x in p] for p in result]
            return {'branches': branches, 'paths': paths}
    
    # Fallback: find any valid pair
    for u, v in P['edges']:
        if v in adj[u]:
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    u, v = P['edges'][0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
# EVOLVE-BLOCK-END
