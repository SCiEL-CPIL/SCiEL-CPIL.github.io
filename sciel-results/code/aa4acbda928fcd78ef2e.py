import time
import sys
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # ---- Max-flow based feasibility check ----
    def check_and_find_paths(branches):
        """
        Check if branches form a valid clique subdivision using max-flow.
        Returns dict of paths or None if infeasible.
        
        Model: For each pair (u,v) of branches, we need a simple path.
        All interior vertices across all paths must be distinct and not in branches.
        
        We use a flow network:
        - Split each non-branch vertex w into w_in -> w_out with capacity 1
        - For each edge (a,b) in original graph:
          - if both a,b are non-branch: a_out -> b_in (cap 1) and b_out -> a_in (cap 1)
          - if a is branch, b non-branch: a -> b_in (cap 1) and b_out -> a (cap 1)  
          - if both branches: direct edge (cap 1)
        - Source connects to each branch, each branch connects to sink
        - But we need paths for ALL pairs simultaneously...
        
        Actually, the correct model: we need vertex-disjoint paths for ALL pairs.
        This is a multi-commodity flow problem, which is NP-hard in general.
        
        Alternative: use the fact that for a clique subdivision, we can think of it as
        finding |B| choose 2 vertex-disjoint paths. 
        
        Simpler approach: Use a single max-flow where we send |B|(|B|-1)/2 units of flow,
        with each pair having its own "commodity". But multi-commodity is hard.
        
        Practical approach: Greedy assignment with backtracking, but use BFS to find
        shortest available paths, processing pairs in order of fewest options first.
        """
        b_set = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        num_pairs = len(pairs)
        
        if num_pairs == 0:
            return {}
        
        # For each pair, find all shortest paths (length 1, 2, 3 interior vertices max for efficiency)
        # Precompute path options
        path_options = {}
        for u, v in pairs:
            options = []
            # Direct edge
            if v in adj[u]:
                options.append([u, v])
            # Length 2 (one interior vertex)
            for mid in range(n):
                if mid in b_set:
                    continue
                if u in adj[mid] and mid in adj[v]:
                    options.append([u, mid, v])
            # Length 3 (two interior vertices) - only if needed
            if not options:
                # BFS for shortest path
                forbidden = b_set - {u, v}
                visited = {u}
                parent = {u: None}
                queue = deque([u])
                found = False
                while queue:
                    if time.time() > deadline:
                        return None
                    cur = queue.popleft()
                    if cur == v:
                        path = []
                        node = v
                        while node is not None:
                            path.append(node)
                            node = parent[node]
                        path.reverse()
                        if len(path) <= 5:  # limit path length
                            options.append(path)
                        found = True
                        break
                    for nb in sorted(adj[cur]):
                        if nb not in visited and nb not in forbidden:
                            visited.add(nb)
                            parent[nb] = cur
                            queue.append(nb)
                if not found:
                    return None
            # Also find a few more alternatives via BFS with different orderings
            if len(options) < 5:
                # Try to find a few more short paths
                forbidden = b_set - {u, v}
                visited = {u}
                parent = {u: None}
                queue = deque([u])
                found_paths = []
                while queue and len(found_paths) < 5:
                    if time.time() > deadline:
                        return None
                    cur = queue.popleft()
                    if cur == v:
                        path = []
                        node = v
                        while node is not None:
                            path.append(node)
                            node = parent[node]
                        path.reverse()
                        if path not in found_paths and len(path) <= 5:
                            found_paths.append(path)
                        continue
                    for nb in sorted(adj[cur], key=lambda x: -len(adj[x])):
                        if nb not in visited and nb not in forbidden:
                            visited.add(nb)
                            parent[nb] = cur
                            queue.append(nb)
                for p in found_paths:
                    if p not in options:
                        options.append(p)
            
            if not options:
                return None
            path_options[(u, v)] = options
        
        # Backtracking: assign paths to pairs, ensuring interior vertices are disjoint
        best_result = [None]
        
        # Order pairs by number of options (most constrained first)
        pairs_sorted = sorted(pairs, key=lambda p: len(path_options[p]))
        
        def bt(idx, used_interior, assigned):
            if time.time() > deadline:
                return
            if idx == num_pairs:
                best_result[0] = dict(assigned)
                return
            u, v = pairs_sorted[idx]
            for path in path_options[(u, v)]:
                mids = set(path[1:-1])
                if mids & used_interior:
                    continue
                if mids & b_set:
                    continue
                assigned[(u, v)] = path
                new_used = used_interior | mids
                bt(idx + 1, new_used, assigned)
                if best_result[0] is not None:
                    return
                del assigned[(u, v)]
        
        bt(0, set(), {})
        return best_result[0]
    
    # ---- Bron-Kerbosch for maximum clique ----
    max_cliques = []
    
    def bronk(R, P_set, X):
        if time.time() > deadline:
            return
        if not P_set and not X:
            if R:
                max_cliques.append(list(R))
            return
        candidates = list(P_set | X)
        if not candidates:
            return
        pivot = max(candidates, key=lambda u: len(adj[u] & P_set))
        for v in list(P_set - adj[pivot]):
            bronk(R | {v}, P_set & adj[v], X & adj[v])
            P_set = P_set - {v}
            X = X | {v}
            if time.time() > deadline:
                return
    
    bronk(set(), set(range(n)), set())
    max_cliques.sort(key=lambda c: -len(c))
    
    best_branches = None
    best_paths = None
    best_size = 0
    
    # Try each maximal clique, then extend greedily
    for clique in max_cliques:
        if time.time() > deadline:
            break
        if len(clique) <= best_size:
            continue
        
        paths = check_and_find_paths(clique)
        if paths is not None and len(clique) > best_size:
            best_size = len(clique)
            best_branches = list(clique)
            best_paths = paths
        
        # Greedy extension
        current = list(clique)
        if paths is None:
            continue
        remaining = [v for v in range(n) if v not in set(current)]
        remaining.sort(key=lambda v: -len(adj[v] & set(current)))
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v in remaining[:]:
                if time.time() > deadline:
                    break
                test = current + [v]
                if len(test) <= best_size:
                    continue
                test_paths = check_and_find_paths(test)
                if test_paths is not None:
                    current = test
                    best_size = len(current)
                    best_branches = list(current)
                    best_paths = test_paths
                    remaining.remove(v)
                    remaining.sort(key=lambda w: -len(adj[w] & set(current)))
                    improved = True
                    break
    
    # ---- Randomized search: try random branch sets and extend ----
    if time.time() < deadline - 5:
        random.seed(42)
        # Try various random subsets of high-degree vertices
        high_deg = sorted(range(n), key=lambda v: -len(adj[v]))
        
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Random seed set of size 3-5 from high-degree vertices
            seed_size = random.randint(3, min(5, n))
            seed = random.sample(high_deg[:12], seed_size)
            
            paths = check_and_find_paths(seed)
            if paths is None:
                continue
            
            current = list(seed)
            remaining = [v for v in range(n) if v not in set(current)]
            remaining.sort(key=lambda v: -len(adj[v] & set(current)))
            
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                # Try adding vertices in order of connectivity to current
                for v in remaining[:]:
                    if time.time() > deadline - 5:
                        break
                    test = current + [v]
                    if len(test) <= best_size:
                        continue
                    test_paths = check_and_find_paths(test)
                    if test_paths is not None:
                        current = test
                        best_size = len(current)
                        best_branches = list(current)
                        best_paths = test_paths
                        remaining.remove(v)
                        remaining.sort(key=lambda w: -len(adj[w] & set(current)))
                        improved = True
                        break
    
    # ---- Also try: start from pairs and build up ----
    if time.time() < deadline - 3:
        # Try all pairs of high-degree vertices as seeds
        hd = high_deg[:8]
        for i in range(len(hd)):
            if time.time() > deadline - 3:
                break
            for j in range(i + 1, len(hd)):
                if time.time() > deadline - 3:
                    break
                seed = [hd[i], hd[j]]
                paths = check_and_find_paths(seed)
                if paths is None:
                    continue
                current = list(seed)
                remaining = [v for v in range(n) if v not in set(current)]
                remaining.sort(key=lambda v: -len(adj[v] & set(current)))
                
                improved = True
                while improved and time.time() < deadline - 3:
                    improved = False
                    for v in remaining[:]:
                        if time.time() > deadline - 3:
                            break
                        test = current + [v]
                        if len(test) <= best_size:
                            continue
                        test_paths = check_and_find_paths(test)
                        if test_paths is not None:
                            current = test
                            best_size = len(current)
                            best_branches = list(current)
                            best_paths = test_paths
                            remaining.remove(v)
                            remaining.sort(key=lambda w: -len(adj[w] & set(current)))
                            improved = True
                            break
    
    # Fallback
    if best_branches is None:
        best_branches = [0, 1]
        best_paths = {(0, 1): [0, 1]}
    
    branches = [int(b) for b in sorted(best_branches)]
    paths_list = []
    for u, v in combinations(sorted(best_branches), 2):
        p = best_paths.get((u, v), [u, v])
        paths_list.append([int(x) for x in p])
    
    return {'branches': branches, 'paths': paths_list}
