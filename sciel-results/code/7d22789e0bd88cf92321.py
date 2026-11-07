"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    deadline = time.time() + 85.0
    n = P['n']
    
    # Build adjacency sets and list
    adj_sets = [set() for _ in range(n)]
    adj_list = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj_sets[u].add(v)
        adj_sets[v].add(u)
        adj_list[u].add(v)
        adj_list[v].add(u)
    
    def find_clique_greedy():
        """Find a large clique using greedy heuristic with multiple orderings."""
        best_clique = set()
        
        # Try multiple vertex orderings
        # Order by degree descending
        for order in [sorted(range(n), key=lambda x: -len(adj_sets[x])),
                      sorted(range(n), key=lambda x: len(adj_sets[x]))]:
            clique = set()
            for v in order:
                if all(v in adj_sets[u] for u in clique):
                    clique.add(v)
            if len(clique) > len(best_clique):
                best_clique = clique
        
        return list(best_clique)
    
    def find_clique_bron_kerbosch():
        """Find maximum clique using Bron-Kerbosch with pivot."""
        best_clique = [0, None]
        
        def bron_kerbosch(R, P_set, X):
            if time.time() > deadline:
                return
            if not P_set and not X:
                if len(R) > best_clique[0]:
                    best_clique[0] = len(R)
                    best_clique[1] = set(R)
                return
            
            # Choose pivot
            if P_set:
                pivot = max(P_set, key=lambda v: len(P_set & adj_sets[v]) + len(X & adj_sets[v]))
                candidates = P_set - adj_sets[pivot]
            else:
                candidates = X
            
            for v in list(candidates):
                if time.time() > deadline:
                    return
                bron_kerbosch(R | {v}, P_set & adj_sets[v], X & adj_sets[v])
                P_set.remove(v)
                X.add(v)
        
        bron_kerbosch(set(), set(range(n)), set())
        
        if best_clique[1] is not None:
            return list(best_clique[1])
        return [0, 1]
    
    def find_clique_maximal():
        """Find a maximal clique by expanding from each vertex."""
        best_clique = set()
        
        for start in range(n):
            if time.time() > deadline:
                break
            clique = {start}
            candidates = set(adj_sets[start])
            changed = True
            while changed:
                changed = False
                for v in list(candidates):
                    if all(v in adj_sets[u] for u in clique):
                        clique.add(v)
                        candidates.discard(v)
                        changed = True
                # Try to add more
                new_candidates = set()
                for v in list(clique):
                    new_candidates |= adj_sets[v]
                new_candidates -= clique
                for v in list(new_candidates):
                    if all(v in adj_sets[u] for u in clique):
                        clique.add(v)
                        changed = True
            if len(clique) > len(best_clique):
                best_clique = clique
        
        return list(best_clique)
    
    # Find a good clique
    clique1 = find_clique_greedy()
    clique2 = find_clique_maximal()
    
    # Try Bron-Kerbosch if time permits
    if time.time() < deadline - 5.0:
        clique3 = find_clique_bron_kerbosch()
        candidates = [clique1, clique2, clique3]
    else:
        candidates = [clique1, clique2]
    
    # Pick the largest clique
    best_clique = max(candidates, key=len)
    
    # Ensure at least 2 branches
    if len(best_clique) < 2:
        best_clique = [0, 1]
    
    # Sort for consistency
    best_clique = sorted(best_clique)
    
    # Build paths: for each pair of branch vertices, find a simple path
    # using only non-branch vertices in the interior
    branches_set = set(best_clique)
    non_branches = set(range(n)) - branches_set
    
    def find_path(src, dst, forbidden):
        """Find a simple path from src to dst avoiding forbidden vertices in interior."""
        if src == dst:
            return [src]
        
        # BFS
        visited = {src}
        parent = {src: None}
        queue = deque([src])
        
        while queue:
            u = queue.popleft()
            if u == dst:
                # Reconstruct path
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for v in adj_list[u]:
                if v not in visited:
                    if v == dst or v not in forbidden:
                        visited.add(v)
                        parent[v] = u
                        queue.append(v)
        
        return None
    
    paths = {}
    valid = True
    
    for i, j in combinations(range(len(best_clique)), 2):
        u, v = best_clique[i], best_clique[j]
        # Forbidden: all branch vertices except u and v
        forbidden = branches_set - {u, v}
        path = find_path(u, v, forbidden)
        if path is None:
            valid = False
            break
        paths[(u, v)] = path
    
    if not valid:
        # Fallback: use direct edges if possible, or reduce clique
        # Try to find a valid subset
        best_subset = None
        for size in range(len(best_clique), 1, -1):
            for subset in combinations(best_clique, size):
                subset_set = set(subset)
                all_paths = True
                sub_paths = {}
                for i, j in combinations(range(size), 2):
                    u, v = subset[i], subset[j]
                    forbidden = subset_set - {u, v}
                    path = find_path(u, v, forbidden)
                    if path is None:
                        all_paths = False
                        break
                    sub_paths[(u, v)] = path
                if all_paths:
                    best_subset = subset
                    paths = sub_paths
                    break
            if best_subset is not None:
                break
        
        if best_subset is None:
            # Last resort: just use an edge
            u, v = P['edges'][0]
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        
        best_clique = list(best_subset)
    
    # Build final result
    branch_list = [int(v) for v in best_clique]
    path_list = []
    for i, j in combinations(range(len(branch_list)), 2):
        u, v = branch_list[i], branch_list[j]
        key = (min(u, v), max(u, v))
        p = paths.get(key)
        if p is None:
            # Try reversed
            key2 = (max(u, v), min(u, v))
            p = paths.get(key2)
        if p is not None:
            path_list.append([int(x) for x in p])
        else:
            # Should not happen, but fallback to direct edge
            path_list.append([int(u), int(v)])
    
    return {'branches': branch_list, 'paths': path_list}

# EVOLVE-BLOCK-END
