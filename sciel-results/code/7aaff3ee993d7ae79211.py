import time
import sys
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s limit)
    
    n = P['n']
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find maximum clique using Bron-Kerbosch with pivot
    best_clique = []
    
    def bronk(R, P_set, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        # Choose pivot u from P_set ∪ X
        candidates = list(P_set | X)
        if not candidates:
            return
        # Pick pivot with max degree in P_set
        pivot = max(candidates, key=lambda u: len(adj[u] & P_set))
        # For each v in P_set \ N(pivot)
        for v in list(P_set - adj[pivot]):
            bronk(R | {v}, P_set & adj[v], X & adj[v])
            P_set = P_set - {v}
            X = X | {v}
            if time.time() > deadline:
                return
    
    bronk(set(), set(range(n)), set())
    
    def find_path_bfs(src, dst, forbidden):
        """Find shortest path from src to dst avoiding forbidden vertices (except endpoints)."""
        if src == dst:
            return [src]
        visited = {src}
        parent = {src: None}
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            for nb in adj[cur]:
                if nb not in visited and (nb == dst or nb not in forbidden):
                    visited.add(nb)
                    parent[nb] = cur
                    if nb == dst:
                        # Reconstruct path
                        path = []
                        node = dst
                        while node is not None:
                            path.append(node)
                            node = parent[node]
                        path.reverse()
                        return path
                    queue.append(nb)
        return None
    
    def try_assign_paths(branches, order=None):
        """Try to assign disjoint paths for all pairs. Returns paths dict or None."""
        b_set = set(branches)
        k = len(branches)
        pairs = list(combinations(sorted(branches), 2))
        
        if order is None:
            # Default: direct edges first, then by distance
            def pair_key(p):
                u, v = p
                if v in adj[u]:
                    return (0, 0, u, v)
                # Estimate distance via BFS depth
                return (1, 0, u, v)
            order = sorted(pairs, key=pair_key)
        
        paths = {}
        used_interior = set()
        
        for u, v in order:
            if v in adj[u]:
                paths[(u, v)] = [u, v]
            else:
                forbidden = b_set - {u, v}
                forbidden |= used_interior
                path = find_path_bfs(u, v, forbidden)
                if path is None:
                    return None
                paths[(u, v)] = path
                for mid in path[1:-1]:
                    used_interior.add(mid)
        
        return paths
    
    def can_form_clique_subdivision(branches):
        """Check if branches can form a clique subdivision. Try multiple orderings."""
        b_set = set(branches)
        k = len(branches)
        pairs = list(combinations(sorted(branches), 2))
        
        # Try default ordering
        result = try_assign_paths(branches, None)
        if result is not None:
            return result
        
        # Try random orderings
        num_attempts = min(50, max(10, 200 // k))
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            random_order = pairs[:]
            random.shuffle(random_order)
            result = try_assign_paths(branches, random_order)
            if result is not None:
                return result
        
        return None
    
    # Start with max clique
    current_branches = list(best_clique)
    current_paths = can_form_clique_subdivision(current_branches)
    
    if current_paths is None:
        # Fallback: use direct edges only for the clique
        current_paths = {}
        for u, v in combinations(sorted(current_branches), 2):
            current_paths[(u, v)] = [u, v]
    
    # Try to add more vertices greedily
    remaining = [v for v in range(n) if v not in set(current_branches)]
    improved = True
    while improved:
        improved = False
        for v in remaining[:]:
            if time.time() > deadline:
                break
            test_branches = current_branches + [v]
            test_paths = can_form_clique_subdivision(test_branches)
            if test_paths is not None:
                current_branches = test_branches
                current_paths = test_paths
                remaining.remove(v)
                improved = True
    
    # Build result
    branches = [int(b) for b in sorted(current_branches)]
    paths_list = []
    for u, v in combinations(sorted(branches), 2):
        p = current_paths.get((u, v), [u, v])
        paths_list.append([int(x) for x in p])
    
    return {'branches': branches, 'paths': paths_list}
