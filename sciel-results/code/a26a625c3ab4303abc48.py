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
    
    start_time = time.time()
    deadline = start_time + 85.0

    def find_k_shortest_paths(src, dst, forbidden, k=5):
        """Find up to k shortest paths from src to dst avoiding forbidden vertices."""
        if src == dst:
            return [[src]]
        all_paths = []
        visited = {src: [src]}
        queue = deque([(src, [src])])
        while queue and len(all_paths) < k:
            node, path = queue.popleft()
            for nb in sorted(adj[node]):
                if nb == dst:
                    new_path = path + [dst]
                    if new_path not in all_paths:
                        all_paths.append(new_path)
                    if len(all_paths) >= k:
                        return all_paths
                elif nb not in forbidden and nb not in visited:
                    visited[nb] = path + [nb]
                    queue.append((nb, path + [nb]))
        return all_paths

    def try_csp(branches, deadline):
        """CSP backtracking: assign one path per pair, no shared internal vertices."""
        branches = list(branches)
        branch_set = set(branches)
        pairs = list(itertools.combinations(branches, 2))
        if not pairs:
            return []
        
        pair_candidates = {}
        for u, v in pairs:
            forbidden = branch_set - {u, v}
            paths = find_k_shortest_paths(u, v, forbidden, k=5)
            if not paths:
                return None
            pair_candidates[(u, v)] = paths
        
        # Most constrained first
        pair_order = sorted(pairs, key=lambda p: len(pair_candidates[p]))
        
        assignment = {}
        used_internal = set()
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == len(pair_order):
                return True
            u, v = pair_order[idx]
            for path in pair_candidates[(u, v)]:
                internal = set(path[1:-1])
                if internal & used_internal:
                    continue
                if internal & branch_set:
                    continue
                assignment[(u, v)] = path
                used_internal.update(internal)
                if backtrack(idx + 1):
                    return True
                used_internal.difference_update(internal)
                del assignment[(u, v)]
            return False
        
        if backtrack(0):
            return [assignment[p] for p in pairs]
        return None

    def try_greedy(branches, deadline):
        """Greedy with multiple orderings."""
        branches = list(branches)
        branch_set = set(branches)
        pairs = list(itertools.combinations(branches, 2))
        if not pairs:
            return []
        
        orderings = []
        # Direct edges first
        orderings.append(sorted(pairs, key=lambda p: (0 if p[1] in adj[p[0]] else 1, len(adj[p[0]]) + len(adj[p[1]]))))
        # Non-direct first
        orderings.append(sorted(pairs, key=lambda p: (1 if p[1] in adj[p[0]] else 0, len(adj[p[0]]) + len(adj[p[1]]))))
        # Random
        shuffled = list(pairs)
        random.shuffle(shuffled)
        orderings.append(shuffled)
        # Degree-based
        orderings.append(sorted(pairs, key=lambda p: len(adj[p[0]]) + len(adj[p[1]]), reverse=True))
        
        for pair_order in orderings:
            if time.time() > deadline:
                return None
            paths = {}
            used_internal = set()
            success = True
            for u, v in pair_order:
                forbidden = branch_set - {u, v} | used_internal
                path_list = find_k_shortest_paths(u, v, forbidden, k=1)
                if not path_list:
                    success = False
                    break
                paths[(u, v)] = path_list[0]
                for x in path_list[0][1:-1]:
                    used_internal.add(x)
            if success:
                return [paths[p] for p in pairs]
        return None

    def try_build(branches, deadline):
        """Try CSP then greedy."""
        result = try_csp(branches, deadline)
        if result is not None:
            return result
        result = try_greedy(branches, deadline)
        if result is not None:
            return result
        return None

    def find_max_clique(deadline):
        """Bron-Kerbosch with pivot to find largest clique."""
        best_clique = []
        
        def bronk(R, P_set, X):
            if time.time() > deadline:
                return
            if not P_set and not X:
                if len(R) > len(best_clique):
                    best_clique[:] = list(R)
                return
            if P_set or X:
                pivot = max(P_set | X, key=lambda v: len(adj[v] & P_set))
                candidates = [v for v in P_set if v not in adj[pivot]]
            else:
                return
            for v in candidates:
                bronk(R | {v}, P_set & adj[v], X & adj[v])
                P_set.remove(v)
                X.add(v)
        
        bronk(set(), set(range(n)), set())
        return best_clique

    def expand(branches, deadline):
        """Greedily try to add more branch vertices."""
        current = list(branches)
        current_set = set(current)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try all vertices not in current
            candidates = [v for v in range(n) if v not in current_set]
            # Sort by degree descending
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)
            for cand in candidates:
                if time.time() > deadline:
                    break
                test = current + [cand]
                paths = try_build(test, deadline)
                if paths is not None:
                    current = test
                    current_set.add(cand)
                    improved = True
                    break
        return current

    # Phase 1: Find a large clique
    clique = find_max_clique(deadline)
    if len(clique) < 2:
        # Fallback: find a large set of mutually connected vertices
        clique = [0, 1]  # first edge
    
    # Phase 2: Verify and expand from clique
    best_branches = list(clique)
    best_paths = try_build(best_branches, deadline)
    if best_paths is None:
        # Try smaller subsets of the clique
        for size in range(len(clique) - 1, 1, -1):
            for subset in itertools.combinations(clique, size):
                paths = try_build(list(subset), deadline)
                if paths is not None:
                    best_branches = list(subset)
                    best_paths = paths
                    break
            if best_paths is not None:
                break
    
    # Phase 3: Expand
    if best_paths is not None:
        expanded = expand(best_branches, deadline)
        if len(expanded) > len(best_branches):
            paths = try_build(expanded, deadline)
            if paths is not None:
                best_branches = expanded
                best_paths = paths
    
    # Phase 4: Try different starting cliques/subsets for better expansion
    if time.time() < deadline:
        # Try expanding from various pairs and triples
        all_vertices = list(range(n))
        all_vertices.sort(key=lambda x: len(adj[x]), reverse=True)
        
        tried = set()
        for i in range(min(10, len(all_vertices))):
            if time.time() > deadline:
                break
            start = all_vertices[i]
            for j in range(i + 1, min(15, len(all_vertices))):
                if time.time() > deadline:
                    break
                end = all_vertices[j]
                pair = tuple(sorted([start, end]))
                if pair in tried:
                    continue
                tried.add(pair)
                # Only try if they're adjacent or have a path
                if end in adj[start]:
                    test = [start, end]
                    paths = try_build(test, deadline)
                    if paths is not None:
                        expanded = expand(test, deadline)
                        if len(expanded) > len(best_branches):
                            paths2 = try_build(expanded, deadline)
                            if paths2 is not None:
                                best_branches = expanded
                                best_paths = paths2
    
    # Final: ensure we have at least 2 branches
    if best_branches is None or len(best_branches) < 2:
        u, v = P['edges'][0]
        best_branches = [u, v]
        best_paths = [[u, v]]
    
    # Build output
    branches_sorted = sorted(best_branches)
    out_paths = []
    for i in range(len(branches_sorted)):
        for j in range(i + 1, len(branches_sorted)):
            u, v = branches_sorted[i], branches_sorted[j]
            # Find the path
            path = None
            for p in best_paths:
                if p[0] == u and p[-1] == v:
                    path = p
                    break
                if p[0] == v and p[-1] == u:
                    path = list(reversed(p))
                    break
            if path is None:
                path = [u, v]
            out_paths.append(path)
    
    return {
        'branches': [int(b) for b in branches_sorted],
        'paths': out_paths
    }
# EVOLVE-BLOCK-END
