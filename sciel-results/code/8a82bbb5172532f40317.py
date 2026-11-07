import time
import itertools
from collections import deque

def construct():
    P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    start_time = time.time()
    deadline = start_time + 85.0

    def time_left():
        return deadline - time.time()

    def find_disjoint_paths(branches, max_depth=8):
        """
        Find vertex-disjoint paths between all pairs of branches.
        Uses a layered approach: route pairs in order of "difficulty" (fewest candidates first).
        """
        if len(branches) < 2:
            return None

        branch_set = frozenset(branches)
        pairs = list(itertools.combinations(branches, 2))
        
        # Pre-compute candidate paths for each pair
        # To save time, we limit the number of paths we consider per pair
        max_candidates_per_pair = 20

        # We need to order the pairs. A good heuristic is to start with pairs that have
        # fewer direct connections or are "farther" apart, or simply random/greedy.
        # Let's try sorting by the number of common neighbors or just randomize and try multiple orders.
        
        def get_candidate_paths(u, v, blocked):
            """Find simple paths from u to v avoiding blocked vertices and other branches."""
            paths = []
            # BFS/DFS
            # Since n is small (24), we can use DFS with pruning.
            # To speed up, we can use BFS to find shortest paths first, then longer ones if needed.
            # But for disjointness, shortest paths are often better (use fewer resources).
            
            def dfs(path, visited):
                if len(paths) >= max_candidates_per_pair:
                    return
                curr = path[-1]
                if curr == v:
                    if len(path) > 1:
                        # Check if internal vertices are blocked
                        internal = set(path[1:-1])
                        if not (internal & blocked) and not (internal & branch_set):
                            paths.append(list(path))
                    return
                
                # Depth limit
                if len(path) - 1 >= max_depth:
                    return
                
                # Heuristic: sort neighbors by degree or distance to v
                # For now, just iterate
                for nb in adj[curr]:
                    if nb in visited:
                        continue
                    if nb in branch_set and nb != v:
                        continue
                    if nb in blocked:
                        continue
                    
                    visited.add(nb)
                    dfs(path + [nb], visited)
                    visited.remove(nb)
            
            dfs([u], {u})
            
            # Sort by length (shortest first)
            paths.sort(key=len)
            return paths

        # Backtracking search for disjoint paths
        def backtrack(remaining_pairs, blocked, assignment):
            if time.time() > deadline:
                return False
            if not remaining_pairs:
                return True
            
            # Choose the next pair to route.
            # Heuristic: choose the pair with the fewest available candidate paths given current blocked set.
            # This is the "most constrained" variable.
            
            best_pair_idx = -1
            best_candidates = None
            min_count = float('inf')
            
            for i, (u, v) in enumerate(remaining_pairs):
                cands = get_candidate_paths(u, v, blocked)
                count = len(cands)
                if count == 0:
                    return False # Dead end
                if count < min_count:
                    min_count = count
                    best_pair_idx = i
                    best_candidates = cands
            
            if best_pair_idx == -1:
                return False
            
            u, v = remaining_pairs[best_pair_idx]
            rest_pairs = remaining_pairs[:best_pair_idx] + remaining_pairs[best_pair_idx+1:]
            
            for path in best_candidates:
                internal = set(path[1:-1])
                new_blocked = blocked | internal
                assignment[(u, v)] = path
                
                if backtrack(rest_pairs, new_blocked, assignment):
                    return True
                
                del assignment[(u, v)]
            
            return False

        blocked = set()
        assignment = {}
        if backtrack(pairs, blocked, assignment):
            return [assignment[p] for p in pairs]
        return None

    # Strategy:
    # 1. Find a large clique using Bron-Kerbosch.
    # 2. Try to extend the clique by adding vertices that are connected to all existing branches
    #    (or can be connected via disjoint paths).
    # 3. If extension fails, try other cliques or random sets.

    def bron_kerbosch(R, P, X):
        """Standard Bron-Kerbosch with pivot."""
        if not P and not X:
            yield R
            return
        
        # Pivot selection
        if P:
            u = max(P, key=lambda x: len(adj[x] & P) - len(adj[x] & X))
            candidates = P - adj[u]
        else:
            candidates = P
        
        for v in candidates:
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            yield from bron_kerbosch(new_R, new_P, new_X)
            P.remove(v)
            X.add(v)

    # Find all maximal cliques
    all_cliques = []
    P_set = set(range(n))
    for clique in bron_kerbosch(set(), P_set, set()):
        all_cliques.append(clique)
    
    # Sort cliques by size (largest first)
    all_cliques.sort(key=len, reverse=True)
    
    best_branches = None
    best_paths = None
    best_size = 0

    # Try cliques first
    for clique in all_cliques:
        if time.time() > deadline - 5:
            break
        
        branches = list(clique)
        paths = find_disjoint_paths(branches)
        if paths:
            if len(branches) > best_size:
                best_branches = branches
                best_paths = paths
                best_size = len(branches)
        
        # Try to extend the clique
        # Add vertices that are adjacent to all vertices in the clique? No, that's just a larger clique.
        # Add vertices that are NOT in the clique but can be connected via disjoint paths.
        # This is harder. Let's just try random additions.
        
        # Attempt to add one vertex at a time
        current_branches = list(branches)
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            candidates = [v for v in range(n) if v not in current_branches]
            # Sort candidates by connectivity to current branches
            def score(v):
                return sum(1 for b in current_branches if b in adj[v])
            candidates.sort(key=score, reverse=True)
            
            for v in candidates[:5]: # Try top 5
                new_branches = current_branches + [v]
                paths = find_disjoint_paths(new_branches)
                if paths:
                    current_branches = new_branches
                    best_paths = paths
                    best_branches = current_branches
                    best_size = len(current_branches)
                    changed = True
                    break

    # If we didn't find anything good, try random sets
    # Start from high-degree vertices
    vertices = list(range(n))
    vertices.sort(key=lambda v: -len(adj[v]))
    
    for target_size in [8, 7, 6, 5, 4, 3, 2]:
        if time.time() > deadline - 5:
            break
        if best_size >= target_size:
            continue
            
        # Try random samples from top vertices
        pool = vertices[:15]
        num_tries = 20
        for _ in range(num_tries):
            if time.time() > deadline - 5:
                break
            
            if len(pool) < target_size:
                continue
                
            rand_branches = random.sample(pool, target_size)
            paths = find_disjoint_paths(rand_branches)
            if paths:
                best_branches = rand_branches
                best_paths = paths
                best_size = target_size
                break

    # Fallback
    if best_size == 0:
        for u in range(n):
            for v in range(u+1, n):
                if v in adj[u]:
                    best_branches = [u, v]
                    best_paths = [[u, v]]
                    best_size = 2
                    break
            if best_size > 0:
                break

    if best_size >= 2 and best_paths is not None:
        branches = [int(b) for b in best_branches]
        formatted_paths = []
        for p in best_paths:
            formatted_paths.append([int(x) for x in p])
        
        return {'branches': branches, 'paths': formatted_paths}
    
    if P['edges']:
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    else:
        return {'branches': [0, 1], 'paths': [[0, 1]]}

import random

# EVOLVE-BLOCK-END
