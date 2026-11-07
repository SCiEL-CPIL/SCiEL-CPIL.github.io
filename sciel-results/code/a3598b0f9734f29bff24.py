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
    deadline = start_time + 88.0 # Leave 2s margin for 90s limit

    def find_paths_for_branches(branches):
        """
        Attempts to find internally disjoint paths for all pairs in 'branches'.
        Returns a dict mapping (u, v) to path list, or None if impossible.
        """
        branches = list(branches)
        branch_set = set(branches)
        pairs = list(itertools.combinations(branches, 2))
        
        if not pairs:
            return {}

        # Pre-compute potential paths? No, too many.
        # We will use backtracking.
        
        # Order pairs: those with fewer available intermediate vertices first?
        # Or just fixed order for consistency, then shuffle?
        # Let's sort by distance (approx) or just fixed.
        # A good heuristic: pairs that are NOT directly connected are harder.
        # Also, pairs with fewer common neighbors are harder.
        
        def path_difficulty(p):
            u, v = p
            if v in adj[u]:
                return 0 # Direct edge, very easy
            # Count common neighbors (potential length 2 paths)
            common = len(adj[u] & adj[v])
            return -common # Fewer common neighbors = higher difficulty number = sort later? 
            # Actually, we want to assign HARD constraints first in CSP to prune early.
            # So we want difficulty to be high for hard ones.
            # Let's define difficulty as: 0 if direct, 1 if common neighbor, 2 otherwise.
            # And we want to process high difficulty first? 
            # Standard CSP: Most Constrained Variable First.
            # Here, "variable" is the pair. "Constraint" is availability of disjoint paths.
            # If a pair has few options, it's constrained.
            # Let's just use a heuristic: sort by (not direct, -common_neighbors)
            # Direct: 0, -large. Non-direct: 1, -small.
            # This puts non-direct with few options first? 
            # Let's try sorting by: (1 if not direct else 0, -len(adj[u]&adj[v]))
            # This puts non-direct first, and among them, those with fewer common neighbors first.
        
        pair_order = sorted(pairs, key=lambda p: (0 if p[1] in adj[p[0]] else 1, -len(adj[p[0]] & adj[p[1]])))
        
        assignment = {}
        used_internal = set()
        
        # To speed up, we can pre-generate a list of candidate paths for each pair
        # avoiding branch_set.
        # But the "used_internal" set changes.
        # So we must generate on the fly or filter.
        
        def get_candidate_paths(u, v, forbidden_internal):
            """Find simple paths from u to v avoiding forbidden_internal and branch_set \ {u,v}."""
            # BFS is good for short paths.
            # We want short paths to leave room for others.
            paths = []
            visited = {u: [u]}
            queue = deque([(u, [u])])
            # Limit search depth to avoid explosion? 
            # Max path length in a subdivision shouldn't be too long. 
            # n=24, max simple path length 24.
            # But we want short ones.
            while queue:
                if time.time() > deadline - 2:
                    return paths # Timeout, return what we have
                node, path = queue.popleft()
                for nb in sorted(adj[node]):
                    if nb == v:
                        paths.append(path + [v])
                        # We can break if we have enough? 
                        # Or collect a few.
                    elif nb not in forbidden_internal and nb not in branch_set:
                        if nb not in visited:
                            visited[nb] = path + [nb]
                            queue.append((nb, path + [nb]))
                    # Note: we don't add v to visited, we just return path.
                    # We don't visit branch_set vertices as intermediates.
            
            # Sort paths by length
            paths.sort(key=len)
            return paths

        def backtrack(idx):
            if time.time() > deadline - 2:
                return False
            if idx == len(pair_order):
                return True
            
            u, v = pair_order[idx]
            
            # Generate candidates
            candidates = get_candidate_paths(u, v, used_internal)
            
            # If no candidates, fail
            if not candidates:
                return False
            
            # Try each candidate
            for path in candidates:
                internal = set(path[1:-1])
                # Check disjointness (should be guaranteed by get_candidate_paths if passed used_internal)
                # But we passed used_internal, so it should be fine.
                # However, get_candidate_paths might return paths that share vertices with each other?
                # No, we pick ONE path.
                
                # Assign
                assignment[(u, v)] = path
                used_internal.update(internal)
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                used_internal.difference_update(internal)
                del assignment[(u, v)]
            
            return False

        if backtrack(0):
            return assignment
        return None

    def find_max_clique(deadline):
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
        
        bronk(frozenset(), frozenset(range(n)), frozenset())
        cliques.sort(key=len, reverse=True)
        return cliques

    # Phase 1: Find largest clique and verify it's a valid subdivision (it is, by definition)
    cliques = find_max_clique(deadline - 5)
    
    best_result = None
    best_size = 0
    
    # Try the largest cliques
    for clique in cliques[:20]:
        if time.time() > deadline - 10:
            break
        if len(clique) <= best_size:
            continue
        # A clique is always a valid subdivision (paths are just edges)
        branches = list(clique)
        paths = {}
        for u, v in itertools.combinations(branches, 2):
            paths[(u, v)] = [u, v]
        best_result = {'branches': branches, 'paths': [paths[p] for p in itertools.combinations(branches, 2)]}
        best_size = len(branches)

    # Phase 2: Try to expand the best clique or find larger non-clique subdivisions
    # Strategy: Start with a set of candidates (top degree vertices) and try to find a subset that works.
    
    # Get top 12 vertices by degree
    top_vertices = sorted(range(n), key=lambda x: -len(adj[x]))
    candidates = top_vertices[:12]
    
    # Also add vertices from the largest clique
    for v in cliques[0] if cliques else []:
        if v not in candidates:
            candidates.append(v)
    
    # Randomize candidates order for search
    random.shuffle(candidates)
    
    # Try to build a subdivision of size 5, 6, 7...
    # We can use a greedy expansion or a search over subsets.
    
    # Let's try all subsets of size 5 from candidates? 
    # C(12, 5) = 792. That's small.
    # C(12, 6) = 924.
    # C(12, 7) = 792.
    # C(12, 8) = 495.
    
    # We will iterate through subset sizes from 8 down to best_size + 1.
    # For each size, iterate through combinations.
    
    # To avoid duplicates and speed up, we can fix the first vertex?
    # Or just use itertools.combinations.
    
    found_larger = False
    for size in range(8, best_size, -1):
        if time.time() > deadline - 5:
            break
        if found_larger:
            break
            
        # Generate combinations of candidates of this size
        # To make it faster, we can restrict to subsets that contain a large clique?
        # Or just try all.
        
        for combo in itertools.combinations(candidates, size):
            if time.time() > deadline - 5:
                break
            
            # Quick check: if any pair in combo is not connected by a path in the graph (ignoring disjointness), skip.
            # But they are in the graph, so paths exist.
            # The hard part is disjointness.
            
            # Try to find paths
            paths_dict = find_paths_for_branches(combo)
            if paths_dict is not None:
                # Valid subdivision found
                branches = list(combo)
                pairs = list(itertools.combinations(branches, 2))
                paths_list = [paths_dict[p] for p in pairs]
                best_result = {'branches': branches, 'paths': paths_list}
                best_size = size
                found_larger = True
                break # Found one of this size, try larger? No, we are going down in size. 
                # Wait, the loop is `for size in range(8, best_size, -1)`. 
                # So if we find size 8, we break the inner loop and the outer loop continues?
                # No, `found_larger` breaks the outer loop.
                # This is correct: we want the MAXIMUM size. So we start with 8, 7, 6...
                # If we find 8, we stop.
                # But what if 9 exists? We didn't check 9.
                # We should check larger sizes first.
                # My range is `range(8, best_size, -1)`. This goes 8, 7, ...
                # I should check up to len(candidates).
                
                # Let's restart the search logic to check larger sizes first.
                pass 

    # Restart search logic properly
    # Check sizes from len(candidates) down to best_size + 1
    
    for size in range(len(candidates), best_size, -1):
        if time.time() > deadline - 5:
            break
        
        # Heuristic: Only consider combinations that contain at least one edge? 
        # Or just all.
        
        # To reduce search space, we can prioritize combinations with more edges within the subset.
        # But itertools.combinations doesn't sort by internal edges.
        # We can generate all, sort by internal edge count, and try best first.
        
        combos = list(itertools.combinations(candidates, size))
        
        # Sort by number of edges in the induced subgraph
        def count_edges(combo):
            c_set = set(combo)
            count = 0
            for u in combo:
                count += len(adj[u] & c_set)
            return count // 2
        
        combos.sort(key=count_edges, reverse=True)
        
        # Limit number of combos to try to avoid timeout
        max_combos_to_try = 100
        for combo in combos[:max_combos_to_try]:
            if time.time() > deadline - 5:
                break
            
            paths_dict = find_paths_for_branches(combo)
            if paths_dict is not None:
                branches = list(combo)
                pairs = list(itertools.combinations(branches, 2))
                paths_list = [paths_dict[p] for p in pairs]
                best_result = {'branches': branches, 'paths': paths_list}
                best_size = size
                break # Found max size, stop
        
        if best_size >= size:
            break # Found a subdivision of this size, no need to check smaller ones

    # Phase 3: Local Search / Hill Climbing
    # If we found a valid subdivision, try to swap vertices to increase size?
    # Or try to add a vertex?
    
    if best_result is not None:
        current_branches = list(best_result['branches'])
        
        # Try to add one more vertex
        for v in range(n):
            if time.time() > deadline - 2:
                break
            if v in current_branches:
                continue
            
            new_branches = current_branches + [v]
            paths_dict = find_paths_for_branches(new_branches)
            if paths_dict is not None:
                branches = list(new_branches)
                pairs = list(itertools.combinations(branches, 2))
                paths_list = [paths_dict[p] for p in pairs]
                best_result = {'branches': branches, 'paths': paths_list}
                best_size = len(branches)
                break # Found a larger one, stop adding? Or continue?
                # If we add one, maybe we can add another?
                # Let's loop again.
                # But time is tight.
        
        # Try swapping a low-degree vertex for a high-degree one?
        # This is complex. Let's stick to the result.

    # Fallback
    if best_result is None:
        # Use the largest clique
        if cliques:
            branches = cliques[0]
            pairs = list(itertools.combinations(branches, 2))
            paths = [[u, v] for u, v in pairs]
            best_result = {'branches': branches, 'paths': paths}
        else:
            u, v = P['edges'][0]
            best_result = {'branches': [u, v], 'paths': [[u, v]]}

    # Final validation and formatting
    branches = [int(b) for b in best_result['branches']]
    paths = [[int(x) for x in p] for p in best_result['paths']]
    
    # Double check validity
    branch_set = set(branches)
    used_internal = set()
    valid = True
    for u, v in itertools.combinations(branches, 2):
        found = False
        for p in paths:
            if p[0] == u and p[-1] == v:
                found = True
                for x in p[1:-1]:
                    if x in branch_set or x in used_internal:
                        valid = False
                    used_internal.add(x)
                # Check adjacency
                for i in range(len(p)-1):
                    if p[i+1] not in adj[p[i]]:
                        valid = False
                break
        if not found:
            valid = False
            
    if not valid:
        # Fallback to clique
        if cliques:
            branches = [int(b) for b in cliques[0]]
            pairs = list(itertools.combinations(branches, 2))
            paths = [[int(u), int(v)] for u, v in pairs]
            return {'branches': branches, 'paths': paths}
        else:
            u, v = P['edges'][0]
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    return {'branches': branches, 'paths': paths}
# EVOLVE-BLOCK-END
