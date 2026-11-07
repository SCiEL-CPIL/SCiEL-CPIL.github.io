import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges_list = P['edges']
    edge_set = set()
    for e in edges_list:
        a, b = e
        if a > b:
            a, b = b, a
        edge_set.add((a, b))
    
    # Build adjacency matrix for fast lookup
    adj_matrix = [[False] * n for _ in range(n)]
    for a, b in edge_set:
        adj_matrix[a][b] = True
        adj_matrix[b][a] = True

    # Precompute edge index for O(1) lookup
    edge_index = {}
    for a, b in edge_set:
        edge_index[(a, b)] = True
        edge_index[(b, a)] = True

    def get_maximal_bicliques():
        """
        Enumerate all maximal bicliques in the graph.
        A maximal biclique is a biclique that cannot be extended by adding any vertex to either side.
        We use a simple but effective enumeration strategy based on edges.
        """
        bicliques = set()  # Store as frozensets to avoid duplicates
        
        # For each edge, try to expand it into a maximal biclique
        # We do this by considering all possible subsets of neighbors
        # But to keep it tractable, we use a heuristic-based enumeration
        
        # Strategy: For each edge (u, v), the maximal bicliques containing it
        # can be found by looking at common neighbors and expanding.
        # However, exhaustive enumeration is expensive. Instead, we'll generate
        # a good set of candidate bicliques using multiple strategies.
        
        # Strategy 1: For each edge, find the maximal biclique using the 
        # "maximal expansion" approach
        for u, v in edge_set:
            L = {u}
            R = {v}
            
            # Iteratively expand
            changed = True
            while changed:
                changed = False
                # Try to add to L
                for x in range(n):
                    if x in L or x in R:
                        continue
                    # Check if x connects to all in R
                    can_add = True
                    for y in R:
                        if not adj_matrix[x][y]:
                            can_add = False
                            break
                    if can_add:
                        L.add(x)
                        changed = True
                
                # Try to add to R
                for y in range(n):
                    if y in L or y in R:
                        continue
                    # Check if y connects to all in L
                    can_add = True
                    for x in L:
                        if not adj_matrix[x][y]:
                            can_add = False
                            break
                    if can_add:
                        R.add(y)
                        changed = True
            
            # Store the biclique (sorted for canonical form)
            biclique = (frozenset(L), frozenset(R))
            bicliques.add(biclique)
            # Also store the swapped version if different
            if L != R:
                biclique_swapped = (frozenset(R), frozenset(L))
                bicliques.add(biclique_swapped)
        
        # Strategy 2: For each vertex, find maximal bicliques where it's in L
        for center in range(n):
            # Find all maximal bicliques containing center in L
            # This is done by looking at subsets of N(center)
            neighbors = [x for x in range(n) if adj_matrix[center][x]]
            
            # For small neighbor sets, enumerate all subsets
            if len(neighbors) <= 8:
                from itertools import combinations
                for r in range(1, len(neighbors) + 1):
                    for subset in combinations(neighbors, r):
                        R = set(subset)
                        L = {center}
                        
                        # Expand L: find all vertices that connect to all in R
                        # and are not in R
                        expanded_L = {center}
                        for x in range(n):
                            if x in R:
                                continue
                            can_add = True
                            for y in R:
                                if not adj_matrix[x][y]:
                                    can_add = False
                                    break
                            if can_add:
                                expanded_L.add(x)
                        
                        # Expand R: find all vertices that connect to all in expanded_L
                        # and are not in expanded_L
                        expanded_R = set(R)
                        for y in range(n):
                            if y in expanded_L:
                                continue
                            can_add = True
                            for x in expanded_L:
                                if not adj_matrix[x][y]:
                                    can_add = False
                                    break
                            if can_add:
                                expanded_R.add(y)
                        
                        # Check if this is a valid biclique
                        valid = True
                        for l in expanded_L:
                            for r in expanded_R:
                                if not adj_matrix[l][r]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        
                        if valid:
                            biclique = (frozenset(expanded_L), frozenset(expanded_R))
                            bicliques.add(biclique)
                            biclique_swapped = (frozenset(expanded_R), frozenset(expanded_L))
                            bicliques.add(biclique_swapped)
        
        return bicliques

    def greedy_set_cover(candidates, edges_to_cover):
        """
        Greedy set cover algorithm.
        candidates: list of (frozenset_L, frozenset_R, frozenset_edges)
        edges_to_cover: set of edges that need to be covered
        Returns: list of selected candidates (indices)
        """
        remaining = set(edges_to_cover)
        selected = []
        
        while remaining:
            best_idx = -1
            best_count = 0
            
            for idx, (L, R, edges) in enumerate(candidates):
                if idx in selected:
                    continue
                # Count how many remaining edges this candidate covers
                count = 0
                for e in edges:
                    if e in remaining:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_idx = idx
            
            if best_idx == -1 or best_count == 0:
                # Fallback: pick a single edge
                e = next(iter(remaining))
                # Find a candidate that covers this edge (should be a single-edge biclique)
                # If none, create one
                found = False
                for idx, (L, R, edges) in enumerate(candidates):
                    if e in edges and idx not in selected:
                        selected.append(idx)
                        remaining.discard(e)
                        found = True
                        break
                if not found:
                    # Create a single-edge biclique
                    a, b = e
                    single = (frozenset({a}), frozenset({b}), frozenset({e}))
                    candidates.append(single)
                    selected.append(len(candidates) - 1)
                    remaining.discard(e)
            else:
                selected.append(best_idx)
                L, R, edges = candidates[best_idx]
                for e in edges:
                    remaining.discard(e)
        
        return selected

    def local_search_merge(bicliques, edge_set):
        """
        Try to merge bicliques to reduce the count.
        bicliques: list of (frozenset_L, frozenset_R)
        Returns: improved list of bicliques
        """
        improved = True
        current = list(bicliques)
        
        while improved:
            improved = False
            n_b = len(current)
            for i in range(n_b):
                for j in range(i + 1, n_b):
                    L1, R1 = current[i]
                    L2, R2 = current[j]
                    
                    # Try merging: L_new = L1 | L2, R_new = R1 | R2
                    L_new = L1 | L2
                    R_new = R1 | R2
                    
                    # Check disjointness
                    if L_new & R_new:
                        continue
                    
                    # Check if union is a valid biclique in the graph
                    valid = True
                    for l in L_new:
                        for r in R_new:
                            if not adj_matrix[l][r]:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Check that the union covers exactly the union of edges
                        # (This is automatically true if the biclique is valid and 
                        # we're only merging, since all cross-edges in the union 
                        # are graph edges)
                        # Merge
                        current[i] = (L_new, R_new)
                        current.pop(j)
                        improved = True
                        break
                if improved:
                    break
                    
        return current

    def branch_and_bound(candidates, edges_to_cover, current_count, best_count):
        """
        Branch and bound for set cover.
        candidates: list of (frozenset_L, frozenset_R, frozenset_edges)
        edges_to_cover: set of edges that need to be covered
        current_count: number of bicliques selected so far
        best_count: current best solution count
        Returns: (best_count, best_selection)
        """
        if not edges_to_cover:
            return current_count, []
        
        if current_count >= best_count:
            return best_count, None
        
        # Lower bound: at least ceil(|edges_to_cover| / max_candidate_size)
        max_size = 0
        for L, R, edges in candidates:
            size = len(edges & edges_to_cover)
            if size > max_size:
                max_size = size
        
        if max_size == 0:
            return best_count, None
        
        lower_bound = current_count + (len(edges_to_cover) + max_size - 1) // max_size
        if lower_bound >= best_count:
            return best_count, None
        
        # Pick the edge with fewest covering candidates
        edge_counts = {}
        for e in edges_to_cover:
            count = 0
            for L, R, edges in candidates:
                if e in edges:
                    count += 1
            edge_counts[e] = count
        
        # Sort edges by count (ascending)
        sorted_edges = sorted(edges_to_cover, key=lambda e: edge_counts.get(e, 0))
        
        best_local_count = best_count
        best_local_selection = None
        
        for e in sorted_edges:
            # Try all candidates that cover this edge
            for idx, (L, R, edges) in enumerate(candidates):
                if e not in edges:
                    continue
                
                new_remaining = edges_to_cover - edges
                new_count = current_count + 1
                
                count, selection = branch_and_bound(
                    candidates, new_remaining, new_count, best_local_count
                )
                
                if count < best_local_count:
                    best_local_count = count
                    best_local_selection = [idx] + (selection if selection else [])
        
        return best_local_count, best_local_selection

    # Main algorithm
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    # Step 1: Generate candidate bicliques
    candidates_set = get_maximal_bicliques()
    
    # Convert to list with edge sets
    candidates = []
    for L, R in candidates_set:
        edges = frozenset()
        for l in L:
            for r in R:
                e = (min(l, r), max(l, r))
                edges = edges | frozenset({e})
        candidates.append((L, R, edges))
    
    # Step 2: Greedy set cover to get initial solution
    greedy_selection = greedy_set_cover(candidates, edge_set)
    greedy_count = len(greedy_selection)
    
    # Step 3: Local search to improve
    greedy_bicliques = [(candidates[i][0], candidates[i][1]) for i in greedy_selection]
    improved_bicliques = local_search_merge(greedy_bicliques, edge_set)
    
    # Step 4: Try branch and bound if time permits
    # Only try if greedy solution is not too large
    if time.time() < deadline - 10.0 and greedy_count <= 20:
        best_count, best_selection = branch_and_bound(
            candidates, edge_set, 0, greedy_count
        )
        
        if best_selection is not None and best_count < len(improved_bicliques):
            improved_bicliques = [(candidates[i][0], candidates[i][1]) for i in best_selection]
    
    # Step 5: Final local search
    final_bicliques = local_search_merge(improved_bicliques, edge_set)
    
    # Convert to required format
    result = []
    for L, R in final_bicliques:
        left = [int(x) for x in L]
        right = [int(x) for x in R]
        result.append([left, right])
    
    return result
