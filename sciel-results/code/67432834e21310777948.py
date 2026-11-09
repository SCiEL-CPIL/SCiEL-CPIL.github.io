"""Biclique partitioning of a random graph using greedy set-cover with local search."""
import time
import random
import itertools

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    n = P['n']
    edges = P['edges']
    
    # Build adjacency structure
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Total edges to cover
    all_edges = set()
    for u, v in edges:
        if u < v:
            all_edges.add((u, v))
        else:
            all_edges.add((v, u))
    
    total_edges = len(all_edges)
    
    def find_max_biclique(remaining_edges):
        """Find a large biclique from remaining edges using edge-pair seeding."""
        if not remaining_edges:
            return None, None
        
        # Build adjacency from remaining edges for fast lookup
        rem_adj = [set() for _ in range(n)]
        for u, v in remaining_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        # Degree in remaining graph
        deg = [len(rem_adj[i]) for i in range(n)]
        
        # Sort vertices by degree descending
        verts_by_deg = sorted(range(n), key=lambda x: deg[x], reverse=True)
        
        best_L = None
        best_R = None
        best_size = 0
        
        # Try seeding from top-degree vertices
        for start in verts_by_deg[:min(10, n)]:
            if deg[start] == 0:
                break
            
            # Get neighbors in remaining graph
            neighbors = rem_adj[start]
            
            # Try both orientations
            for orientation in [0, 1]:
                if orientation == 0:
                    L = {start}
                    R = set(neighbors)
                else:
                    L = set(neighbors)
                    R = {start}
                
                # Iteratively refine using intersection of neighbor sets
                changed = True
                while changed:
                    changed = False
                    
                    # Refine R: keep only vertices adjacent to all in L
                    if L:
                        # Intersection of neighbors of all vertices in L
                        common = set.intersection(*(rem_adj[u] for u in L))
                        new_R = R & common
                        if new_R != R:
                            R = new_R
                            changed = True
                    
                    # Refine L: keep only vertices adjacent to all in R
                    if R:
                        common = set.intersection(*(rem_adj[v] for v in R))
                        new_L = L & common
                        if new_L != L:
                            L = new_L
                            changed = True
                
                if L and R:
                    size = len(L) * len(R)
                    if size > best_size:
                        best_size = size
                        best_L = L
                        best_R = R
        
        # If no biclique found, fallback to single edge
        if best_L is None:
            u, v = next(iter(remaining_edges))
            return {u}, {v}
        
        return best_L, best_R
    
    def greedy_partition():
        """Greedy set-cover style partition."""
        remaining = set(all_edges)
        bicliques = []
        
        while remaining:
            if time.time() > deadline:
                break
            
            L, R = find_max_biclique(remaining)
            if L is None or not L or not R:
                # Fallback: take one edge
                u, v = next(iter(remaining))
                L = {u}
                R = {v}
            
            # Remove all edges between L and R from remaining
            for u in L:
                for v in R:
                    e = (min(u, v), max(u, v))
                    remaining.discard(e)
            
            bicliques.append((list(L), list(R)))
        
        return bicliques, remaining
    
    def try_local_improvement(bicliques):
        """Try to merge bicliques or find better partitions."""
        if len(bicliques) <= 1:
            return bicliques
        
        # Compute all edges covered by each biclique
        biclique_edges = []
        for L, R in bicliques:
            edges_set = set()
            for u in L:
                for v in R:
                    e = (min(u, v), max(u, v))
                    edges_set.add(e)
            biclique_edges.append(edges_set)
        
        # Try merging pairs of bicliques
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            new_bicliques = list(bicliques)
            new_edges = list(biclique_edges)
            
            for i in range(len(new_bicliques)):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, len(new_bicliques)):
                    if time.time() > deadline - 10:
                        break
                    
                    # Check if we can merge bicliques i and j
                    # A merge is valid if the union forms a valid biclique
                    # This is expensive, so only try if the bicliques are small
                    if len(new_bicliques[i][0]) * len(new_bicliques[i][1]) > 20:
                        continue
                    if len(new_bicliques[j][0]) * len(new_bicliques[j][1]) > 20:
                        continue
                    
                    L_i, R_i = new_bicliques[i]
                    L_j, R_j = new_bicliques[j]
                    
                    # Try different merge strategies
                    # Strategy 1: L = L_i ∪ L_j, R = R_i ∩ R_j (if valid)
                    # Strategy 2: L = L_i ∩ L_j, R = R_i ∪ R_j (if valid)
                    # Strategy 3: L = L_i ∪ R_j, R = R_i ∩ L_j (if valid)
                    # etc.
                    
                    # For simplicity, try: can we form a biclique from all edges in both?
                    all_edges_merge = new_edges[i] | new_edges[j]
                    
                    # Try to find a biclique covering all these edges
                    # Get all vertices involved
                    all_verts = set()
                    for u, v in all_edges_merge:
                        all_verts.add(u)
                        all_verts.add(v)
                    
                    if len(all_verts) > 15:
                        continue  # Too large to check quickly
                    
                    # Try all bipartitions of all_verts
                    # This is exponential, so limit to small cases
                    if len(all_verts) > 8:
                        continue
                    
                    verts_list = sorted(all_verts)
                    found_merge = False
                    
                    # Try each vertex as potential left side seed
                    for v in verts_list:
                        # Try L containing v, R not containing v
                        # Use the same refinement approach
                        L = {v}
                        R = set(all_verts) - {v}
                        
                        changed = True
                        while changed:
                            changed = False
                            # Refine R
                            common = set()
                            for u in L:
                                common = set(common | (set(u for u2 in R if (min(u, u2), max(u, u2)) in all_edges_merge)))
                            new_R = R & common
                            if new_R != R:
                                R = new_R
                                changed = True
                            # Refine L
                            common = set()
                            for v2 in R:
                                common = set(common | (set(v2 for u2 in L if (min(u2, v2), max(u2, v2)) in all_edges_merge)))
                            new_L = L & common
                            if new_L != L:
                                L = new_L
                                changed = True
                        
                        if L and R:
                            # Check if L×R covers all edges in the merge
                            covered = set()
                            for u in L:
                                for v2 in R:
                                    covered.add((min(u, v2), max(u, v2)))
                            if covered == all_edges_merge:
                                # Valid merge!
                                new_bicliques = [x for k, x in enumerate(new_bicliques) if k not in (i, j)]
                                new_bicliques.append((list(L), list(R)))
                                new_edges = [x for k, x in enumerate(new_edges) if k not in (i, j)]
                                new_edges.append(covered)
                                improved = True
                                found_merge = True
                                break
                        if found_merge:
                            break
                    if found_merge:
                        break
                
                if improved:
                    break
            
            if improved:
                bicliques = new_bicliques
                biclique_edges = new_edges
        
        return bicliques
    
    # Run greedy with multiple random restarts
    best_bicliques = None
    best_count = float('inf')
    
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Shuffle the edge order to get different greedy outcomes
        shuffled_edges = list(all_edges)
        random.shuffle(shuffled_edges)
        
        # Temporarily replace all_edges with shuffled version
        saved_all_edges = all_edges
        all_edges = set(shuffled_edges)
        
        bicliques, remaining = greedy_partition()
        
        # Restore all_edges
        all_edges = saved_all_edges
        
        # If there are remaining edges (due to deadline), add them as single-edge bicliques
        if remaining:
            for u, v in remaining:
                bicliques.append(([u], [v]))
        
        count = len(bicliques)
        if count < best_count:
            best_count = count
            best_bicliques = bicliques
        
        # Try local improvement if time permits
        if time.time() < deadline - 10 and len(bicliques) > 1:
            improved = try_local_improvement(bicliques)
            if len(improved) < best_count:
                best_count = len(improved)
                best_bicliques = improved
    
    if best_bicliques is None:
        # Fallback: one biclique per edge
        best_bicliques = []
        for u, v in all_edges:
            best_bicliques.append(([u], [v]))
    
    bicliques = best_bicliques
    
    # Convert to required format: list of [left_vertices, right_vertices]
    result = []
    for L, R in bicliques:
        result.append([list(L), list(R)])
    
    return result
