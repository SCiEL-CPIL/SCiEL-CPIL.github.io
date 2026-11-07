import time
import random
import itertools

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = P['n']
    edges = P['edges']
    edge_set = set()
    for e in edges:
        u, v = e
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    
    # Vertices with edges
    active_vertices = set()
    for u, v in edge_set:
        active_vertices.add(u)
        active_vertices.add(v)
    
    # Seed: one biclique per active vertex, covering all its incident edges
    # Biclique: [vertex, neighbors]
    best_bicliques = []
    for u in sorted(active_vertices):
        best_bicliques.append([list([u]), list(adj[u])])
    
    best_count = len(best_bicliques)
    
    # Helper: check if a set of edges can be covered by a biclique [L, R]
    # where all edges in the set are in L x R
    def edges_covered_by_biclique(edge_list, L, R):
        L_set = set(L)
        R_set = set(R)
        for u, v in edge_list:
            if u not in L_set or v not in R_set:
                if u not in R_set or v not in L_set:
                    return False
        return True
    
    # Helper: get the set of edges covered by a biclique
    def biclique_edges(L, R):
        edges = set()
        for u in L:
            for v in R:
                if (u, v) in edge_set or (v, u) in edge_set:
                    if u < v:
                        edges.add((u, v))
                    else:
                        edges.add((v, u))
        return edges
    
    # Check if a biclique is valid: all cross pairs must be host edges
    def is_valid_biclique(L, R):
        L_set = set(L)
        R_set = set(R)
        for u in L:
            for v in R:
                if u == v:
                    continue
                if (u, v) not in edge_set and (v, u) not in edge_set:
                    return False
        return True
    
    # Check if the collection of bicliques partitions all edges exactly once
    def is_valid_partition(bicliques):
        covered = set()
        for L, R in bicliques:
            if not L or not R:
                return False
            if not is_valid_biclique(L, R):
                return False
            for u in L:
                for v in R:
                    if u == v:
                        continue
                    if (u, v) in edge_set or (v, u) in edge_set:
                        if u < v:
                            e = (u, v)
                        else:
                            e = (v, u)
                        if e in covered:
                            return False
                        covered.add(e)
                    else:
                        return False
        return covered == edge_set
    
    # Try to merge two bicliques
    def try_merge(bi, bj, bicliques):
        """Try to merge biclique bi and bj. Returns new bicliques if successful."""
        L1, R1 = bicliques[bi]
        L2, R2 = bicliques[bj]
        
        # Option 1: merge into [L1 ∪ L2, R1 ∩ R2]
        R_intersect = list(set(R1) & set(R2))
        if R_intersect:
            L_union = list(set(L1) | set(L2))
            new_b = [L_union, R_intersect]
            if is_valid_biclique(L_union, R_intersect):
                # Check if this covers all edges from both bicliques
                e1 = biclique_edges(L1, R1)
                e2 = biclique_edges(L2, R2)
                e_new = biclique_edges(L_union, R_intersect)
                if e1 | e2 == e_new:
                    new_bicliques = [b for i, b in enumerate(bicliques) if i != bi and i != bj]
                    new_bicliques.append(new_b)
                    if is_valid_partition(new_bicliques):
                        return new_bicliques
        
        # Option 2: merge into [L1 ∩ L2, R1 ∪ R2]
        L_intersect = list(set(L1) & set(L2))
        if L_intersect:
            R_union = list(set(R1) | set(R2))
            new_b = [L_intersect, R_union]
            if is_valid_biclique(L_intersect, R_union):
                e1 = biclique_edges(L1, R1)
                e2 = biclique_edges(L2, R2)
                e_new = biclique_edges(L_intersect, R_union)
                if e1 | e2 == e_new:
                    new_bicliques = [b for i, b in enumerate(bicliques) if i != bi and i != bj]
                    new_bicliques.append(new_b)
                    if is_valid_partition(new_bicliques):
                        return new_bicliques
        
        # Option 3: expand one biclique to absorb the other
        # Try to expand biclique 1 to cover biclique 2's edges
        # New biclique 1: [L1', R1'] such that it covers e1 ∪ e2
        # This is more complex, so we'll try a simpler approach:
        # Can we find L', R' such that biclique(L', R') covers e1 ∪ e2 and is valid?
        
        e_combined = biclique_edges(L1, R1) | biclique_edges(L2, R2)
        
        # Try all pairs of subsets
        # This is expensive, so limit the search
        all_vertices_in_combined = set()
        for u, v in e_combined:
            all_vertices_in_combined.add(u)
            all_vertices_in_combined.add(v)
        
        # Try small bicliques
        verts_list = list(all_vertices_in_combined)
        if len(verts_list) <= 12:
            for size in range(1, min(len(verts_list) + 1, 7)):
                for L_subset in itertools.combinations(verts_list, size):
                    L_set = set(L_subset)
                    # R must be vertices that are connected to all in L via e_combined
                    R_candidates = set()
                    for u, v in e_combined:
                        if u in L_set and v not in L_set:
                            R_candidates.add(v)
                        elif v in L_set and u not in L_set:
                            R_candidates.add(u)
                    if not R_candidates:
                        continue
                    R_list = list(R_candidates)
                    # Check if this biclique covers e_combined
                    e_new = biclique_edges(L_set, R_set := set(R_list))
                    if e_new == e_combined and is_valid_biclique(list(L_set), R_list):
                        new_bicliques = [b for i, b in enumerate(bicliques) if i != bi and i != bj]
                        new_bicliques.append([list(L_set), R_list])
                        if is_valid_partition(new_bicliques):
                            return new_bicliques
                        break
        
        return None
    
    # Greedy merge search
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        bicliques = best_bicliques
        for i in range(len(bicliques)):
            if time.time() > deadline - 5.0:
                break
            for j in range(i + 1, len(bicliques)):
                result = try_merge(i, j, bicliques)
                if result is not None:
                    best_bicliques = result
                    best_count = len(result)
                    improved = True
                    break
            if improved:
                break
    
    # Try local search: remove a vertex from one side and add to the other
    if time.time() < deadline - 10.0:
        for _ in range(50):
            if time.time() > deadline - 10.0:
                break
            bicliques = best_bicliques
            found = False
            for i in range(len(bicliques)):
                if found:
                    break
                L, R = bicliques[i]
                # Try moving a vertex from L to R or R to L
                for v in list(L):
                    new_L = [x for x in L if x != v]
                    new_R = list(set(R) | {v})
                    if not new_L or not new_R:
                        continue
                    if is_valid_biclique(new_L, new_R):
                        e_old = biclique_edges(L, R)
                        e_new = biclique_edges(new_L, new_R)
                        # Check if e_new still covers all edges that were covered
                        # and the overall partition is still valid
                        new_bicliques = [bicliques[k] if k != i else [new_L, new_R] for k in range(len(bicliques))]
                        if is_valid_partition(new_bicliques):
                            best_bicliques = new_bicliques
                            found = True
                            break
                if found:
                    break
            if not found:
                break
    
    # Ensure types are correct
    result = []
    for b in best_bicliques:
        L, R = b
        L_int = [int(x) for x in L]
        R_int = [int(x) for x in R]
        result.append([L_int, R_int])
    
    return result
