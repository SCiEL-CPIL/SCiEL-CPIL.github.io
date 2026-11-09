"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s

    n = P['n']
    edges = P['edges']

    # Build adjacency
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))

    def time_left():
        return deadline - time.time()

    # Given L and R (disjoint), check if all cross pairs are edges in remaining
    def is_valid_biclique(L, R, remaining):
        if not L or not R:
            return False
        Ls = set(L)
        Rs = set(R)
        if Ls & Rs:
            return False
        for u in L:
            for v in R:
                if (min(u, v), max(u, v)) not in remaining:
                    return False
        return True

    def biclique_edges(L, R):
        es = set()
        for u in L:
            for v in R:
                if u != v:
                    es.add((min(u, v), max(u, v)))
        return es

    # Expand a biclique (L, R) given remaining edges
    def expand_biclique(L, R, remaining):
        L = set(L)
        R = set(R)
        changed = True
        while changed:
            if time_left() < 0.5:
                break
            changed = False
            # Try to add vertices to L
            for w in range(n):
                if w in L or w in R:
                    continue
                # w can join L if it connects to all in R via remaining edges
                ok = True
                for r in R:
                    if (min(w, r), max(w, r)) not in remaining:
                        ok = False
                        break
                if ok:
                    L.add(w)
                    changed = True
            # Try to add vertices to R
            for w in range(n):
                if w in L or w in R:
                    continue
                ok = True
                for l in L:
                    if (min(w, l), max(w, l)) not in remaining:
                        ok = False
                        break
                if ok:
                    R.add(w)
                    changed = True
        return L, R

    # Try to find a large biclique seeded from an edge
    def find_biclique_from_edge(u, v, remaining):
        """Seed biclique with u on left, v on right, then expand."""
        L, R = expand_biclique({u}, {v}, remaining)
        if not L or not R:
            return None
        # Verify
        if not is_valid_biclique(L, R, remaining):
            return None
        return L, R

    # Try swapping: v on left, u on right
    def find_biclique_swapped(u, v, remaining):
        L, R = expand_biclique({v}, {u}, remaining)
        if not L or not R:
            return None
        if not is_valid_biclique(L, R, remaining):
            return None
        return L, R

    # Greedy: repeatedly find best biclique from remaining edges
    def greedy_partition(remaining, adj_rem):
        bicliques = []
        rem = set(remaining)
        
        while rem:
            if time_left() < 0.5:
                break
            # Score each edge as a seed
            best_biclique = None
            best_size = 0
            best_edges = None
            
            # Try a subset of edges as seeds (prioritize high-degree)
            candidates = list(rem)
            # Sort by degree in remaining graph
            candidates.sort(key=lambda e: (-(len(adj_rem[e[0]]) + len(adj_rem[e[1]])), e[0], e[1]))
            
            # Try top candidates
            tried = 0
            for (u, v) in candidates:
                if time_left() < 0.3:
                    break
                tried += 1
                if tried > 80:  # limit seeds tried per iteration
                    break
                    
                # Try u left, v right
                L1, R1 = find_biclique_from_edge(u, v, rem)
                if L1 and R1:
                    sz = len(L1) * len(R1)
                    if sz > best_size:
                        best_size = sz
                        best_biclique = (L1, R1)
                        best_edges = biclique_edges(L1, R1)
                
                # Try v left, u right
                L2, R2 = find_biclique_swapped(u, v, rem)
                if L2 and R2:
                    sz = len(L2) * len(R2)
                    if sz > best_size:
                        best_size = sz
                        best_biclique = (L2, R2)
                        best_edges = biclique_edges(L2, R2)
            
            if best_biclique is None:
                # Fallback: single edge
                e = next(iter(rem))
                best_biclique = ({e[0]}, {e[1]})
                best_edges = {e}
            
            L, R = best_biclique
            bicliques.append((L, R))
            
            # Remove covered edges
            for e in best_edges:
                rem.discard(e)
                u, v = e
                adj_rem[u].discard(v)
                adj_rem[v].discard(u)
        
        # Handle remaining edges as single-edge bicliques
        for e in rem:
            bicliques.append(({e[0]}, {e[1]}))
        
        return bicliques

    # Local improvement: try to merge or split bicliques
    def try_improve(bicliques, edge_set):
        """Try to reduce the number of bicliques."""
        if time_left() < 1.0:
            return bicliques
        
        improved = True
        while improved:
            if time_left() < 1.0:
                break
            improved = False
            new_bicliques = list(bicliques)
            
            # Try merging pairs
            i = 0
            while i < len(new_bicliques):
                if time_left() < 0.5:
                    break
                L1, R1 = new_bicliques[i]
                j = i + 1
                while j < len(new_bicliques):
                    if time_left() < 0.3:
                        break
                    L2, R2 = new_bicliques[j]
                    
                    # Try merge: (L1∪L2, R1∪R2)
                    Lnew = set(L1) | set(L2)
                    Rnew = set(R1) | set(R2)
                    if Lnew and Rnew and not (Lnew & Rnew):
                        # Check all cross edges are in edge_set
                        ok = True
                        for u in Lnew:
                            for v in Rnew:
                                if (min(u, v), max(u, v)) not in edge_set:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            # Check edge partition: edges_i ∪ edges_j == edges_new
                            ei = biclique_edges(L1, R1)
                            ej = biclique_edges(L2, R2)
                            en = biclique_edges(Lnew, Rnew)
                            if ei | ej == en and not (ei & ej):
                                new_bicliques.pop(j)
                                new_bicliques[i] = (Lnew, Rnew)
                                improved = True
                                j = i + 1
                                continue
                    j += 1
                i += 1
            
            bicliques = new_bicliques
        
        return bicliques

    # Multiple restarts with different orderings
    best_result = None
    best_count = float('inf')
    
    # Build initial adjacency for remaining
    adj_rem = [set() for _ in range(n)]
    for u, v in edge_set:
        adj_rem[u].add(v)
        adj_rem[v].add(u)
    
    # Run greedy
    bicliques = greedy_partition(edge_set, [set(s) for s in adj_rem])
    
    # Improve
    bicliques = try_improve(bicliques, edge_set)
    
    # Convert to list format
    result = []
    for L, R in bicliques:
        L_list = sorted([int(x) for x in L])
        R_list = sorted([int(x) for x in R])
        result.append([L_list, R_list])
    
    # Verify
    covered = set()
    valid = True
    for L, R in result:
        for u in L:
            for v in R:
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in covered:
                    valid = False
                    break
                if key not in edge_set:
                    valid = False
                    break
                covered.add(key)
            if not valid:
                break
    
    if not valid or covered != edge_set:
        # Fallback: star decomposition
        result = []
        for u in range(n):
            neighbors = [v for v in adj[u]]
            if neighbors:
                result.append([[int(u)], [int(v) for v in sorted(neighbors)]])
    
    return result
# EVOLVE-BLOCK-END
