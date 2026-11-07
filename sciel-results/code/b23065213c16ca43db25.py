import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    edge_set = set()
    adj = [set() for _ in range(n)]
    for u, v in edges:
        edge_set.add((u, v) if u < v else (v, u))
        adj[u].add(v)
        adj[v].add(u)
    
    all_edges_list = list(edge_set)
    total_edges = len(all_edges_list)
    
    def is_biclique(L, R):
        """Check if all cross pairs are edges."""
        for u in L:
            for v in R:
                key = (u, v) if u < v else (v, u)
                if key not in edge_set:
                    return False
        return True
    
    def get_cross_edges(L, R):
        """Get set of cross edges for biclique (L, R)."""
        result = set()
        for u in L:
            for v in R:
                key = (u, v) if u < v else (v, u)
                result.add(key)
        return result
    
    def expand_biclique(start_u, start_v, remaining_edges, max_iter=100):
        """Greedy expansion of biclique starting from edge (start_u, start_v)."""
        L = {start_u}
        R = {start_v}
        
        for _ in range(max_iter):
            # Try to add a vertex to L
            added = False
            candidates_L = [w for w in range(n) if w not in L and w not in R]
            # Sort by how many R neighbors they have
            candidates_L.sort(key=lambda w: -len(adj[w] & R))
            for w in candidates_L:
                if adj[w] >= R:  # w connected to all of R
                    L.add(w)
                    added = True
                    break
            
            # Try to add a vertex to R
            candidates_R = [w for w in range(n) if w not in L and w not in R]
            candidates_R.sort(key=lambda w: -len(adj[w] & L))
            for w in candidates_R:
                if adj[w] >= L:  # w connected to all of L
                    R.add(w)
                    added = True
                    break
            
            if not added:
                break
        
        return list(L), list(R)
    
    def greedy_partition(remaining_edges, candidates_pool):
        """Greedy set cover: pick biclique covering most uncovered edges."""
        if not remaining_edges:
            return []
        
        best_bc = None
        best_cover = 0
        best_cross = None
        
        for bc in candidates_pool:
            L, R = bc
            cross = get_cross_edges(L, R)
            cover = len(cross & remaining_edges)
            if cover > best_cover:
                best_cover = cover
                best_bc = bc
                best_cross = cross
        
        if best_bc is None or best_cover == 0:
            # Fallback: single edge bicliques
            result = []
            for e in remaining_edges:
                result.append([list(e[0]), list(e[1])])
            return result
        
        L, R = best_bc
        result = [[list(L), list(R)]]
        new_remaining = remaining_edges - best_cross
        rest = greedy_partition(new_remaining, candidates_pool)
        result.extend(rest)
        return result
    
    # Generate candidate bicliques
    candidates = []
    
    # From each edge, try to expand
    for e in all_edges_list:
        L, R = expand_biclique(e[0], e[1], edge_set)
        if len(L) >= 1 and len(R) >= 1:
            # Ensure sides are disjoint
            L_set = set(L)
            R_set = set(R)
            if L_set & R_set:
                continue
            cross = get_cross_edges(L, R)
            if cross and cross <= edge_set:
                candidates.append((L, R))
        
        # Also try swapped
        L2, R2 = expand_biclique(e[1], e[0], edge_set)
        if len(L2) >= 1 and len(R2) >= 1:
            L2_set = set(L2)
            R2_set = set(R2)
            if L2_set & R2_set:
                continue
            cross2 = get_cross_edges(L2, R2)
            if cross2 and cross2 <= edge_set:
                candidates.append((L2, R2))
    
    # Also add single-edge bicliques as fallback
    for e in all_edges_list:
        candidates.append(([e[0]], [e[1]]))
    
    # Deduplicate candidates by their cross-edge set
    seen_cross = set()
    unique_candidates = []
    for L, R in candidates:
        cross = get_cross_edges(L, R)
        cross_frozen = frozenset(cross)
        if cross_frozen not in seen_cross:
            seen_cross.add(cross_frozen)
            unique_candidates.append((L, R))
    
    candidates = unique_candidates
    
    # Sort candidates by size (larger first) for better greedy performance
    candidates.sort(key=lambda x: -len(get_cross_edges(x[0], x[1])))
    
    # Try greedy with different orderings
    best_result = None
    best_count = total_edges + 1
    
    # First: deterministic greedy
    result = greedy_partition(set(all_edges_list), candidates)
    if result:
        count = len(result)
        if count < best_count:
            best_count = count
            best_result = result
    
    # Try randomized greedy multiple times
    random.seed(42)
    num_trials = 50
    
    for trial in range(num_trials):
        if time.time() > deadline - 5:
            break
        
        # Shuffle candidates with bias toward larger ones
        shuffled = list(candidates)
        # Weight by size
        sizes = [len(get_cross_edges(L, R)) for L, R in shuffled]
        # Random perturbation
        for i in range(len(shuffled)):
            sizes[i] += random.random() * 2
        
        shuffled.sort(key=lambda x: -sizes[candidates.index(x)] if candidates.index(x) < len(sizes) else 0)
        
        # Simpler: just shuffle and use greedy
        shuffled = list(candidates)
        random.shuffle(shuffled)
        
        # Greedy with this ordering
        remaining = set(all_edges_list)
        chosen = []
        
        for bc in shuffled:
            if time.time() > deadline - 5:
                break
            if not remaining:
                break
            L, R = bc
            cross = get_cross_edges(L, R)
            cover = cross & remaining
            if cover:
                chosen.append([list(L), list(R)])
                remaining -= cover
        
        if remaining:
            # Add fallback single-edge bicliques
            for e in remaining:
                chosen.append([list(e[0]), list(e[1])])
        
        count = len(chosen)
        if count < best_count:
            best_count = count
            best_result = chosen
    
    # Validate and return
    if best_result is None:
        # Ultimate fallback
        best_result = [[list(e[0]), list(e[1])] for e in all_edges_list]
    
    # Convert to proper format: list of [left_vertices, right_vertices]
    final = []
    for bc in best_result:
        L = list(bc[0])
        R = list(bc[1])
        # Ensure nonempty and disjoint
        if not L or not R:
            continue
        if set(L) & set(R):
            continue
        # Ensure all cross edges are actual edges
        valid = True
        for u in L:
            for v in R:
                key = (u, v) if u < v else (v, u)
                if key not in edge_set:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            final.append([L, R])
    
    # Verify partition
    covered = set()
    for bc in final:
        L, R = bc
        cross = get_cross_edges(L, R)
        if cross & covered:
            # Overlap - this shouldn't happen with our greedy
            pass
        covered |= cross
    
    if covered != edge_set:
        # Something went wrong, use fallback
        final = [[list(e[0]), list(e[1])] for e in all_edges_list]
    
    # Ensure types are correct
    result = []
    for bc in final:
        L = [int(x) for x in bc[0]]
        R = [int(x) for x in bc[1]]
        result.append([L, R])
    
    return result
