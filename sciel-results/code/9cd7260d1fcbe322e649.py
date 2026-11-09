import time
import random
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets for the full graph
    adj_full = [set() for _ in range(n)]
    for u, v in edges:
        adj_full[u].add(v)
        adj_full[v].add(u)
        
    # Normalize edges
    norm_edges = set()
    for u, v in edges:
        if u < v:
            norm_edges.add((u, v))
        else:
            norm_edges.add((v, u))
            
    all_edges = list(norm_edges)
    m = len(all_edges)
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    best_bicliques = None
    best_count = float('inf')
    
    def expand_biclique(seed_u, seed_v, uncovered_set, n):
        """
        Expand the seed edge (seed_u, seed_v) into a maximal biclique
        using only edges present in uncovered_set.
        """
        L = {seed_u}
        R = {seed_v}
        
        changed = True
        while changed:
            changed = False
            
            candidates_L = []
            for w in range(n):
                if w in L or w in R:
                    continue
                can_add = True
                for r in R:
                    edge = (min(w, r), max(w, r))
                    if edge not in uncovered_set:
                        can_add = False
                        break
                if can_add:
                    candidates_L.append(w)
            
            if candidates_L:
                L.update(candidates_L)
                changed = True
                continue
            
            candidates_R = []
            for w in range(n):
                if w in L or w in R:
                    continue
                can_add = True
                for l in L:
                    edge = (min(l, w), max(l, w))
                    if edge not in uncovered_set:
                        can_add = False
                        break
                if can_add:
                    candidates_R.append(w)
            
            if candidates_R:
                R.update(candidates_R)
                changed = True
        
        covered_edges = set()
        for l in L:
            for r in R:
                covered_edges.add((min(l, r), max(l, r)))
                
        return L, R, covered_edges

    def greedy_set_cover(uncovered_edges_list, n):
        """
        Greedy set cover: repeatedly pick the biclique covering the most uncovered edges.
        """
        uncovered = set(uncovered_edges_list)
        bicliques = []
        
        while uncovered:
            if time.time() > deadline:
                return None
            
            best_coverage_count = -1
            best_L = None
            best_R = None
            best_covered = None
            
            if len(uncovered) <= 40:
                candidates_seeds = list(uncovered)
            else:
                sample_size = min(len(uncovered), 50)
                candidates_seeds = random.sample(list(uncovered), sample_size)
            
            for seed_edge in candidates_seeds:
                u, v = seed_edge
                L, R, covered = expand_biclique(u, v, uncovered, n)
                
                if not L or not R:
                    continue
                
                count = len(covered)
                if count > best_coverage_count:
                    best_coverage_count = count
                    best_L = L
                    best_R = R
                    best_covered = covered
            
            if best_L is None:
                u, v = next(iter(uncovered))
                best_L = {u}
                best_R = {v}
                best_covered = {(u, v)}
            
            bicliques.append((list(best_L), list(best_R)))
            uncovered -= best_covered
            
        return bicliques

    def try_merge(bicliques, all_edges_set, n):
        """Try to merge pairs of bicliques if their union forms a valid biclique."""
        bicliques = [list(b) for b in bicliques]
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(bicliques)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(bicliques)):
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    merged_L = sorted(set(L1) | set(L2))
                    merged_R = sorted(set(R1) | set(R2))
                    
                    if not merged_L or not merged_R:
                        continue
                    
                    valid = True
                    for l in merged_L:
                        for r in merged_R:
                            edge = (min(l, r), max(l, r))
                            if edge not in all_edges_set:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        bicliques[i] = (merged_L, merged_R)
                        bicliques.pop(j)
                        improved = True
                        break
                if improved:
                    break
        return bicliques

    # Run multiple restarts with different random seeds
    rng = random.Random(42)
    for restart in range(300):
        if time.time() > deadline:
            break
        
        rng.seed(restart)
        shuffled = all_edges[:]
        rng.shuffle(shuffled)
        
        result = greedy_set_cover(shuffled, n)
        
        if result is None:
            continue
        
        result = try_merge(result, norm_edges, n)
        
        if len(result) < best_count:
            best_count = len(result)
            best_bicliques = result
            
        if best_count == 1:
            break

    # Fallback if no valid partition found (should not happen)
    if best_bicliques is None:
        best_bicliques = []
        uncovered = set(all_edges)
        while uncovered:
            u, v = next(iter(uncovered))
            best_bicliques.append([[u], [v]])
            uncovered.remove((u, v))

    # Convert to required format: list of [left, right]
    final_result = []
    for L, R in best_bicliques:
        final_result.append([list(L), list(R)])
        
    return final_result
