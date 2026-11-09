import time
import random
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency set for O(1) edge check
    adj = set()
    for u, v in edges:
        adj.add((min(u, v), max(u, v)))
        adj.add((max(u, v), min(u, v)))
        
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
    deadline = start_time + 55
    
    def edge_exists(u, v, uncovered):
        """Check if edge (u,v) exists in uncovered set"""
        if u < v:
            return (u, v) in uncovered
        else:
            return (v, u) in uncovered
    
    def expand_biclique(seed_u, seed_v, uncovered):
        """
        Expand the seed edge (seed_u, seed_v) into a maximal biclique
        using only edges present in uncovered.
        """
        L = {seed_u}
        R = {seed_v}
        
        changed = True
        while changed:
            changed = False
            
            # Try to add vertices to L
            candidates_L = []
            for w in range(n):
                if w in L or w in R:
                    continue
                # Check if w is connected to all r in R via edges in uncovered
                can_add = True
                for r in R:
                    if not edge_exists(w, r, uncovered):
                        can_add = False
                        break
                if can_add:
                    candidates_L.append(w)
            
            if candidates_L:
                L.update(candidates_L)
                changed = True
                continue
            
            # Try to add vertices to R
            candidates_R = []
            for w in range(n):
                if w in L or w in R:
                    continue
                can_add = True
                for l in L:
                    if not edge_exists(l, w, uncovered):
                        can_add = False
                        break
                if can_add:
                    candidates_R.append(w)
            
            if candidates_R:
                R.update(candidates_R)
                changed = True
                continue
        
        # The resulting biclique covers all edges between L and R
        covered_edges = set()
        for l in L:
            for r in R:
                if l < r:
                    covered_edges.add((l, r))
                else:
                    covered_edges.add((r, l))
                
        return L, R, covered_edges

    def greedy_set_cover(uncovered_edges, strategy='random'):
        """
        Greedy set cover: repeatedly pick a biclique covering the most uncovered edges.
        """
        uncovered = set(uncovered_edges)
        bicliques = []
        
        while uncovered:
            if time.time() > deadline:
                return None
            
            best_L = None
            best_R = None
            best_covered = None
            best_count = -1
            
            # Strategy: sample multiple seed edges and pick the best expansion
            sample_size = min(len(uncovered), 50)
            if sample_size < 10:
                candidates_seeds = list(uncovered)
            else:
                candidates_seeds = random.sample(list(uncovered), sample_size)
            
            for seed_edge in candidates_seeds:
                u, v = seed_edge
                L, R, covered = expand_biclique(u, v, uncovered)
                
                if not L or not R:
                    continue
                
                count = len(covered)
                if count > best_count:
                    best_count = count
                    best_L = L
                    best_R = R
                    best_covered = covered
            
            if best_L is None:
                # Fallback
                u, v = next(iter(uncovered))
                best_L = {u}
                best_R = {v}
                best_covered = {(u, v)}
            
            bicliques.append((list(best_L), list(best_R)))
            uncovered -= best_covered
            
        return bicliques

    def get_edges_from_bicliques(bicliques):
        edges = set()
        for L, R in bicliques:
            for u in L:
                for v in R:
                    if u < v:
                        edges.add((u, v))
                    else:
                        edges.add((v, u))
        return edges

    def can_merge(b1, b2):
        """
        Check if two bicliques can be merged into one.
        b1 = (L1, R1), b2 = (L2, R2)
        Possible merges:
        1. L1∪L2, R1∪R2 - all cross edges must exist
        2. L1∪R2, R1∪L2 - all cross edges must exist (if we swap sides)
        """
        L1, R1 = b1
        L2, R2 = b2
        
        # Check if L1 and L2 are disjoint, R1 and R2 are disjoint
        if L1 & L2 or R1 & R2:
            return None
        
        # Try merge 1: L = L1∪L2, R = R1∪R2
        L = L1 | L2
        R = R1 | R2
        valid = True
        for u in L:
            for v in R:
                if u == v:
                    continue
                if not edge_exists(u, v, adj):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return (list(L), list(R))
        
        # Try merge 2: L = L1∪R2, R = R1∪L2
        L = L1 | R2
        R = R1 | L2
        valid = True
        for u in L:
            for v in R:
                if u == v:
                    continue
                if not edge_exists(u, v, adj):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return (list(L), list(R))
        
        return None

    def local_search(bicliques):
        """
        Try to improve the partition by merging pairs of bicliques.
        """
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
                
            n_bicliques = len(bicliques)
            for i in range(n_bicliques):
                for j in range(i + 1, n_bicliques):
                    if time.time() > deadline:
                        return bicliques
                        
                    # Try to merge bicliques[i] and bicliques[j]
                    merged = can_merge(bicliques[i], bicliques[j])
                    
                    if merged is not None:
                        # Remove i and j, add merged
                        bicliques.pop(j)
                        bicliques.pop(i)
                        bicliques.append(merged)
                        improved = True
                        break # Restart the outer loop as indices changed
                if improved:
                    break
                    
        return bicliques

    def try_split_and_recover(bicliques):
        """
        Try to split a biclique and see if we can cover its edges with fewer bicliques.
        This is a heuristic that may help in some cases.
        """
        if time.time() > deadline:
            return bicliques
            
        best_result = bicliques
        best_count = len(bicliques)
        
        # Try each biclique
        for idx in range(len(bicliques)):
            if time.time() > deadline:
                break
                
            L, R = bicliques[idx]
            edges_in_biclique = get_edges_from_bicliques([bicliques[idx]])
            
            if len(edges_in_biclique) <= 2:
                continue
            
            # Try to cover these edges with greedy
            recovered = greedy_set_cover(edges_in_biclique, strategy='random')
            
            if recovered is not None and len(recovered) < len([bicliques[idx]]):
                # This shouldn't happen since we're trying to cover with fewer, but check
                pass
            elif recovered is not None and len(recovered) == 1:
                # Same number, no improvement
                pass
        
        return best_result

    best_bicliques = None
    best_count = float('inf')
    
    # Run multiple restarts with different strategies
    for restart in range(300):
        if time.time() > deadline:
            break
        
        random.seed(restart)
        
        # Initial Greedy
        result = greedy_set_cover(all_edges, strategy='random')
        
        if result is None:
            continue
            
        # Local Search
        result = local_search(result)
        
        if len(result) < best_count:
            best_count = len(result)
            best_bicliques = result
            
        if best_count <= 2:
            break

    # Fallback
    if best_bicliques is None:
        best_bicliques = []
        uncovered = set(all_edges)
        while uncovered:
            u, v = next(iter(uncovered))
            best_bicliques.append([[u], [v]])
            uncovered.remove((u, v))

    # Convert to required format
    final_result = []
    for L, R in best_bicliques:
        # Ensure types are list of int
        final_result.append([list(L), list(R)])
        
    return final_result
