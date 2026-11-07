import time
import random
from collections import deque

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

    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for a, b in edge_set:
        adj[a].add(b)
        adj[b].add(a)

    def expand_biclique(u, v, remaining, vertex_order):
        """
        Expand seed edge (u,v) into a maximal biclique within 'remaining'.
        Returns (L, R) as sets, or None if invalid.
        """
        L = {u}
        R = {v}
        
        # We need L x R to be fully covered by 'remaining'
        # Start is valid since (u,v) is in remaining (assumed)
        
        changed = True
        while changed:
            changed = False
            
            # Try to add to L
            for x in vertex_order:
                if x in L or x in R:
                    continue
                # x must connect to all in R via edges in remaining
                can_add = True
                for y in R:
                    e = (min(x, y), max(x, y))
                    if e not in remaining:
                        can_add = False
                        break
                if can_add:
                    L.add(x)
                    changed = True
            
            # Try to add to R
            for y in vertex_order:
                if y in L or y in R:
                    continue
                can_add = True
                for x in L:
                    e = (min(x, y), max(x, y))
                    if e not in remaining:
                        can_add = False
                        break
                if can_add:
                    R.add(y)
                    changed = True
        
        # Validate
        for l in L:
            for r in R:
                e = (min(l, r), max(l, r))
                if e not in remaining:
                    return None, None
        
        return L, R

    def get_greedy_partition(seed_val, max_seeds=15, time_limit=5.0):
        random.seed(seed_val)
        remaining = set(edge_set)
        bicliques = []
        start = time.time()
        
        while remaining:
            if time.time() - start > time_limit:
                # Fallback: cover rest with single edges
                for e in remaining:
                    bicliques.append(({e[0]}, {e[1]}))
                break
            
            # Compute degrees in remaining
            rem_deg = [0] * n
            for a, b in remaining:
                rem_deg[a] += 1
                rem_deg[b] += 1
            
            # Sort edges by degree sum (density heuristic)
            sorted_edges = sorted(remaining, key=lambda e: rem_deg[e[0]] + rem_deg[e[1]], reverse=True)
            
            best_L = None
            best_R = None
            best_count = 0
            
            # Try top K seeds
            K = min(max_seeds, len(sorted_edges))
            for seed_edge in sorted_edges[:K]:
                u, v = seed_edge
                vertex_order = list(range(n))
                random.shuffle(vertex_order)
                
                L, R = expand_biclique(u, v, remaining, vertex_order)
                
                if L is not None:
                    count = len(L) * len(R)
                    if count > best_count:
                        best_count = count
                        best_L = L
                        best_R = R
            
            if best_L is None or best_count == 0:
                # Single edge fallback
                e = next(iter(remaining))
                bicliques.append(({e[0]}, {e[1]}))
                remaining.discard(e)
            else:
                bicliques.append((best_L, best_R))
                for l in best_L:
                    for r in best_R:
                        e = (min(l, r), max(l, r))
                        remaining.discard(e)
                
        return bicliques

    def merge_pass(bicliques, max_iter=50):
        """
        Attempt to merge pairs of bicliques if their union forms a valid biclique
        and covers exactly the union of their edges.
        """
        bicliques = list(bicliques)
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            n_b = len(bicliques)
            
            for i in range(n_b):
                for j in range(i + 1, n_b):
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    
                    # Try merging: L_new = L1 | L2, R_new = R1 | R2
                    # But we must ensure L_new and R_new are disjoint.
                    # Also, we must ensure that all edges in L_new x R_new are in the union of edges covered by i and j.
                    
                    L_new = L1 | L2
                    R_new = R1 | R2
                    
                    if L_new & R_new:
                        continue
                    
                    # Check if L_new x R_new is a subset of (E_i U E_j)
                    # And conversely, E_i U E_j is a subset of L_new x R_new (which is true by definition of coverage)
                    # Actually, we need L_new x R_new == E_i U E_j for a clean merge into one biclique.
                    # If L_new x R_new contains edges not in E_i U E_j, the merge is invalid (covers extra edges).
                    # If E_i U E_j contains edges not in L_new x R_new, that's impossible since E_i subset L1xR1 subset L_newxR_new.
                    
                    E_i = set()
                    for l in L1:
                        for r in R1:
                            E_i.add((min(l, r), max(l, r)))
                    
                    E_j = set()
                    for l in L2:
                        for r in R2:
                            E_j.add((min(l, r), max(l, r)))
                    
                    E_target = E_i | E_j
                    
                    valid = True
                    for l in L_new:
                        for r in R_new:
                            e = (min(l, r), max(l, r))
                            if e not in E_target:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Merge
                        bicliques[i] = (L_new, R_new)
                        bicliques.pop(j)
                        improved = True
                        break
                if improved:
                    break
                    
        return bicliques

    def local_search_refine(bicliques, time_limit=10.0):
        """
        Try to remove edges from a biclique to make it mergeable, or split/swap.
        For now, just do merge_pass multiple times with slight perturbations if needed.
        """
        bicliques = list(bicliques)
        start = time.time()
        
        # Multiple merge passes
        while time.time() - start < time_limit:
            new_bics = merge_pass(bicliques)
            if len(new_bics) == len(bicliques):
                break
            bicliques = new_bics
            
        return bicliques

    # Main logic
    start_time = time.time()
    deadline = start_time + 80.0
    
    best_bicliques = None
    best_count = float('inf')
    
    # Run multiple greedy instances
    for seed in range(100):
        if time.time() > deadline - 15.0:
            break
            
        bics = get_greedy_partition(seed, max_seeds=10, time_limit=3.0)
        
        if len(bics) < best_count:
            best_count = len(bics)
            best_bicliques = bics
            
    # Refine the best one
    if best_bicliques:
        final_bics = local_search_refine(best_bicliques, time_limit=10.0)
        best_count = len(final_bics)
        best_bicliques = final_bics
        
    # If we have time, try refining other top solutions?
    # For now, stick to the best.
    
    # Convert to required format
    result = []
    for L, R in best_bicliques:
        left = [int(x) for x in L]
        right = [int(x) for x in R]
        result.append([left, right])
        
    return result

if __name__ == '__main__':
    res = construct()
    print(len(res))
