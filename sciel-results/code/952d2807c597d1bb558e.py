import time
import random
import itertools

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _expand_biclique(L, R, adj, uncovered):
    """
    Expand L and R to be a maximal biclique with respect to uncovered edges.
    Returns (L, R) as sets.
    """
    L = set(L)
    R = set(R)
    changed = True
    while changed:
        changed = False
        # Find candidates for L: vertices connected to all in R via uncovered edges
        # v can be in L if for all r in R, (v, r) in uncovered
        # This is equivalent to: v is in intersection of adj[r] for all r in R, 
        # but restricted to edges in uncovered.
        
        # Optimization: Precompute common neighbors in uncovered?
        # For small n=30, simple loops are fine.
        
        candidates_L = []
        for v in range(len(adj)):
            if v in L or v in R:
                continue
            # Check if v is connected to all in R via uncovered edges
            valid = True
            for r in R:
                if frozenset((v, r)) not in uncovered:
                    valid = False
                    break
            if valid:
                candidates_L.append(v)
        
        if candidates_L:
            # Add all valid candidates to L? 
            # Adding all might be risky if they disconnect R? 
            # No, if v is connected to all R, adding v to L doesn't remove edges from R.
            # It just adds edges (v, r) to the biclique.
            # So we can add all.
            for v in candidates_L:
                L.add(v)
                changed = True
        
        candidates_R = []
        for v in range(len(adj)):
            if v in L or v in R:
                continue
            valid = True
            for l in L:
                if frozenset((v, l)) not in uncovered:
                    valid = False
                    break
            if valid:
                candidates_R.append(v)
        
        if candidates_R:
            for v in candidates_R:
                R.add(v)
                changed = True
                
    return L, R

def _count_covered(L, R, uncovered):
    count = 0
    for u in L:
        for v in R:
            if frozenset((u, v)) in uncovered:
                count += 1
    return count

def _greedy_partition(edges, adj, n, time_limit):
    uncovered = set(frozenset(e) for e in edges)
    bicliques = []
    start_time = time.time()
    
    while uncovered:
        if time.time() > time_limit:
            # Fallback: trivial partition
            for e in uncovered:
                u, v = e
                bicliques.append([u, [v]])
            return bicliques

        best_biclique = None
        best_count = 0
        
        # Heuristic: Pick a few random seeds from uncovered
        # Also try the "most constrained" edge?
        # Let's try all edges in uncovered if small, else random sample.
        # n=30, E~150. E^2 is 22500. Expansion is fast.
        # Let's just iterate all uncovered edges to be safe, but break early if we find a huge one?
        
        seeds = list(uncovered)
        # Shuffle to avoid bias? Or keep deterministic for reproducibility?
        # The task says "random graph", but P is fixed.
        # Let's use random sampling if E is large, else all.
        if len(seeds) > 50:
            random.shuffle(seeds)
            seeds = seeds[:50] # Sample top 50? No, this might miss the best.
            # Better: Use a priority queue or just check all if it's fast enough.
            # With n=30, checking all 150 edges is very fast.
            # Let's stick to all for now, but optimize expansion.
            
        for e in seeds:
            u, v = e
            # Start with {u}, {v}
            L, R = _expand_biclique({u}, {v}, adj, uncovered)
            
            count = _count_covered(L, R, uncovered)
            if count > best_count:
                best_count = count
                best_biclique = (L, R)
                
                # Early termination if we found a biclique covering all remaining edges?
                if count == len(uncovered):
                    break

        if best_biclique is None:
            # Should not happen
            break
            
        L, R = best_biclique
        bicliques.append([list(L), list(R)])
        
        # Remove covered edges
        for u in L:
            for v in R:
                e = frozenset((u, v))
                if e in uncovered:
                    uncovered.remove(e)
                    
    return bicliques

def _try_merge(bicliques, adj, edges_set):
    """
    Try to merge pairs of bicliques.
    Returns True if a merge occurred.
    """
    n_b = len(bicliques)
    merged = False
    i = 0
    while i < n_b:
        if i >= len(bicliques): break
        L1, R1 = bicliques[i][0], bicliques[i][1]
        
        j = i + 1
        while j < len(bicliques):
            L2, R2 = bicliques[j][0], bicliques[j][1]
            
            # Check if union forms a valid biclique
            # L_new = L1 U L2, R_new = R1 U R2
            # Must ensure L_new and R_new are disjoint? 
            # In our construction, L and R are disjoint.
            # If L1 intersects R2, we can't simply union.
            # We must check if the cross edges between L_new and R_new are all in the graph.
            # AND, the set of edges covered must be exactly the union of edges from B1 and B2.
            
            L_new = set(L1) | set(L2)
            R_new = set(R1) | set(R2)
            
            # Check disjointness
            if L_new & R_new:
                j += 1
                continue
            
            # Check if all cross edges exist in graph
            valid = True
            edges_in_new_biclique = set()
            for u in L_new:
                for v in R_new:
                    e = frozenset((u, v))
                    if e not in edges_set:
                        valid = False
                        break
                    edges_in_new_biclique.add(e)
                if not valid:
                    break
            
            if not valid:
                j += 1
                continue
                
            # Check if edges_in_new_biclique is exactly the union of edges from B1 and B2
            edges_b1 = set()
            for u in L1:
                for v in R1:
                    edges_b1.add(frozenset((u, v)))
                    
            edges_b2 = set()
            for u in L2:
                for v in R2:
                    edges_b2.add(frozenset((u, v)))
                    
            union_edges = edges_b1 | edges_b2
            
            if edges_in_new_biclique == union_edges:
                # Merge is valid
                bicliques[i] = [list(L_new), list(R_new)]
                del bicliques[j]
                merged = True
                # Restart inner loop? Or just continue?
                # If we merge, indices shift.
                j = i + 1
            else:
                j += 1
                
        i += 1
        
    return merged

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    edges_set = set(frozenset(e) for e in edges)
    
    start_time = time.time()
    deadline = start_time + 50 # Leave 10s for safety
    
    best_result = None
    best_count = float('inf')
    
    # Multi-restart with randomization in seed selection?
    # The current _greedy_partition is deterministic.
    # Let's add a bit of randomness by shuffling the order of seeds in _greedy_partition?
    # Or just run it once, then do local search.
    
    # 1. Greedy Construct
    bicliques = _greedy_partition(edges, adj, n, deadline - 20)
    
    # 2. Local Search: Merge
    # Run merge multiple times until no more merges
    while time.time() < deadline - 5:
        if not _try_merge(bicliques, adj, edges_set):
            break
            
    current_count = len(bicliques)
    if current_count < best_count:
        best_count = current_count
        best_result = [ [list(map(int, L)), list(map(int, R))] for L, R in bicliques ]
        
    # 3. Try a few more restarts if time permits?
    # The greedy is quite strong. Let's try randomizing the seed order.
    # We can modify _greedy_partition to accept a shuffle flag.
    # For now, let's just return the best found.
    
    if best_result is None:
        # Fallback
        return [[e[0], [e[1]]] for e in edges]
        
    return best_result
