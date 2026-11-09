import random
import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Precompute adjacency matrix
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    # Helper to check if a set of vertices forms a biclique
    # L and R are sets of vertices.
    # Must check all (u,v) in L x R are edges.
    def is_biclique(L, R):
        for u in L:
            for v in R:
                if not adj[u][v]:
                    return False
        return True

    # Helper to get the set of edges covered by L, R
    def get_edges(L, R):
        e_set = set()
        for u in L:
            for v in R:
                if adj[u][v]:
                    e_set.add(frozenset((u, v)))
        return e_set

    # Greedy partition of a set of edges
    def greedy_partition(edge_list):
        remaining = set(frozenset(e) for e in edge_list)
        bicliques = []
        
        while remaining:
            # Pick a random edge to start
            # To improve, we can pick the edge with highest potential
            # But for speed and simplicity, random is okay if we run many times.
            # Let's try to pick the edge that is part of the largest potential biclique.
            
            best_biclique = None
            best_size = 0
            
            # Sample a few edges to find a good starting biclique
            sample_edges = list(remaining)
            random.shuffle(sample_edges)
            sample_edges = sample_edges[:min(10, len(sample_edges))]
            
            for e in sample_edges:
                u, v = tuple(e)
                L = {u}
                R = {v}
                
                # Expand
                changed = True
                while changed:
                    changed = False
                    # Try adding to L
                    for u_cand in range(n):
                        if u_cand in L or u_cand in R:
                            continue
                        # Check if u_cand connected to all in R
                        ok = True
                        for v_r in R:
                            if not adj[u_cand][v_r] or frozenset((u_cand, v_r)) not in remaining:
                                ok = False
                                break
                        if ok:
                            L.add(u_cand)
                            changed = True
                    # Try adding to R
                    for v_cand in range(n):
                        if v_cand in L or v_cand in R:
                            continue
                        ok = True
                        for u_l in L:
                            if not adj[u_l][v_cand] or frozenset((u_l, v_cand)) not in remaining:
                                ok = False
                                break
                        if ok:
                            R.add(v_cand)
                            changed = True
                
                # Calculate coverage
                covered = 0
                for lu in L:
                    for rv in R:
                        if frozenset((lu, rv)) in remaining:
                            covered += 1
                
                if covered > best_size:
                    best_size = covered
                    best_biclique = (L, R)
                    
            if not best_biclique:
                # Fallback
                e = next(iter(remaining))
                u, v = tuple(e)
                best_biclique = ({u}, {v})
                best_size = 1

            L, R = best_biclique
            bicliques.append((L, R))
            
            # Remove covered edges
            for lu in L:
                for rv in R:
                    rem_edge = frozenset((lu, rv))
                    if rem_edge in remaining:
                        remaining.remove(rem_edge)
                        
        return bicliques

    # Main Logic: Iterated Local Search
    best_solution = None
    best_count = float('inf')
    
    start_time = time.time()
    max_time = 55.0 # Leave 5s buffer
    
    # Run multiple iterations
    while time.time() - start_time < max_time:
        # Generate initial solution
        sol = greedy_partition(edges)
        count = len(sol)
        
        if count < best_count:
            best_count = count
            best_solution = sol
            
        # Local Search: K-opt
        # Try to improve the current solution by re-partitioning subsets of bicliques
        # We operate on the list of bicliques
        current_sol = [ (set(L), set(R)) for L, R in sol ]
        
        improved = True
        while improved and time.time() - start_time < max_time:
            improved = False
            # Try K=2, 3
            for K in [2, 3]:
                if len(current_sol) < K:
                    continue
                # Pick K random indices
                indices = random.sample(range(len(current_sol)), K)
                # Collect edges from these bicliques
                edges_to_repartition = []
                for idx in indices:
                    L, R = current_sol[idx]
                    for u in L:
                        for v in R:
                            if adj[u][v]:
                                edges_to_repartition.append((u, v))
                
                # Repartition these edges
                new_bicliques = greedy_partition(edges_to_repartition)
                new_count = len(new_bicliques)
                
                if new_count < K:
                    # Replace the K bicliques with the new ones
                    # Remove the old ones
                    for idx in sorted(indices, reverse=True):
                        current_sol.pop(idx)
                    # Add the new ones
                    current_sol.extend(new_bicliques)
                    improved = True
                    # Update global best if better
                    if len(current_sol) < best_count:
                        best_count = len(current_sol)
                        best_solution = current_sol
                    break # Restart the K loop or just break inner? 
                elif new_count == K:
                    # Optional: accept with small probability to escape local minima?
                    # For now, reject to keep it deterministic-ish and fast.
                    pass

    # Convert best_solution to required format
    if best_solution is None:
        # Should not happen, but fallback
        best_solution = greedy_partition(edges)
        
    result = []
    for L, R in best_solution:
        L_list = [int(x) for x in sorted(L)]
        R_list = [int(x) for x in sorted(R)]
        result.append([L_list, R_list])
        
    return result
