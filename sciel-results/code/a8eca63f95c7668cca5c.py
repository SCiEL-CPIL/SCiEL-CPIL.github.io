import time
import sys
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets for the original graph
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
    
    total_edges = len(edge_set)
    
    # Helper: Get the set of edges covered by a biclique (L, R)
    def get_biclique_edges(L, R):
        e_set = set()
        for u in L:
            for v in R:
                if u != v:
                    e_set.add((min(u, v), max(u, v)))
        return e_set
    
    # Helper: Expand a seed edge (u, v) to a maximal biclique in the FULL graph
    # This ensures all cross pairs are valid host edges.
    def expand_biclique(u, v):
        L = {u}
        R = {v}
        changed = True
        while changed:
            changed = False
            # Expand L: add w if all R are neighbors of w
            new_L = []
            for w in range(n):
                if w not in L and w not in R:
                    if R.issubset(adj[w]):
                        new_L.append(w)
            for w in new_L:
                L.add(w)
                changed = True
            
            # Expand R: add w if all L are neighbors of w
            new_R = []
            for w in range(n):
                if w not in L and w not in R:
                    if L.issubset(adj[w]):
                        new_R.append(w)
            for w in new_R:
                R.add(w)
                changed = True
        return L, R

    def solve(deadline):
        remaining_edges = set(edge_set)
        used_edges = set()
        bicliques = []
        
        # Pre-compute degrees for heuristic
        deg = [len(adj[i]) for i in range(n)]
        
        while remaining_edges:
            if time.time() > deadline:
                break
                
            best_biclique = None
            best_coverage = 0
            best_L = None
            best_R = None
            
            # We want to find a biclique that covers many remaining edges.
            # Strategy: Sample seeds from remaining edges.
            # To be fast, we don't check ALL remaining edges if there are many, but for n=30 it's small.
            # However, expanding a biclique is O(n^2) or so.
            # Let's sample up to 50 seeds if remaining > 50, else all.
            
            seed_list = list(remaining_edges)
            if len(seed_list) > 50:
                random.shuffle(seed_list)
                seed_list = seed_list[:50]
            else:
                # For small sets, maybe sort by degree sum to prioritize "dense" areas
                seed_list.sort(key=lambda e: deg[e[0]] + deg[e[1]], reverse=True)
            
            for seed in seed_list:
                u, v = seed
                L, R = expand_biclique(u, v)
                
                # Check if this biclique covers any used edges. 
                # If it does, we can't use it as-is because we need a PARTITION.
                # However, we can restrict R to only vertices such that (u,v) in R are not used?
                # Actually, the constraint is "cross-edge sets must partition all pinned graph edges exactly once".
                # This means if an edge is already covered by a previous biclique, it cannot be covered again.
                # So we need to find a subset of L x R that contains NO used edges.
                # Since L x R is a complete bipartite subgraph in the host, and we are adding it to the partition,
                # any edge in L x R that is in used_edges makes this candidate invalid UNLESS we remove those vertices/edges.
                # But removing edges from a biclique breaks the biclique property (it's no longer a complete bipartite subgraph in the host).
                # Wait, the definition says "Every cross pair must be a host edge". It does NOT say the biclique must be maximal.
                # So we can take ANY subset of L x R.
                # However, if we take a subset, it's still a biclique?
                # A biclique is defined by L and R. The edges are L x R.
                # If we remove an edge (u,v) from L x R, we can either remove u from L or v from R.
                # If we remove u from L, we might lose other edges.
                # Actually, the simplest way to handle "partition" is:
                # When we pick a biclique (L, R), we only cover the edges in L x R that are in remaining_edges.
                # But L x R must be a set of host edges.
                # If (u,v) in L x R is in used_edges, we cannot cover it.
                # So we must ensure that L x R is a subset of remaining_edges? No, that's too strict.
                # We just need to pick a set of edges E' such that E' is a biclique (complete bipartite in host) and E' is disjoint from used_edges.
                # If we expand to maximal (L, R), some edges in L x R might be used.
                # We need to find the largest biclique subgraph of (L, R) that is disjoint from used_edges.
                # This is equivalent to:
                # Start with L, R.
                # While there is an edge (u,v) in L x R that is in used_edges:
                #   Remove u from L OR v from R. Which one to remove?
                #   Heuristic: Remove the vertex with higher degree in the current (L, R) intersection with remaining?
                #   Or simply: if we remove u, we lose all edges incident to u in this biclique.
                #   Let's try to prune.
                
                # Let's compute the set of valid edges in this biclique
                valid_edges = set()
                for u in L:
                    for v in R:
                        if u != v:
                            e = (min(u, v), max(u, v))
                            if e in remaining_edges:
                                valid_edges.add(e)
                
                # If no valid edges, skip
                if not valid_edges:
                    continue
                
                # Now, can we represent valid_edges as a biclique?
                # Not necessarily. valid_edges is a subset of L x R.
                # We need to find L' subset of L and R' subset of R such that L' x R' = valid_edges?
                # No, we just need L' x R' to be a subset of valid_edges?
                # No, the biclique covers L' x R'. All those edges must be in remaining_edges.
                # So we need to find L', R' such that L' x R' is a subset of remaining_edges.
                # This is the "Maximum Biclique in a bipartite graph" problem where the bipartite graph is defined by remaining_edges induced on L, R?
                # Actually, the graph of remaining edges is bipartite between L and R? Not necessarily, there might be edges within L or within R, but we only care about cross edges.
                # So we have a bipartite graph G' with partitions L and R, and edges are remaining_edges.
                # We want to find a complete bipartite subgraph (biclique) in G'.
                # Since L and R come from a maximal biclique in the HOST graph, G' is a subgraph of the complete bipartite graph.
                # We can find the largest biclique in G' by pruning.
                
                # Greedy pruning:
                L_prime = set(L)
                R_prime = set(R)
                
                # Iterate until stable
                changed = True
                while changed:
                    changed = False
                    # For each u in L_prime, check if all v in R_prime are in remaining_edges (i.e., (u,v) is remaining)
                    # If not, remove u? Or remove the specific v?
                    # If we remove u, we might lose many edges.
                    # If we remove v, we might lose many edges.
                    
                    # Let's try to remove vertices that are "blocking" many edges.
                    # Actually, a simpler approach:
                    # If (u,v) is NOT in remaining_edges, then u cannot be connected to v in our biclique.
                    # So either u is not in L' or v is not in R'.
                    
                    # Let's try to keep the largest possible.
                    # This is essentially finding a maximal biclique in the bipartite graph of remaining edges between L and R.
                    # For small L, R (which they should be, as they are subsets of original maximal bicliques), this is fast.
                    
                    # Let's just try to prune based on the first conflict found.
                    # To be more robust, let's try to find the maximal biclique in the bipartite graph (L, R, E_rem).
                    
                    # Since L and R are small (typically < 10), we can afford a bit more work.
                    # Let's try to find the largest subset.
                    # Actually, the previous "expand" already made L and R maximal in the HOST.
                    # The remaining edges form a subgraph.
                    # We can use a simple greedy:
                    # Sort L by degree in G' (L-R remaining edges)
                    # Sort R by degree in G'
                    # Remove the vertex with lowest degree? No, that removes too much.
                    
                    # Alternative:
                    # Since we just need A valid biclique, not necessarily the largest, we can be simpler.
                    # But we want to minimize count, so larger is better.
                    
                    # Let's try:
                    # If there is a conflict (u,v) not in remaining, 
                    # check if u has many other valid neighbors in R_prime.
                    # If yes, remove v.
                    # If no, remove u.
                    
                    # This is getting complicated. Let's stick to a simpler heuristic:
                    # If (u,v) not in remaining, remove the one with lower degree in the current valid subgraph.
                    
                    # Let's precompute degrees in the valid subgraph
                    deg_L = {u: 0 for u in L_prime}
                    deg_R = {v: 0 for v in R_prime}
                    for u in L_prime:
                        for v in R_prime:
                            e = (min(u, v), max(u, v))
                            if e in remaining_edges:
                                deg_L[u] += 1
                                deg_R[v] += 1
                    
                    # Find a conflict
                    conflict_u = None
                    conflict_v = None
                    for u in L_prime:
                        if deg_L[u] == 0:
                            L_prime.remove(u)
                            changed = True
                            break
                        for v in R_prime:
                            e = (min(u, v), max(u, v))
                            if e not in remaining_edges:
                                conflict_u = u
                                conflict_v = v
                                break
                        if conflict_u:
                            break
                    
                    if not L_prime or not R_prime:
                        break
                        
                    if conflict_u is not None:
                        # Remove the one with lower degree
                        if deg_L[conflict_u] <= deg_R[conflict_v]:
                            L_prime.remove(conflict_u)
                        else:
                            R_prime.remove(conflict_v)
                        changed = True
                        # Recompute degrees? For small sets, yes.
                        # But this loop might be slow.
                        # Let's just recompute degrees every time we change.
                        deg_L = {u: 0 for u in L_prime}
                        deg_R = {v: 0 for v in R_prime}
                        for u in L_prime:
                            for v in R_prime:
                                e = (min(u, v), max(u, v))
                                if e in remaining_edges:
                                    deg_L[u] += 1
                                    deg_R[v] += 1
                        # Check for zero degree vertices again
                        for u in list(L_prime):
                            if deg_L[u] == 0:
                                L_prime.remove(u)
                                changed = True
                        for v in list(R_prime):
                            if deg_R[v] == 0:
                                R_prime.remove(v)
                                changed = True
                    else:
                        # No conflict found in this pass?
                        # Wait, if no conflict found, it means L_prime x R_prime is a subset of remaining_edges.
                        # So we are done.
                        changed = False
                        break

                if L_prime and R_prime:
                    coverage = len(L_prime) * len(R_prime)
                    # Verify all are remaining
                    # They should be by construction of the pruning loop if it terminated correctly.
                    # But let's be safe and count actual remaining edges covered.
                    actual_covered = 0
                    for u in L_prime:
                        for v in R_prime:
                            e = (min(u, v), max(u, v))
                            if e in remaining_edges:
                                actual_covered += 1
                    
                    if actual_covered > best_coverage:
                        best_coverage = actual_covered
                        best_L = L_prime
                        best_R = R_prime
            
            if best_L is not None:
                L = list(best_L)
                R = list(best_R)
                bicliques.append([L, R])
                
                # Update remaining and used
                for u in L:
                    for v in R:
                        e = (min(u, v), max(u, v))
                        if e in remaining_edges:
                            remaining_edges.remove(e)
                            used_edges.add(e)
            else:
                # Fallback: pick any remaining edge
                # This should not happen if graph is connected, but just in case.
                if remaining_edges:
                    e = next(iter(remaining_edges))
                    u, v = e
                    bicliques.append([[u], [v]])
                    remaining_edges.remove(e)
                    used_edges.add(e)
                else:
                    break
        
        return bicliques

    # Run multiple restarts to find the best partition
    best_result = None
    best_count = float('inf')
    
    # Set a seed for reproducibility in this run, but different for different restarts?
    # random.seed(42)
    
    num_restarts = 10
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Slightly perturb the randomness
        random.seed(i * 12345 + 67890)
        
        result = solve(deadline)
        
        # Validate result
        valid = True
        if len(result) == 0 and total_edges > 0:
            valid = False
        else:
            covered_edges = set()
            for L, R in result:
                if not L or not R:
                    valid = False
                    break
                # Check types
                if not isinstance(L, (list, tuple)) or not isinstance(R, (list, tuple)):
                    valid = False
                    break
                for u in L:
                    if not isinstance(u, int) or isinstance(u, bool):
                        valid = False
                        break
                if not valid: break
                for v in R:
                    if not isinstance(v, int) or isinstance(v, bool):
                        valid = False
                        break
                if not valid: break
                
                # Check disjoint sides
                if set(L).intersection(set(R)):
                    valid = False
                    break
                    
                # Check cross pairs are host edges and not repeated
                for u in L:
                    for v in R:
                        e = (min(u, v), max(u, v))
                        if e not in edge_set:
                            valid = False
                            break
                        if e in covered_edges:
                            valid = False
                            break
                        covered_edges.add(e)
                if not valid: break
            
            # Check all edges covered
            if valid and covered_edges != edge_set:
                valid = False
        
        if valid:
            count = len(result)
            if count < best_count:
                best_count = count
                best_result = result
        else:
            # If invalid, we might want to try again or fallback
            pass

    if best_result is None:
        # Fallback to a trivial valid solution: star decomposition or single edges
        # Single edges is always valid and covers all.
        best_result = []
        for u, v in edges:
            best_result.append([[u], [v]])
    
    # Convert to required format
    final_result = []
    for L, R in best_result:
        final_result.append([[int(x) for x in L], [int(y) for y in R]])
        
    return final_result
