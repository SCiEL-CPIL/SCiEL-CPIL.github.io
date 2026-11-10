# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time
    import sys

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def build_partition(n, k, p, seed):
        # Generate the graph
        rng = Random(seed)
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))

        # Build adjacency for fast clique checking
        # adj[u] = set of neighbors
        adj = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u+1, k):
                adj[u].add(v)
                adj[v].add(u)
        for (u, v) in cross_edges:
            adj[u].add(v)
            adj[v].add(u)

        # Uncovered edges
        uncovered = set()
        for u in range(k):
            for v in range(u+1, k):
                uncovered.add((u, v))
        for (u, v) in cross_edges:
            uncovered.add((u, v))

        # Candidate cliques
        # 1. Big clique C
        # 2. Maximal cliques involving independent vertices
        # We generate a list of potential cliques.
        # To keep it fast, we limit the size or number.
        
        candidates = []
        
        # Add big clique
        candidates.append(list(range(k)))
        
        # For each independent vertex, add {v} U N(v)
        for v in range(k, n):
            clique = [v] + list(adj[v])
            candidates.append(clique)
            
        # Also add some random subsets? No, let's stick to maximal.
        
        # Greedy Set Cover
        partition = []
        
        # To speed up, we can pre-compute edges for each candidate
        # But that's memory heavy.
        # We'll compute on the fly.
        
        # Sort candidates by size descending?
        # Or just iterate and pick best?
        # Iterating all candidates each step is O(M * E).
        # M ~ n. E ~ n^2.
        # Steps ~ E.
        # Total ~ n^3. 90^3 = 729k. Fast.
        
        active_candidates = candidates[:]
        
        while uncovered:
            best_clique = None
            best_count = 0
            
            for clique in active_candidates:
                # Count uncovered edges in this clique
                count = 0
                # Optimization: if count + (len(clique) * (len(clique)-1)/2 - best_count) <= best_count, skip?
                # Just compute.
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        u, v = clique[i], clique[j]
                        if u > v: u, v = v, u
                        if (u, v) in uncovered:
                            count += 1
                            
                if count > best_count:
                    best_count = count
                    best_clique = clique
            
            if best_clique is None or best_count == 0:
                # Should not happen if uncovered is not empty
                break
                
            partition.append(best_clique)
            
            # Remove covered edges
            for i in range(len(best_clique)):
                for j in range(i+1, len(best_clique)):
                    u, v = best_clique[i], best_clique[j]
                    if u > v: u, v = v, u
                    uncovered.discard((u, v))
                    
            # Remove best_clique from active to avoid re-picking same clique?
            # If we keep it, it might be picked again if it still covers edges?
            # No, we removed its edges. So count will be 0.
            # But checking it is wasted time.
            active_candidates.remove(best_clique)
            
            # If active_candidates is empty but uncovered is not, we have a problem.
            # This can happen if we didn't generate all possible cliques.
            # Fallback: add single edges
            if not active_candidates and uncovered:
                 # Add remaining edges as single cliques
                 for (u, v) in list(uncovered):
                     partition.append([u, v])
                 uncovered = set()
                 break

        return partition

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(build_partition(*spec))
    
    return results
# EVOLVE-BLOCK-END
