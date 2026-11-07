def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        
        # Build adjacency for cross edges to find neighbors in K for each v in I
        # K is 0..k-1, I is k..n-1
        # Edges are sorted.
        
        # We need to cover K with one clique: [0, ..., k-1]
        cliques = [list(range(k))]
        
        # For each v in I, find neighbors in K
        # Initialize neighbor sets for v in I
        # Since n is small (max 90), we can just iterate edges.
        
        # Create a set of neighbors for each v in I
        # Actually, we just need to know if N_v is empty or not, and what it is.
        
        # Let's collect edges involving I
        # Edge (u, v) where u < k and v >= k
        # Or (u, v) where u >= k and v < k? 
        # In instance: 
        # edges from K: u in 0..k-1, v in u+1..k-1. Both in K.
        # edges from Cross: v in k..n-1, u in 0..k-1. So u is in K, v is in I.
        # So all cross edges are (u, v) with u < k <= v.
        
        neighbors_in_k = {v: [] for v in range(k, n)}
        
        for u, v in edges:
            if v >= k: # This is a cross edge (u, v)
                neighbors_in_k[v].append(u)
            # else it's an edge in K, already covered by the main clique
        
        for v in range(k, n):
            if neighbors_in_k[v]:
                # Create clique: neighbors_in_k[v] + [v]
                # Ensure it's a list of ints
                clique = neighbors_in_k[v] + [v]
                cliques.append(clique)
                
        results.append(cliques)
        
    return results
