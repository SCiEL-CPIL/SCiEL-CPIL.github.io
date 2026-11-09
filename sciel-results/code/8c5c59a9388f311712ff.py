"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time

    start_time = time.time()
    deadline = 120.0  # Safety margin from 150s limit

    def instance(n, k, p, seed):
        """Generates the split graph edges as per the task specification."""
        rng = Random(seed)
        edges = []
        # Edges within the clique part [0, k)
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # Edges between clique part [0, k) and independent set [k, n)
        # The task description says: "each cross pair joined with probability p"
        # The seed program iterates: for v in range(k, n) for u in range(k)
        # We must replicate this exact random number consumption to get the same graph.
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, k, p, seed):
        """
        Finds a minimal clique partition for the split graph.
        
        Structure:
        - Clique part K = {0, ..., k-1}
        - Independent set I = {k, ..., n-1}
        
        Strategy:
        1. The edges within K form a complete graph. They are covered by the single clique K.
        2. The edges between K and I form a bipartite graph. 
           Since I is an independent set, any clique containing a vertex v in I 
           can contain no other vertex from I. 
           Thus, all edges incident to v must be covered by cliques containing v.
           The most efficient way is to use the single clique {v} U N(v) for each v in I 
           that has at least one neighbor.
           This covers all edges incident to v exactly once.
           Edges are not covered by any other clique because:
           - K does not contain v.
           - Other cliques {u} U N(u) for u in I, u != v, do not contain v (since I is independent) 
             and do not contain any neighbor of v in K? 
             Wait. A clique {u} U N(u) contains vertices from K that are neighbors of u.
             Does it contain an edge (x, v) where x in K? No, it doesn't contain v.
             So edge (x, v) is only in the clique containing v.
        
        Therefore, the set of cliques is:
        - [0, ..., k-1]
        - For each v in [k, n-1] with deg(v) > 0: [v] + sorted(neighbors(v))
        """
        edges = instance(n, k, p, seed)
        
        # Build adjacency list for efficient neighbor lookup
        # We only need neighbors for vertices in I to form the cliques,
        # but we need to verify edges later.
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = []
        
        # 1. The clique covering the clique part K
        # This covers all edges (u, v) where u, v < k
        K_clique = list(range(k))
        if len(K_clique) >= 2:
            cliques.append(K_clique)
        # Note: If k < 2, there are no edges within K, so no clique needed for K part?
        # But K is a clique. If k=1, no edges. If k=0, no edges.
        # The problem states "clique on the first k vertices". 
        # If k=10, it's a clique.
        # The rule: "Every listed set must be a clique with at least two vertices".
        # If k < 2, we don't list K_clique because it would have size < 2.
        # But in our instances k is 10, 20, 30. So it's always >= 2.
        
        # 2. Cliques covering cross edges
        # For each v in I (k to n-1), if it has neighbors, form a clique
        for v in range(k, n):
            neighbors = adj[v]
            if neighbors:
                # The clique is {v} U neighbors
                # Neighbors are all in K, so they form a clique with v.
                clique_v = [v] + sorted(list(neighbors))
                cliques.append(clique_v)
        
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    
    for spec in specs:
        if time.time() > deadline - 5:
            # Fallback to seed solution if we are running out of time (unlikely)
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
            continue
            
        cliques = solve_instance(*spec)
        
        # Optional: Verification step (fast for small n)
        # Check 1: Each clique is a valid clique of size >= 2
        # Check 2: Every edge is covered exactly once
        
        # Build edge set for verification
        edge_set = set()
        for u, v in instance(spec[0], spec[1], spec[2], spec[3]):
            edge_set.add((min(u, v), max(u, v)))
            
        covered_edges = {}
        for c in cliques:
            if len(c) < 2:
                raise ValueError("Clique must have at least 2 vertices")
            # Check if it's a clique
            for i in range(len(c)):
                for j in range(i+1, len(c)):
                    u, v = c[i], c[j]
                    e = (min(u, v), max(u, v))
                    if e not in edge_set:
                        raise ValueError(f"Invalid clique: {c} contains non-edge {e}")
                    if e in covered_edges:
                        raise ValueError(f"Edge {e} covered multiple times")
                    covered_edges[e] = True
            
        # Check all edges covered
        if len(covered_edges) != len(edge_set):
            raise ValueError("Not all edges covered")
            
        results.append(cliques)

    return results
# EVOLVE-BLOCK-END
