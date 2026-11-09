"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    from collections import defaultdict

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        
        # Build adjacency sets for the full graph
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Separate vertices into K (clique part) and I (independent set part)
        K = list(range(k))
        I = list(range(k, n))
        
        # We need to partition all edges into cliques.
        # Key constraint: No two cliques can share more than one vertex from K,
        # because if two cliques share two vertices u, v in K, then the edge (u, v)
        # would be in both cliques, violating the "exactly one" condition.
        # 
        # Strategy:
        # 1. Group independent vertices by their exact neighborhood in K.
        #    For each group G with neighborhood S (S = N(v) for any v in G),
        #    the set S ∪ G forms a clique (since all v in G are adjacent to all u in S,
        #    and K is a clique).
        # 2. However, we cannot simply use one clique per group if neighborhoods overlap,
        #    because K-K edges in the intersection would be covered twice.
        # 3. To handle this, we process groups in an order and assign each group a subset
        #    of its neighborhood that is "free" (i.e., K-K edges within that subset are not
        #    already covered by previous cliques).
        # 4. After assigning cliques for all independent groups, any remaining K-K edges
        #    (i.e., edges within K not covered by any group clique) are covered by additional
        #    cliques consisting solely of vertices from K.
        
        # Step 1: Group independent vertices by neighborhood mask
        groups = defaultdict(list)
        for v in I:
            mask = 0
            for u in K:
                if u in adj[v]:
                    mask |= (1 << u)
            groups[mask].append(v)
        
        # Convert masks to sets of K vertices for easier manipulation
        mask_to_set = {}
        for mask in groups:
            s = set()
            for u in K:
                if mask & (1 << u):
                    s.add(u)
            mask_to_set[mask] = s
        
        # Step 2: Process groups to assign cliques
        # We want to maximize the size of each clique (S ∪ G) to minimize the total number of cliques.
        # But we must avoid covering K-K edges twice.
        # 
        # Idea: Process groups in descending order of group size? Or neighborhood size?
        # Let's try: For each group, we want to assign a subset S' ⊆ S such that:
        #   - All K-K edges within S' are not yet covered.
        #   - S' is as large as possible (to cover more K-K edges and reduce remaining work).
        #   - The clique is S' ∪ G.
        # 
        # But wait: If we assign S' ⊊ S, then the cross edges between (S \ S') and G are NOT covered
        # by this clique. They must be covered by other cliques. But G is assigned to this clique,
        # so any other clique containing a vertex v in G cannot contain two vertices from K that are
        # already in S'? Actually, the constraint is global: no two cliques can share two K vertices.
        # 
        # Let's rethink. If we use a clique C = S' ∪ G, then:
        #   - All cross edges between S' and G are covered.
        #   - All K-K edges within S' are covered.
        #   - The cross edges between (S \ S') and G are NOT covered by C.
        #   - These edges must be covered by other cliques. But any other clique that covers an edge
        #     (u, v) with u in (S \ S') and v in G must contain u and v. Since v is in G, and G is
        #     already in clique C, can v be in another clique? Yes, a vertex can be in multiple cliques,
        #     but an edge can be in only one. The constraint is on edges, not vertices.
        #     However, if another clique C2 contains v and two vertices u1, u2 from K, then the edge (u1, u2)
        #     would be in C2. If (u1, u2) was already covered by C (if u1, u2 in S'), that's a conflict.
        #     So we must ensure that no other clique contains two K vertices that are in S'.
        # 
        # This suggests that once we assign S' to a clique, no other clique can contain two vertices from S'.
        # 
        # This is getting complicated. Let's use a simpler, more robust approach:
        # 
        # Approach: Greedy edge-based clique partition, but optimized for split graphs.
        # 1. Start with all edges uncovered.
        # 2. While there are uncovered edges:
        #    a. Pick an uncovered edge (u, v).
        #    b. Find the largest possible clique C containing (u, v) such that all edges in C are uncovered.
        #    c. Add C to the partition and remove its edges from uncovered.
        # 
        # To find the largest clique containing (u, v):
        #   - Start with C = {u, v}.
        #   - Try to add vertices w one by one. A vertex w can be added if:
        #     * w is adjacent to all vertices in C.
        #     * All edges between w and vertices in C are uncovered.
        #   - To be efficient, we can prioritize adding vertices from I (independent set) if possible,
        #     because they only connect to K, and adding them doesn't create K-K edges.
        #     But actually, adding a vertex from K creates K-K edges, which might conflict with future cliques.
        #     However, since we're removing edges, it's fine as long as we don't double-cover.
        # 
        # Let's implement this greedy approach. It should be efficient enough for n=90.
        
        uncovered = set(edge_set)
        cliques = []
        
        # Precompute a list of vertices for iteration
        vertices = list(range(n))
        
        while uncovered:
            # Pick the first uncovered edge (for determinism)
            edge = sorted(uncovered)[0]
            u, v = edge
            
            # Find the maximal clique containing (u, v) in the subgraph of uncovered edges
            C = {u, v}
            
            # Try to add more vertices
            changed = True
            while changed:
                changed = False
                for w in vertices:
                    if w in C:
                        continue
                    # Check if w is adjacent to all vertices in C and all edges are uncovered
                    can_add = True
                    for x in C:
                        if w == x:
                            continue
                        e = (min(w, x), max(w, x))
                        if e not in uncovered:
                            can_add = False
                            break
                        if x not in adj[w]:
                            can_add = False
                            break
                    if can_add:
                        C.add(w)
                        changed = True
            
            # Add the clique to the partition
            clique_list = list(C)
            if len(clique_list) >= 2:
                cliques.append(clique_list)
                # Remove all edges within this clique from uncovered
                for i in range(len(clique_list)):
                    for j in range(i + 1, len(clique_list)):
                        e = (min(clique_list[i], clique_list[j]), max(clique_list[i], clique_list[j]))
                        uncovered.discard(e)
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve_instance(*spec))
    
    return results
# EVOLVE-BLOCK-END
