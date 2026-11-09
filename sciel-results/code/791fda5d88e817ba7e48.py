"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def min_clique_cover_bipartite(left_nodes, right_nodes, adj):
        """
        Find minimum number of cliques covering all edges in a bipartite graph.
        left_nodes: list of left vertices (clique part)
        right_nodes: list of right vertices (independent set part)
        adj: dict mapping left vertex -> set of right vertices it's connected to
        
        A clique in the bipartite graph is a complete bipartite subgraph K_{a,b}.
        We need to cover all edges with minimum such cliques.
        
        This is equivalent to finding a minimum edge cover by complete bipartite subgraphs.
        For sparse graphs, we can use a greedy approach or exact methods for small cases.
        
        Strategy: 
        1. If the bipartite graph is complete, one clique suffices.
        2. Otherwise, use a greedy set cover approach: repeatedly pick the largest complete bipartite subgraph.
        3. For very small cases, use exact search.
        """
        if not left_nodes or not right_nodes:
            return []
        
        # Build adjacency sets
        adj_set = {u: set(adj.get(u, set())) for u in left_nodes}
        
        # Check if complete bipartite
        total_edges = sum(len(s) for s in adj_set.values())
        if total_edges == len(left_nodes) * len(right_nodes):
            return [[list(left_nodes) + list(right_nodes)]]
        
        # Greedy: repeatedly find largest complete bipartite subgraph
        remaining_edges = set()
        for u in left_nodes:
            for v in adj_set[u]:
                remaining_edges.add((u, v))
        
        if not remaining_edges:
            return []
        
        cliques = []
        
        # Precompute: for each subset of right nodes, which left nodes are connected to all of them
        # For efficiency, work with bitmasks if right_nodes is small enough
        n_right = len(right_nodes)
        right_idx = {v: i for i, v in enumerate(right_nodes)}
        
        # Represent each left node's neighborhood as a bitmask over right_nodes
        left_masks = {}
        for u in left_nodes:
            mask = 0
            for v in adj_set[u]:
                mask |= (1 << right_idx[v])
            left_masks[u] = mask
        
        # Find all maximal complete bipartite subgraphs (cliques in the bipartite graph)
        # A clique is defined by a set of left nodes L' and right nodes R' such that
        # every node in L' is connected to every node in R'.
        # This means the intersection of neighborhoods of L' contains R', and R' is exactly that intersection (for maximality).
        
        # Greedy approach:
        while remaining_edges:
            # Find the largest complete bipartite subgraph in remaining edges
            # Try all subsets of right nodes (if small) or use greedy
            best_clique = None
            best_size = 0
            
            # For each subset of right nodes that could form a clique with some left nodes
            # Since right_nodes might be up to 60, we can't enumerate all subsets.
            # Instead, use a greedy heuristic:
            
            # Pick a random edge to start
            import random
            random.seed(42)  # Fixed seed for determinism
            edge = random.choice(list(remaining_edges))
            u0, v0 = edge
            
            # Build candidate clique starting from this edge
            # Left nodes connected to v0 in remaining edges
            left_candidates = [u for u in left_nodes if (u, v0) in remaining_edges]
            # Right nodes connected to u0 in remaining edges  
            right_candidates = [v for v in right_nodes if (u0, v) in remaining_edges]
            
            # Find the largest complete bipartite subgraph containing (u0, v0)
            # Greedily add nodes
            L = set(left_candidates)
            R = set(right_candidates)
            
            # Iterate until no more nodes can be added
            changed = True
            while changed:
                changed = False
                # Find left nodes connected to all right nodes in R
                new_L = set()
                for u in left_nodes:
                    if u not in L:
                        # Check if u is connected to all v in R in remaining edges
                        if all((u, v) in remaining_edges for v in R):
                            new_L.add(u)
                if new_L:
                    L.update(new_L)
                    changed = True
                
                # Find right nodes connected to all left nodes in L
                new_R = set()
                for v in right_nodes:
                    if v not in R:
                        if all((u, v) in remaining_edges for u in L):
                            new_R.add(v)
                if new_R:
                    R.update(new_R)
                    changed = True
            
            clique_vertices = list(L) + list(R)
            clique_edges = [(u, v) for u in L for v in R]
            
            if len(clique_edges) > best_size:
                best_size = len(clique_edges)
                best_clique = clique_vertices
            
            # Add this clique
            cliques.append(best_clique)
            # Remove its edges from remaining
            for u in best_clique[:len(left_nodes)]:
                for v in best_clique[len(left_nodes):]:
                    remaining_edges.discard((u, v))
        
        return cliques

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Separate edges: clique part (both < k), cross edges (one < k, one >= k)
        clique_edges = []
        cross_edges = []
        for u, v in edges:
            if u < k and v < k:
                clique_edges.append((u, v))
            else:
                cross_edges.append((u, v))
        
        # The clique part: all vertices 0..k-1 form a clique
        # We need to cover clique_edges with cliques, but since they're all in one clique {0..k-1},
        # we can just use that one clique if there are any clique edges.
        # But we also need to cover cross edges.
        
        # Strategy: 
        # 1. The full clique {0..k-1} covers all clique_edges.
        # 2. For cross edges, we need to cover them with cliques. Each such clique can include
        #    some subset of {0..k-1} and some subset of {k..n-1}.
        #    Since {k..n-1} is independent, a clique containing cross edges can contain at most
        #    one vertex from {k..n-1} OR multiple if they share the same neighborhood in {0..k-1}.
        
        # Let's group independent vertices by their neighborhood in the clique part.
        # Two independent vertices can be in the same clique iff they have the same neighborhood.
        
        # Build neighborhood for each independent vertex
        indep_vertices = list(range(k, n))
        neigh = {}
        for v in indep_vertices:
            neighbors_in_clique = set()
            for u in range(k):
                if (u, v) in cross_edges or (v, u) in cross_edges:
                    neighbors_in_clique.add(u)
            neigh[v] = neighbors_in_clique
        
        # Group by neighborhood
        from collections import defaultdict
        groups = defaultdict(list)
        for v in indep_vertices:
            key = frozenset(neigh[v])
            groups[key].append(v)
        
        # Now, for each group, we can form a clique with its neighborhood in the clique part.
        # But we want to minimize the total number of cliques.
        
        # The clique part {0..k-1} needs to be covered. If we include all of {0..k-1} in one clique,
        # that covers all clique edges. But we might be able to split it to reduce the total count.
        
        # Actually, the optimal strategy for split graphs:
        # - The clique part is a clique, so it can be one clique.
        # - Cross edges: group independent vertices by neighborhood. For each distinct neighborhood S,
        #   the set S ∪ {all indep vertices with neighborhood S} forms a clique.
        # - But we can also merge: if two neighborhoods S1 ⊆ S2, we might be able to share.
        
        # Simpler approach: 
        # 1. One clique for the entire clique part {0..k-1} (covers all clique_edges).
        # 2. For each distinct neighborhood S (non-empty), one clique S ∪ group(S).
        #    But this might not be minimal.
        
        # Let's try: the minimum clique cover of the split graph.
        # Known result: for split graphs, the minimum clique cover can be found by:
        # - The clique part K and independent set I.
        # - For each subset of I, the shadow is the intersection of their neighborhoods in K.
        # - A clique can contain a subset of I and their common neighbors in K.
        
        # Greedy algorithm:
        # 1. Start with the full clique K = {0..k-1} as one clique.
        # 2. For cross edges, use the grouping by neighborhood.
        
        # Actually, let's think differently. Each clique in the cover can be:
        # - A subset of K only (covers clique edges)
        # - A subset of K plus some vertices from I (covers cross edges)
        # - Since I is independent, a clique can contain at most the vertices from I that share the same neighborhood.
        
        # Optimal approach:
        # Let's compute the minimum number of cliques needed.
        # 
        # Key insight: In a split graph, any clique is either:
        # (a) A subset of K, or
        # (b) A set S ⊆ K plus a set T ⊆ I where every vertex in T is adjacent to every vertex in S.
        #
        # For case (b), T must be a subset of the independent vertices whose neighborhood contains S.
        #
        # To minimize the number of cliques:
        # 1. We need to cover all edges in K (the clique part). One clique K covers all of them.
        # 2. We need to cover all cross edges. Each cross edge (u, v) with u ∈ K, v ∈ I must be in some clique.
        #    Such a clique must contain u and v, and possibly other vertices.
        #
        # Strategy: 
        # - Use one clique for K: covers all clique edges.
        # - For cross edges, group I by their exact neighborhood in K. For each group with neighborhood S,
        #   use one clique S ∪ group. This covers all cross edges incident to vertices in that group.
        # - Total cliques = 1 + (number of distinct non-empty neighborhoods).
        #
        # But we can do better: if we don't use the full K as a separate clique, we might reduce the total.
        # For example, if all cross edges are covered by cliques that include parts of K, we might not need
        # a separate clique for K if K's edges are already covered... but no, K's edges are only covered by cliques containing both endpoints in K.
        
        # Let's just go with: 1 clique for K + one clique per distinct neighborhood.
        # This is simple and should be close to optimal.
        
        # Actually, we can potentially merge the K clique with some cross-edge cliques.
        # If a cross-edge clique contains all of K, then it also covers all clique edges.
        # So if there's a group with neighborhood = K, we can use K ∪ group as one clique,
        # which covers both clique edges and those cross edges.
        
        # Let's compute:
        # - Distinct neighborhoods of independent vertices
        # - If one neighborhood is the full K, we can use that clique to cover clique edges too.
        
        distinct_neighs = list(groups.keys())
        
        # Check if any neighborhood is the full clique
        full_k = frozenset(range(k))
        has_full_neigh = full_k in groups
        
        if has_full_neigh:
            # Use K ∪ groups[full_k] as one clique, which covers all clique edges and cross edges for that group
            cliques = [[list(range(k)) + groups[full_k]]]
            # Remove full_k from consideration
            remaining_neighs = [s for s in distinct_neighs if s != full_k]
        else:
            # Need a separate clique for K
            cliques = [[list(range(k))]]
            remaining_neighs = distinct_neighs
        
        # For each remaining neighborhood, add a clique
        for s in remaining_neighs:
            if s:  # Non-empty neighborhood
                clique = list(s) + groups[s]
                if len(clique) >= 2:
                    cliques.append(clique)
        
        # Verify: every edge must be in exactly one clique
        # This construction might have overlapping cliques, which violates the "exactly one" constraint.
        # We need a partition, not a cover!
        
        # The problem says "every edge must lie in exactly one listed clique", so it's a partition.
        # This means we can't have overlapping cliques.
        
        # Revised approach: we need to partition the edge set into cliques.
        # 
        # For the clique part K: all edges within K must be in exactly one clique.
        # For cross edges: each cross edge must be in exactly one clique.
        
        # Since K is a clique, we can put all K-edges in one clique K.
        # Then for cross edges, we need to partition them into cliques.
        # Each cross-edge clique can contain:
        # - Some subset of K vertices
        # - Some subset of I vertices
        # - Such that all cross edges between them are present, and no K-K edges are included (since those are already in the K clique).
        
        # Wait, if a clique contains two vertices from K, that edge would be in two cliques (the K clique and this one), which is not allowed.
        # So cross-edge cliques can contain at most one vertex from K? No, they can contain multiple K vertices, but then the K-K edges would be double-covered.
        
        # Actually, the constraint is that every edge lies in exactly one clique. So if we put K as a clique, no other clique can contain two K vertices.
        # Therefore, cross-edge cliques can contain at most one vertex from K... no wait, they can contain multiple K vertices as long as we don't count the K-K edges. But the clique is defined by its vertex set, and all edges within it must be covered.
        
        # I think the correct interpretation is: the listed sets are cliques (all pairs within are edges), and every edge in the graph is in exactly one of these cliques. So if a clique contains vertices {a, b, c} where a,b,c ∈ K, then edges (a,b), (a,c), (b,c) are all covered by this clique. If another clique also contains a and b, that would be a problem.
        
        # So we need a partition of the edge set into cliques.
        
        # For split graphs, a known approach:
        # - The clique K can be covered by a single clique {0..k-1}.
        # - Cross edges: for each independent vertex v, its neighbors in K form a set N(v). 
        #   We need to cover all cross edges. Each cross edge (u, v) with u ∈ N(v) must be in some clique.
        #   A clique containing v can also contain any subset of N(v) (and other independent vertices with the same neighborhood).
        #   But to avoid covering K-K edges, the clique should contain at most... actually, it can contain multiple K vertices, but then those K-K edges would be in this clique, not in the K clique.
        
        # So we have two choices for covering K-K edges:
        # (A) Use one clique for all of K, and ensure no other clique contains two K vertices.
        # (B) Distribute K-K edges across multiple cliques.
        
        # Option (A) is simpler. Let's go with that.
        # So: one clique {0..k-1}. All other cliques contain at most one vertex from K? No, they can contain multiple, but then we shouldn't have a separate K clique.
        
        # Let me re-read: "every edge must lie in exactly one listed clique". So it's a partition of E(G) into cliques.
        
        # For split graphs, the optimal partition:
        # - Group independent vertices by their neighborhood in K.
        # - For each group with neighborhood S, the set S ∪ group is a clique.
        # - The remaining K-K edges (those not covered by any S ∪ group) need to be covered.
        #   But S ∪ group only covers K-K edges if |S| ≥ 2 and we consider the edges within S.
        
        # Actually, if we use S ∪ group as a clique, it covers all edges within S (K-K edges) and all cross edges between S and group.
        # So we need to partition K-K edges across the various S's.
        
        # This is getting complex. Let's use a simpler heuristic:
        # 1. One clique for K: covers all K-K edges.
        # 2. For each independent vertex v, if it has any neighbors in K, we need to cover its cross edges.
        #    Group vertices with the same neighborhood. For each group, use one clique containing the neighborhood and the group.
        #    But this clique would also contain K-K edges within the neighborhood, which are already covered by the K clique.
        #    This violates the "exactly one" constraint.
        
        # So we can't use this directly. We need a different approach.
        
        # Alternative: don't use a separate K clique. Instead, distribute K-K edges across the cross-edge cliques.
        # For each group with neighborhood S, the clique S ∪ group covers K-K edges within S.
        # We need to ensure that every K-K edge is covered by exactly one such clique.
        # This means the sets S (neighborhoods) must form a partition of the K-K edges, which is hard.
        
        # Given the complexity, let's use a greedy edge partition approach:
        # 1. Sort all edges.
        # 2. Greedily form cliques: for each edge, if it's not covered, start a new clique with that edge,
        #    and add as many other uncovered edges as possible that form a clique with the current set.
        
        # This is a general greedy clique partition algorithm. It might not be optimal but should work.
        
        # Let's implement a greedy approach:
        edge_set = set((u, v) for u, v in edges)
        covered = set()
        cliques = []
        
        # Sort edges for determinism
        sorted_edges = sorted(edge_set)
        
        for e in sorted_edges:
            if e in covered:
                continue
            
            # Start a new clique with this edge
            u, v = e
            clique_vertices = {u, v}
            clique_edges = {e}
            
            # Try to add more edges that form a clique
            # An edge (a, b) can be added if a and b are both in clique_vertices, or
            # one is in clique_vertices and the other is connected to all vertices in clique_vertices.
            
            changed = True
            while changed:
                changed = False
                for f in sorted_edges:
                    if f in covered or f in clique_edges:
                        continue
                    a, b = f
                    # Check if adding (a, b) keeps the set a clique
                    # Both a and b must be connected to all existing vertices in the clique
                    if a in clique_vertices and b not in clique_vertices:
                        if all((a, x) in edge_set and (b, x) in edge_set for x in clique_vertices - {a}):
                            clique_vertices.add(b)
                            clique_edges.add(f)
                            changed = True
                    elif b in clique_vertices and a not in clique_vertices:
                        if all((b, x) in edge_set and (a, x) in edge_set for x in clique_vertices - {b}):
                            clique_vertices.add(a)
                            clique_edges.add(f)
                            changed = True
                    elif a not in clique_vertices and b not in clique_vertices:
                        if all((a, x) in edge_set and (b, x) in edge_set for x in clique_vertices):
                            clique_vertices.add(a)
                            clique_vertices.add(b)
                            clique_edges.add(f)
                            changed = True
            
            # Add this clique
            clique_list = list(clique_vertices)
            if len(clique_list) >= 2:
                cliques.append(clique_list)
            covered.update(clique_edges)
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve_instance(*spec))
    
    return results
# EVOLVE-BLOCK-END
