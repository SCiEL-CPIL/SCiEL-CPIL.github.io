"""Strategy: Merge universal neighbors into the core clique to reduce clique count."""

def construct():
    from random import Random
    import time

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        """
        Returns a list of cliques covering every edge exactly once.
        Strategy:
        1. Identify all vertices in I (range k..n) that are connected to ALL vertices in C (range 0..k).
           These are 'universal' with respect to C.
        2. Form one large clique: C U {universal vertices in I}.
           This covers all C-C edges and all edges between C and universal I vertices.
        3. For every other cross edge (c, i) where i is NOT universal, form a 2-clique [c, i].
           Since i is not universal, it has at least one missing edge to C. 
           Wait, if i is not universal, it might still be connected to some c's.
           But we cannot put i into the big clique because it's missing at least one edge to C.
           Can we group non-universal i's together? No, because I is an independent set.
           So each non-universal i must be handled edge-by-edge or in small cliques?
           Actually, if i is not universal, we can still form cliques of size > 2?
           No, a clique containing i and some c's must be a subset of N(i) ∪ {i}.
           Since N(i) ⊆ C, and C is a clique, any subset of N(i) ∪ {i} is a clique.
           However, edges {c, i} for c in N(i) are distinct. 
           If we put i with multiple c's in one clique, we cover multiple edges with one clique.
           Is it better to put ALL neighbors of i into one clique [i] + N(i)?
           Yes! For a fixed i, if we create one clique [i] ∪ N(i), it covers all edges incident to i.
           This uses 1 clique for all edges of i.
           The previous strategy used deg(i) cliques of size 2.
           So for ANY i, using one clique [i] ∪ N(i) is strictly better than deg(i) 2-cliques (since deg(i) >= 1, and 1 < deg(i) for deg(i)>1).
           
           Wait, why didn't the current program do this?
           Current program: `cliques.append([u, v])` for each edge.
           If i is connected to 5 core vertices, current uses 5 cliques.
           New strategy: Use 1 clique [i, c1, c2, c3, c4, c5].
           This is valid because C is a clique, so {c1..c5} is a clique, and i is connected to all of them, so {i, c1..c5} is a clique.
           
           So the optimal strategy is:
           - One clique for the core C? No, if we merge i's into separate cliques, the core edges C-C are still uncovered.
           - We need to cover C-C edges. We can use one big clique C.
           - But if we use C as a clique, we cannot include any i in it unless i is universal.
           - If i is universal, we can add it to the C clique.
           - If i is NOT universal, we cannot add it to the C clique.
           - So for non-universal i, we must cover its edges with cliques that do NOT include the missing core vertices.
           - The most efficient way to cover all edges incident to a specific i is to create ONE clique consisting of i and all its neighbors in C: [i] ∪ N(i).
           - This covers all edges {c, i} for c in N(i).
           - Does this conflict with the C clique? 
             The C clique covers edges {c1, c2} for c1, c2 in C.
             The i-clique covers edges {c, i}.
             There is no overlap in edges.
             Are there any edges covered twice? No.
             Are there any edges missed? 
             C-C edges: covered by C clique.
             C-I edges: covered by i-cliques (one per i).
             I-I edges: none exist.
             
           So the total number of cliques is:
           1 (for C) + number of i in I that have at least one neighbor in C.
           Wait, if an i has NO neighbors in C, it contributes 0 edges, so we don't need a clique for it.
           
           Let's verify:
           Instance 1: n=30, k=10. I has 20 vertices.
           p=0.7.
           Expected neighbors per i: 7.
           Prob(i has 0 neighbors) = (1-0.7)^10 = 0.3^10 ≈ 0.000059.
           So almost all 20 i's have neighbors.
           Cliques = 1 + 20 = 21.
           Current program: 1 + sum(deg(i)). Sum(deg(i)) ≈ 20 * 7 = 140. Total 141.
           Huge improvement!
           
           Let's check constraints:
           "Every listed set must be a clique with at least two vertices"
           - C clique: size k=10 (ok).
           - i-clique: [i] ∪ N(i). Size = 1 + deg(i).
           - If deg(i) >= 1, size >= 2.
           - If deg(i) == 0, we don't create a clique.
           
           "Every edge must lie in exactly one listed clique"
           - Edge {c1, c2} in C: in C clique. Not in any i-clique (i-cliques contain i and neighbors of i, but i is not in C, and c1,c2 are in C. Wait.
             Does the i-clique contain edges between core vertices?
             Yes! The clique [i, c1, c2, ...] contains the edge {c1, c2}.
             This edge is ALSO in the C clique.
             This violates "exactly one".
             
           Ah! This is the catch.
           If we use a clique [i] ∪ N(i), it includes all edges among N(i).
           Those edges are also in the main C clique.
           So we cannot use the full C clique AND the i-cliques simultaneously without double-covering C-C edges.
           
           We need a partition of edges.
           Option A:
           - Use one big clique for C.
           - For each i, we need to cover edges {c, i} for c in N(i).
           - We cannot use a clique containing two core vertices c1, c2 and i, because {c1, c2} is already covered by C clique.
           - So any clique containing i can contain AT MOST ONE core vertex.
           - Therefore, each edge {c, i} must be in a distinct clique if we stick to this structure?
           - No, wait. If a clique contains i and c1, it covers {c1, i}. It does NOT cover {c1, c2} unless c2 is also in the clique.
           - If we restrict i-cliques to size 2 ([c, i]), then we don't cover any C-C edges.
           - If we allow i-cliques to be larger, say [i, c1, c2], we cover {c1, c2} again.
           - So, to avoid double covering C-C edges, any clique that contains an I-vertex i can contain AT MOST ONE vertex from C.
           - Proof: If clique K contains i, c1, c2 (c1 != c2, both in C), then edge {c1, c2} is in K.
             If K is distinct from the main C clique, then {c1, c2} is covered twice (once in C clique, once in K).
             If we don't use a single C clique, but partition C edges differently?
             
           Let's rethink.
           We need to partition the edge set E into cliques.
           E = E_C ∪ E_CI. (E_II is empty).
           E_C is a complete graph K_k.
           E_CI is a bipartite graph between C and I.
           
           Constraint: Each clique in the partition is a clique in G.
           A clique in G can be:
           1. Subset of C (covers only E_C edges).
           2. {i} ∪ S where S ⊆ N(i) ⊆ C (covers edges {i, s} for s in S, AND edges within S).
           
           If we use type 2 cliques, they cover internal edges of S.
           So the edges covered by type 2 cliques are a subset of E_CI ∪ E_C.
           Specifically, for a type 2 clique K_i = {i} ∪ S_i, it covers:
           - Edges {i, s} for s in S_i (from E_CI).
           - Edges {s1, s2} for s1, s2 in S_i (from E_C).
           
           We must partition E.
           Let P_C be the set of cliques of type 1 (subsets of C).
           Let P_I be the set of cliques of type 2 (containing exactly one i from I).
           
           Edges in E_C must be covered exactly once by P_C ∪ P_I.
           Edges in E_CI must be covered exactly once by P_I (since P_C has no I vertices).
           
           For a fixed i, let N(i) be its neighbors in C.
           All edges {i, c} for c in N(i) must be covered by cliques in P_I that contain i.
           Since i is in I, and I is independent, i can only be in cliques of type 2.
           Can i be in multiple cliques?
           Yes.
           Let K_i1, K_i2, ... be the cliques in P_I containing i.
           Each K_ij = {i} ∪ S_ij where S_ij ⊆ N(i).
           The edges {i, c} are covered if c is in some S_ij.
           Since each edge {i, c} must be covered exactly once, the sets S_ij must form a PARTITION of N(i).
           i.e., S_i1 ∪ S_i2 ∪ ... = N(i) and S_ij ∩ S_ik = ∅ for j != k.
           
           Now consider the edges in E_C covered by these cliques.
           The clique K_ij = {i} ∪ S_ij covers all edges within S_ij.
           So, for each i, the partition of N(i) into S_ij determines which edges in E_C are "consumed" by the i-cliques.
           Specifically, the set of E_C edges covered by i-cliques is ∪_j E(S_ij), where E(S) is the set of edges in the induced subgraph on S.
           
           The remaining edges in E_C (those not covered by any i-clique) must be covered by P_C (cliques entirely within C).
           To minimize total cliques, we want to minimize |P_I| + |P_C|.
           
           |P_I| = sum over i in I of (number of parts in partition of N(i)).
           If we put all of N(i) into one part S_i1 = N(i), then |P_I| contribution for i is 1.
           This consumes E(N(i)) from E_C.
           
           If we split N(i) into singletons {c}, then |P_I| contribution is |N(i)|.
           This consumes no E_C edges (since E({c}) is empty).
           
           Trade-off:
           Using 1 clique for i saves (|N(i)| - 1) cliques in P_I.
           But it consumes E(N(i)) edges from E_C, which might force us to use MORE cliques in P_C to cover the remaining E_C edges.
           Actually, consuming edges from E_C REDUCES the number of edges left for P_C.
           So consuming more E_C edges is generally GOOD for reducing P_C size, PROVIDED the remaining E_C edges can be covered efficiently.
           
           However, P_C must cover the remaining E_C edges with cliques.
           The remaining E_C graph is G_C_remaining = (C, E_C \ ∪_i E(S_i1)).
           We need to partition E(G_C_remaining) into cliques.
           The minimum number of cliques to partition a graph is the chromatic number of the complement? No, it's related to the clique cover number.
           For a complete graph K_k, 1 clique suffices.
           If we remove some edges, the remaining graph is less dense, might need more cliques?
           Actually, if we remove edges, the graph becomes sparser. A sparse graph might need MORE cliques to cover its edges?
           No, a sparse graph has FEWER edges. But the cliques must be cliques in the ORIGINAL graph?
           No, the cliques in the partition must be cliques in G.
           Since G[C] is a clique, any subset of C is a clique in G.
           So P_C can be ANY partition of the remaining edges of G[C] into subsets of C.
           Wait, if we just need to cover the remaining edges, and any subset of C is a clique, we can cover all remaining edges with ONE clique: C itself?
           No, because the cliques in P_C are disjoint in terms of EDGES, not vertices.
           If we use clique C in P_C, it covers ALL edges in E_C.
           But we already used some edges in E_C via P_I.
           So we cannot use C in P_C if any E_C edge was covered by P_I.
           
           We need to partition E_C \ E_covered_by_PI into cliques.
           Let H be the graph (C, E_C \ E_covered_by_PI).
           We need a clique partition of H.
           Since H is a subgraph of a complete graph, any clique in H is a subset of C.
           The minimum number of cliques to partition the edges of H is the "clique partition number" of H.
           
           If we choose S_i1 = N(i) for all i, we cover E(N(i)) for each i.
           The remaining edges are those {c1, c2} such that there is NO i with c1, c2 in N(i).
           i.e., {c1, c2} is not contained in any N(i).
           
           If we choose S_ij = {c} (singletons), we cover NO E_C edges.
           Then H = K_k.
           Clique partition of K_k is 1 (just {C}).
           Total cliques = sum |N(i)| + 1.
           This is the current strategy (roughly, current uses 2-cliques which is singletons).
           
           If we choose larger S_ij, we reduce P_I count but increase complexity of H.
           Does covering more E_C edges with P_I reduce the total?
           Example: k=3. C={1,2,3}. E_C = {{1,2}, {1,3}, {2,3}}.
           Suppose i is connected to all 3. N(i)={1,2,3}.
           Option 1: Singletons. P_I has 3 cliques: {i,1}, {i,2}, {i,3}. Covers no E_C.
           H = K_3. P_C needs 1 clique: {1,2,3}.
           Total = 3 + 1 = 4.
           
           Option 2: One set S={1,2,3}. P_I has 1 clique: {i,1,2,3}. Covers E({1,2,3}) = all E_C.
           H has no edges. P_C needs 0 cliques.
           Total = 1 + 0 = 1.
           
           Option 3: Partition N(i) into {1,2} and {3}.
           P_I has 2 cliques: {i,1,2}, {i,3}.
           {i,1,2} covers edge {1,2}.
           {i,3} covers no E_C.
           Remaining E_C: {1,3}, {2,3}.
           H has edges {1,3}, {2,3}.
           Can we cover these with 1 clique?
           Clique must be a subset of C.
           {1,3} is a clique. {2,3} is a clique.
           Can we use {1,2,3}? No, edge {1,2} is already covered.
           So we need 2 cliques: {1,3} and {2,3}.
           Total = 2 (P_I) + 2 (P_C) = 4.
           
           So Option 2 is best.
           
           General Strategy:
           For each i, we should try to group its neighbors into as few sets as possible, such that the remaining E_C graph is easy to partition.
           Ideally, if we can cover ALL E_C edges using the P_I cliques, we need 0 P_C cliques.
           This requires that for every edge {c1, c2} in C, there exists some i and some part S_ij of N(i) such that {c1, c2} ⊆ S_ij.
           
           If we use one part per i (S_i = N(i)), we cover E(N(i)).
           The union of E(N(i)) for all i must cover E_C.
           If there is an edge {c1, c2} such that no i is connected to both c1 and c2, then {c1, c2} is not covered by any P_I clique (if we use single parts).
           Then we must cover {c1, c2} in P_C.
           
           If we use single parts for all i, P_I size = number of i with N(i) != empty.
           Let U = ∪_i E(N(i)).
           Remaining edges R = E_C \ U.
           We need to partition R into cliques.
           Since R is a subgraph of K_k, we can cover R with at most |V(R)| - 1 cliques? No.
           Actually, since any subset of C is a clique, we can cover any set of edges in R with cliques.
           The minimum number of cliques to cover the edges of a graph H is at least the size of the maximum matching? No.
           For a graph H, the clique partition number is at least the chromatic number of the complement?
           Actually, for small k (10, 20, 30), we can just compute a greedy clique partition for R.
           But wait, R is a subgraph of a complete graph.
           We can simply take each edge in R as a 2-clique?
           That would use |R| cliques.
           Or we can find a large clique in R and remove its edges, repeat.
           
           Given k is small (max 30), we can afford to do a simple optimization for the remaining edges R.
           However, we must ensure that the cliques in P_C are valid cliques in G.
           Since they are subsets of C, they are always valid.
           
           So the plan:
           1. For each i in I, let N_i = N(i).
           2. If N_i is empty, skip i.
           3. Otherwise, create one clique K_i = [i] ∪ N_i.
           4. Mark all edges within N_i as covered.
           5. After processing all i, find all edges in C that are NOT covered.
           6. Partition these remaining edges into cliques.
              Since k <= 30, we can use a greedy approach:
              - While remaining edges exist:
                - Find a maximal clique in the remaining graph.
                - Add it to P_C.
                - Remove its edges from remaining graph.
              
           This should significantly reduce the count compared to using 2-cliques for all cross edges.
           
           Let's verify the "exactly one" constraint again.
           P_I cliques cover:
           - Edges {i, c} for c in N_i.
           - Edges {c1, c2} for c1, c2 in N_i.
           P_C cliques cover:
           - Edges {c1, c2} not covered by any P_I clique.
           
           Is it possible that an edge {c1, c2} is covered by TWO different P_I cliques?
           Yes, if c1, c2 are in N_i and also in N_j for i != j.
           Then {c1, c2} is covered by K_i and K_j.
           This violates "exactly one".
           
           So we cannot just take N_i as a single block for all i independently.
           We must ensure that the sets of E_C edges covered by P_I are DISJOINT.
           
           This is a constraint satisfaction problem.
           For each i, we choose a partition of N_i into sets S_i1, S_i2, ...
           The union of E(S_ij) over all i, j must be a subset of E_C, and these edge sets must be disjoint.
           
           To minimize cliques, we want to minimize sum (number of parts for i) + (cliques for remaining E_C).
           
           Simple heuristic:
           Process i's one by one.
           Maintain a set of "available" edges in C (initially all E_C).
           For each i:
             Find a subset S ⊆ N_i such that E(S) ⊆ available_edges.
             We want S to be as large as possible to save cliques.
             If we find such S, create clique [i] ∪ S.
             Remove E(S) from available_edges.
             Remove S from N_i (conceptually, the remaining neighbors of i must be covered by other cliques? No, i is done?
             Wait, if we don't cover all N_i in one clique, we need more cliques for i.
             But if we cover all N_i in one clique, we use 1 clique for i.
             If we cover part of N_i, we use 1 clique for that part, and need another clique for the rest?
             Yes.
             
             So for each i, we can create multiple cliques.
             Greedy:
             While N_i_remaining is not empty:
               Find the largest clique S in the graph (N_i_remaining, available_edges ∩ E(N_i_remaining)).
               Actually, we just need S ⊆ N_i_remaining such that all edges in S are in available_edges.
               This means S must be a clique in the "available" graph.
               Pick the largest such S.
               Create clique [i] ∪ S.
               Remove E(S) from available_edges.
               Remove S from N_i_remaining.
             
           After all i's are processed, partition the remaining available_edges into cliques (for P_C).
           
           This greedy approach ensures disjointness of covered E_C edges.
           It minimizes P_I cliques locally.
           It leaves a residual graph for P_C.
           
           Implementation details:
           - Represent available_edges as a set of frozensets or bitmasks.
           - k is up to 30, so bitmask of 30 bits fits in an int.
           - For each i, N_i is a bitmask.
           - available is a bitmask of edges? No, edges are pairs.
           - Better: available is a set of (u, v) tuples with u < v.
           - To check if S is a clique in available, we need all pairs in S to be in available.
           
           Since k is small, we can just use sets of integers for vertices.
           
           Let's code this.
    """
    
    start_time = time.time()
    deadline = start_time + 100  # Safety margin

    def solve(n, k, p, seed):
        rng = Random(seed)
        
        # Generate adjacency for cross edges
        # adj[i] is a set of core vertices connected to i
        adj = [set() for _ in range(n)]
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj[v].add(u)
        
        # Initialize available edges in C
        # Edge represented as (u, v) with u < v
        available = set()
        for u in range(k):
            for v in range(u + 1, k):
                available.add((u, v))
        
        cliques = []
        
        # Process each i in I
        for i in range(k, n):
            N_i = set(adj[i])
            while N_i:
                if time.time() > deadline:
                    # Fallback to simple 2-cliques if timeout
                    # But we need to finish the current i and others quickly
                    # Just break and handle rest as 2-cliques
                    break
                
                # Find largest clique S in N_i such that all edges in S are in available
                # Greedy: try to add vertices one by one
                S = set()
                # Sort N_i for determinism? Or just iterate
                # To find a maximal clique, we can use a simple greedy:
                # Iterate through vertices in N_i, add if it forms a clique with current S
                # This finds a maximal clique, not necessarily maximum, but good enough
                for v in sorted(N_i):
                    # Check if v can be added to S
                    # v must be connected to all in S via available edges
                    can_add = True
                    for u in S:
                        edge = (min(u, v), max(u, v))
                        if edge not in available:
                            can_add = False
                            break
                    if can_add:
                        S.add(v)
                
                if not S:
                    # Should not happen if available has edges, but if N_i has isolated nodes in available graph?
                    # If v has no available edges to anyone, S remains empty?
                    # If S is empty, we pick one vertex v.
                    # Edge set of single vertex is empty, so it's always valid.
                    v = next(iter(N_i))
                    S = {v}
                
                # Create clique [i] ∪ S
                clique = [i] + list(S)
                cliques.append(clique)
                
                # Remove edges in S from available
                for u in S:
                    for v in S:
                        if u < v:
                            available.discard((u, v))
                
                # Remove S from N_i
                N_i.difference_update(S)
        
        # If timeout occurred, handle remaining N_i?
        # The loop above might have broken early.
        # We need to ensure all cross edges are covered.
        # If we broke, we didn't cover all edges for some i's.
        # We must cover them with 2-cliques.
        
        # Re-scan to find uncovered cross edges?
        # Better: track covered cross edges?
        # Actually, if we broke out of the while loop for an i, we left N_i non-empty.
        # We need to cover those edges.
        # Let's just iterate all i again and for any edge not covered, add 2-clique.
        # How to know if covered?
        # We can maintain a set of covered cross edges.
        
        # Let's refine the loop to track coverage.
        
        covered_cross = set()
        
        # Redo the logic with tracking
        # Actually, the previous loop added cliques.
        # If it broke, some edges in N_i are not covered by the cliques created for i in that loop.
        # Also, edges for i's not processed (if outer loop broke?) No, outer loop didn't break, only inner.
        
        # Let's restart the solve function with a cleaner implementation that tracks state properly.
        
        return _solve_safe(n, k, p, seed, deadline)

    def _solve_safe(n, k, p, seed, deadline):
        rng = Random(seed)
        adj = [set() for _ in range(n)]
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj[v].add(u)
        
        available = set((u, v) for u in range(k) for v in range(u + 1, k))
        covered_cross = set()
        cliques = []
        
        for i in range(k, n):
            N_i = set(adj[i])
            while N_i:
                if time.time() > deadline:
                    break
                
                # Find maximal clique S in N_i using available edges
                S = set()
                for v in sorted(N_i):
                    can_add = True
                    for u in S:
                        edge = (min(u, v), max(u, v))
                        if edge not in available:
                            can_add = False
                            break
                    if can_add:
                        S.add(v)
                
                if not S:
                    v = next(iter(N_i))
                    S = {v}
                
                cliques.append([i] + list(S))
                
                # Mark cross edges as covered
                for c in S:
                    covered_cross.add((c, i))
                
                # Remove core edges from available
                for u in S:
                    for v in S:
                        if u < v:
                            available.discard((u, v))
                
                N_i.difference_update(S)
            
            # If broke out of while due to timeout, cover remaining N_i with 2-cliques
            if N_i:
                for c in N_i:
                    if (c, i) not in covered_cross:
                        cliques.append([c, i])
                        covered_cross.add((c, i))
        
        # If outer loop finished but inner broke for some, we handled it.
        # What if outer loop itself is slow? n is small (90), k is small (30).
        # Worst case: 90 * 30 iterations. Very fast.
        
        # Now cover remaining core edges
        # available contains edges not covered by P_I.
        # We need to partition available into cliques.
        # Greedy clique partition
        remaining_edges = set(available)
        while remaining_edges:
            if time.time() > deadline:
                # Fallback: use 2-cliques for all remaining
                for e in remaining_edges:
                    cliques.append([e[0], e[1]])
                break
            
            # Find a maximal clique in the graph defined by remaining_edges
            # Vertices are 0..k-1
            # Build adjacency from remaining_edges
            adj_rem = [set() for _ in range(k)]
            for u, v in remaining_edges:
                adj_rem[u].add(v)
                adj_rem[v].add(u)
            
            # Greedy: pick vertex with highest degree, add neighbors that form clique
            # Simpler: pick first vertex, add all neighbors, then filter
            # Even simpler: just take any edge, then try to extend?
            # Since k is small, we can just find a clique.
            # Start with empty set, add vertices greedily.
            S = set()
            for v in range(k):
                can_add = True
                for u in S:
                    if v not in adj_rem[u]:
                        can_add = False
                        break
                if can_add:
                    S.add(v)
            
            if not S:
                # Should not happen if remaining_edges not empty
                break
            
            if len(S) < 2:
                # Single vertex covers no edges. Pick an edge.
                u, v = next(iter(remaining_edges))
                S = {u, v}
            
            cliques.append(list(S))
            
            # Remove edges in S from remaining_edges
            for u in S:
                for v in S:
                    if u < v:
                        remaining_edges.discard((u, v))
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(_solve_safe(*spec, deadline))

    return results
