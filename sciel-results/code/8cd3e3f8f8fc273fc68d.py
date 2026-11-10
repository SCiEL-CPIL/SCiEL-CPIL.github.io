import time
from random import Random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        # 1. Separate edges into C-edges and Cross-edges
        c_edges = []
        cross_edges = []
        
        # Build adjacency for C vertices (0 to k-1)
        c_adj = [set() for _ in range(k)]
        
        for (u, v) in edges:
            if u < k and v < k:
                c_edges.append((u, v))
                c_adj[u].add(v)
                c_adj[v].add(u)
            else:
                # One is in C, one in I
                if u < k:
                    c, i = u, v
                else:
                    c, i = v, u
                cross_edges.append((c, i))
        
        # 2. Identify active independent vertices
        active_i = sorted(list(set(i for _, i in cross_edges)))
        
        # 3. Generate Candidate Cliques
        # A clique in a split graph is either:
        # A) A subset of C (clique in G[C])
        # B) {i} U S where i in I and S is a clique in G[C] subset of N(i)
        
        # We want to cover all edges exactly once.
        # This is an Exact Cover problem (or Set Cover if we allow over-covering? No, "exactly one").
        # However, finding a minimal exact cover is NP-hard.
        # But for k=30, the number of C-edges is ~435.
        # We can formulate this as:
        # Each edge e must be covered by exactly one clique.
        # 
        # Strategy: 
        # 1. Precompute all "maximal" candidate cliques of type B (centered at i in I).
        #    Why maximal? Any sub-clique of a valid clique is also valid, but using larger cliques
        #    covers more edges per clique, which is generally better for minimization.
        #    However, we must be careful: if we use a maximal clique, we might block other cliques.
        #    Let's generate ALL cliques of type B? That's too many (2^k).
        #    Instead, let's generate "maximal cliques" in the subgraph induced by N(i) for each i.
        #    Actually, we need to consider that a C-edge (u,v) might be covered by a clique {i} U S 
        #    only if u,v in S. 
        #    
        #    Let's try a different angle: 
        #    The C-edges form a graph G_C. 
        #    The cross edges form a bipartite graph G_cross.
        #    
        #    Observation: If we choose a partition of the C-edges into cliques (which is just a clique cover of G_C),
        #    and a partition of the cross-edges into cliques of type {i} U S, we have a valid partition.
        #    BUT, a cross-clique {i} U S also covers C-edges inside S.
        #    So the C-edges covered by cross-cliques must be removed from the set of C-edges to be covered by pure C-cliques.
        #    
        #    Let's use a Greedy Set Cover approach with a twist:
        #    We have a set of "items" to cover: All C-edges and All Cross-edges.
        #    We have a set of "sets" (cliques):
        #       - For each i in I, and each clique S in G_C[N(i)], the clique {i} U S covers:
        #           - Cross edges (i, s) for s in S
        #           - C-edges (u, v) for u, v in S
        #       - For each clique T in G_C, the clique T covers:
        #           - C-edges (u, v) for u, v in T
        #    
        #    We want to select a collection of these sets such that every item is covered exactly once.
        #    This is the "Exact Cover by Cliques" problem.
    
        #    Given the constraints (n=90, k=30), we can't do full exact cover search.
        #    We will use a **Greedy Algorithm with Conflict Resolution**:
        #    1. Start with all edges uncovered.
        #    2. Iteratively pick the "best" clique candidate.
        #       - Best is defined by: (Number of newly covered edges) / (Size of clique) ? 
        #       - Or simply: Maximize number of newly covered edges.
        #       - Tie-break: Prefer cliques that cover C-edges (since C-edges are "harder" to cover efficiently later?).
        #       - Actually, covering C-edges with cross-cliques is "free" in terms of cross-edge count, 
        #         but it reduces the number of C-cliques needed.
        #    
        #    Let's refine the candidate generation.
        #    Generating all cliques in G_C is exponential. 
        #    However, we only need cliques that are "maximal" in some sense or "useful".
        #    
        #    Alternative Structural Insight:
        #    For a fixed i, the cross edges incident to i are (i, u) for u in N(i).
        #    If we form a clique {i} U S, S must be a clique in G_C.
        #    To minimize the number of cliques, we want S to be as large as possible?
        #    Not necessarily. If S is large, it covers many C-edges. 
        #    If those C-edges could be covered more efficiently by other means, maybe smaller S is better.
        #    But C-edges can ONLY be covered by:
        #       a) Pure C-cliques
        #       b) Cross-cliques {i} U S where u,v in S.
        #    
        #    Let's try this specific heuristic:
        #    1. Identify all maximal cliques in G_C. Let this set be MC.
        #    2. For each i in I, identify all maximal cliques in G_C[N(i)]. Let this set be MC_i.
        #    3. Candidate set = MC (as pure C-cliques) U Union(MC_i) (as cross-cliques {i} U S).
        #    4. Greedy:
        #       - While uncovered edges remain:
        #           - For each candidate clique K, calculate `gain` = number of edges in K that are currently uncovered.
        #           - If gain == 0, skip.
        #           - Pick K with max gain.
        #           - Add K to solution.
        #           - Mark all edges in K as covered.
        #           - If an edge was already covered by a previous clique, we have a CONFLICT.
        #             This means our greedy choice led to an invalid solution (overlap).
        #             To handle "exactly one", we must ensure we don't pick overlapping cliques.
        #             This turns it into a Packing Problem, not just Cover.
        #             
        #    Since "Exactly One" is a hard constraint, simple greedy fails if it overlaps.
        #    We need a **Greedy Independent Set in the Conflict Graph** approach?
        #    Or, we can use a **Backtracking Search** with strong pruning for the small C-edges?
        #    k=30 is too big for brute force.
        
        #    Let's go back to the "Edge Coloring" idea but applied to the C-edges specifically.
        #    
        #    Revised Strategy:
        #    1. Partition the Cross-Edges first using a greedy algorithm that prioritizes forming large cliques {i} U S.
        #       - For each i, we want to partition N(i) into cliques S_1, S_2, ...
        #       - This covers all cross edges incident to i.
        #       - The union of all S_j for all i covers some set of C-edges. Let this set be E_covered_by_cross.
        #    2. The remaining C-edges E_C \ E_covered_by_cross must be covered by pure C-cliques.
        #       - We need to find a minimum clique cover for the remaining C-edges.
        #       - Note: The remaining C-edges form a subgraph of G_C.
        #       - Minimum clique cover in a general graph is NP-hard, but G_C is a random graph (or subgraph).
        #       - We can use a greedy coloring of the complement graph for the remaining edges?
        #         Clique cover of G is Coloring of G_bar.
        #         So we need to color the complement of the remaining C-edge subgraph.
        
        #    How to optimize Step 1?
        #    When partitioning N(i) into cliques, we want to choose the cliques S_j such that they cover as many C-edges as possible (to minimize the load on Step 2).
        #    This is a "Maximum Weight Clique Partition" where weight is the number of C-edges covered.
        #    
        #    Let's implement a Greedy for Step 1:
        #    For each i in I:
        #       Let N_i = N(i).
        #       We want to partition N_i into cliques.
        #       Greedy: 
        #          - Start with empty list of cliques for i.
        #          - While N_i is not empty:
        #              - Find a maximal clique S in G_C[N_i] that maximizes the number of *uncovered* C-edges within S.
        #                (Note: "Uncovered" means not yet covered by any previous clique for any i).
        #              - Add S to the partition for i.
        #              - Mark all C-edges within S as "covered by cross".
        #              - Remove S from N_i.
        #    
        #    Wait, the order of processing i matters. 
        #    If we process i1 first, it might cover C-edges that i2 could have covered.
        #    Since all cross edges must be covered, and C-edges are "bonus" coverage, 
        #    we should prioritize i's that have high density of C-edges in their neighborhoods.
        
        #    Let's try:
        #    1. Precompute for each i, the list of neighbors in C.
        #    2. Sort i's by the number of C-edges in their neighborhood (descending).
        #    3. For each i in sorted order:
        #       - Partition N(i) into cliques greedily.
        #       - At each step, pick the clique S in remaining N(i) that covers the most *currently uncovered* C-edges.
        #       - If tie, pick the largest S.
        #       - Add {i} U S to solution.
        #       - Update covered C-edges.
        #    4. After all i's are processed, collect remaining uncovered C-edges.
        #    5. Find a minimum clique cover for the remaining C-edges.
        #       - Use a simple greedy coloring on the complement of the remaining graph.
        #       - Or, since it's a subgraph of G_C, we can just greedily pick maximal cliques in the remaining graph.
        
        #    This is a different structural approach: **Decoupled Optimization**.
        #    It separates the cross-edge partitioning (driven by C-edge coverage gain) from the final C-edge cleanup.

        # Implementation:
        
        # Precompute C-edge set for O(1) lookup
        c_edge_set = set(c_edges)
        
        # Initialize state
        covered_c_edges = set()
        cliques = []
        
        # For each i, store its neighbors in C
        i_neighbors = {i: set() for i in active_i}
        for c, i in cross_edges:
            i_neighbors[i].add(c)
            
        # Sort active_i by potential C-edge coverage
        # Potential = number of C-edges within N(i)
        def count_c_edges_in_set(s):
            cnt = 0
            lst = list(s)
            for u_idx in range(len(lst)):
                for v_idx in range(u_idx + 1, len(lst)):
                    u, v = lst[u_idx], lst[v_idx]
                    if (min(u,v), max(u,v)) in c_edge_set:
                        cnt += 1
            return cnt

        # Precompute counts for sorting
        i_potential = {}
        for i in active_i:
            i_potential[i] = count_c_edges_in_set(i_neighbors[i])
            
        sorted_i = sorted(active_i, key=lambda x: (-i_potential[x], x))
        
        # Greedy Partition for Cross Edges
        for i in sorted_i:
            remaining_neighbors = set(i_neighbors[i])
            
            while remaining_neighbors:
                # Find best clique S in G_C[remaining_neighbors]
                # "Best" = max new C-edges covered.
                # Since k is small (<=30), we can try to find a maximal clique greedily.
                # But finding the MAXIMAL clique with max weight is hard.
                # Let's just find a maximal clique in the induced subgraph.
                # To bias towards covering uncovered C-edges, we can order vertices by degree in uncovered C-edges.
                
                # Get vertices in remaining_neighbors
                verts = list(remaining_neighbors)
                
                # Order by number of uncovered C-edges incident to them within remaining_neighbors
                def vertex_score(v):
                    score = 0
                    for u in remaining_neighbors:
                        if u == v: continue
                        e = (min(u,v), max(u,v))
                        if e in c_edge_set and e not in covered_c_edges:
                            score += 1
                    return score

                verts_sorted = sorted(verts, key=vertex_score, reverse=True)
                
                # Greedy Maximal Clique
                S = set()
                for v in verts_sorted:
                    # Check if v is connected to all in S
                    ok = True
                    for u in S:
                        e = (min(u,v), max(u,v))
                        if e not in c_edge_set:
                            ok = False
                            break
                    if ok:
                        S.add(v)
                
                if not S:
                    # Should not happen if remaining_neighbors is not empty, unless no edges?
                    # If no edges in C, any single vertex is a clique.
                    S = {verts_sorted[0]}
                
                # Add clique {i} U S
                clique = [i] + sorted(list(S))
                cliques.append(clique)
                
                # Update covered C-edges
                s_list = list(S)
                for u_idx in range(len(s_list)):
                    for v_idx in range(u_idx + 1, len(s_list)):
                        u, v = s_list[u_idx], s_list[v_idx]
                        e = (min(u,v), max(u,v))
                        if e in c_edge_set:
                            covered_c_edges.add(e)
                            
                # Remove S from remaining
                remaining_neighbors -= S
                
        # Handle Remaining C-Edges
        remaining_c_edges = [e for e in c_edges if e not in covered_c_edges]
        
        if remaining_c_edges:
            # We need to cover these edges with cliques from C.
            # Greedy: Pick maximal cliques in the graph formed by remaining_c_edges.
            # Actually, we can just use the same logic: 
            # While remaining_c_edges:
            #   Pick a vertex with max degree in remaining_c_edges.
            #   Build maximal clique.
            
            # To be efficient, let's just use a simple greedy coloring of the complement?
            # Or just greedy maximal clique extraction.
            
            rem_adj = {u: set() for u in range(k)}
            for u, v in remaining_c_edges:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
                
            while remaining_c_edges:
                # Pick vertex with max degree
                max_deg = -1
                v_max = -1
                for u in rem_adj:
                    if len(rem_adj[u]) > max_deg:
                        max_deg = len(rem_adj[u])
                        v_max = u
                        
                if v_max == -1:
                    break
                    
                # Build maximal clique starting with v_max
                S = {v_max}
                # Candidates: neighbors of v_max in rem_adj
                # Sort by degree?
                candidates = sorted(list(rem_adj[v_max]), key=lambda x: len(rem_adj[x]), reverse=True)
                
                for c in candidates:
                    # Check if c connected to all in S
                    ok = True
                    for u in S:
                        if u not in rem_adj[c]:
                            ok = False
                            break
                    if ok:
                        S.add(c)
                        
                if len(S) >= 2:
                    cliques.append(sorted(list(S)))
                    
                # Remove all edges within S from remaining_c_edges and rem_adj
                s_list = list(S)
                for u_idx in range(len(s_list)):
                    for v_idx in range(u_idx + 1, len(s_list)):
                        u, v = s_list[u_idx], s_list[v_idx]
                        if u in rem_adj and v in rem_adj[u]:
                            rem_adj[u].remove(v)
                            rem_adj[v].remove(u)
                            # Remove from list
                            remaining_c_edges = [e for e in remaining_c_edges if e != (u,v) and e != (v,u)]
                # Also remove edges incident to S that are not in S? No, we only cover edges within S.
                # But we must remove S from consideration? 
                # The remaining edges are those NOT in S.
                # The loop `while remaining_c_edges` will pick the next one.
                # However, `rem_adj` still has edges from S to outside.
                # We should clear S from rem_adj to avoid re-picking?
                # If we don't clear, we might pick a vertex in S again.
                # But its degree in rem_adj might be high.
                # Let's remove S from rem_adj entirely? No, we need the edges to outside for future cliques.
                # But we should mark S as "done" so we don't start a new clique with them?
                # Actually, if we form a clique S, the edges WITHIN S are covered.
                # The edges FROM S TO OUTSIDE are still uncovered.
                # So we should NOT remove S from the graph.
                # But we must ensure we don't pick a clique that overlaps with S?
                # The greedy maximal clique extraction on `rem_adj` will naturally handle this?
                # No. If we pick v_max again, we might form a clique that shares vertices with S.
                # That's allowed as long as EDGES are not shared.
                # But if we form a new clique T sharing vertex v with S, 
                # the edge (v, w) in T is distinct from edges in S.
                # So it's valid.
                # However, we must ensure we don't cover an edge twice.
                # Since we remove edges within S from `remaining_c_edges`, and we only pick edges from `remaining_c_edges`,
                # we are safe.
                # But `rem_adj` is used to find candidates.
                # If we don't update `rem_adj` to reflect that edges within S are gone, 
                # the degree counts are slightly off, but the clique property check `if u not in rem_adj[c]` 
                # relies on `rem_adj` containing the edge.
                # We DID remove edges within S from `rem_adj`.
                # So the clique property check is correct.
                # The only issue is efficiency.
                
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 15:
            # Emergency fallback
            edges = instance(*spec)
            results.append([[int(e[0]), int(e[1])] for e in edges])
            continue
        
        edges = instance(*spec)
        cliques = partition_split_graph(spec[0], spec[1], edges)
        
        # Validate and convert types
        final_cliques = []
        for c in cliques:
            c_ints = list(set(int(x) for x in c))
            if len(c_ints) >= 2:
                final_cliques.append(sorted(c_ints))
        
        results.append(final_cliques)
    
    return results
# EVOLVE-BLOCK-END
