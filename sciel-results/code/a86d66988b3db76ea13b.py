"""Clique partition of split graphs using Greedy Maximal Clique Packing."""
import time
from random import Random

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
        # Build adjacency
        adj = [set() for _ in range(n)]
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Remaining edges to cover
        remaining_edges = set(edges)
        
        # We want to find a collection of cliques that partition the edges.
        # Strategy: Greedy Packing.
        # 1. Identify all maximal cliques that contain at least one vertex from I (independent set).
        #    In a split graph, any clique is either a subset of C (clique vertices) or 
        #    {i} U S where i in I and S is a clique in C (which is all of C, so S is any subset of C).
        #    Actually, since C is a clique, any subset of C is a clique.
        #    So a clique containing i is {i} U S where S is any subset of N(i) ∩ C.
        #    To maximize efficiency, we should pick S to be as large as possible while 
        #    minimizing the "waste" of covering internal edges that might be covered by others.
        
        # However, the "exactly one" constraint is strict. 
        # If we pick a clique {i} U S, we remove all edges in that clique from remaining_edges.
        # If an internal edge (u,v) is in S, it is removed.
        # If later another clique {j} U T contains u and v, that edge is already gone, so no conflict.
        # The only conflict is if we *don't* cover an edge and it remains.
        # So the problem reduces to: Find a set of cliques such that every edge is in exactly one.
        # This is equivalent to finding a clique partition.
        
        # Greedy heuristic:
        # While remaining_edges is not empty:
        #   Find a clique K in the remaining graph that maximizes |E(K) ∩ remaining_edges| / cost(K)
        #   Where cost(K) = 1 if K is an edge-clique, else 1 (we want to minimize count).
        #   Actually, we just want to maximize the number of edges covered per clique.
        #   So, pick the clique that covers the most remaining edges.
        
        # Since n is small (max 90), we can enumerate candidate cliques.
        # Candidate cliques:
        # 1. For each i in I, the maximal clique {i} U N(i). 
        #    But we might want a subset.
        #    However, if we take the full N(i), we cover all cross edges for i and all internal edges in N(i).
        #    Is it ever better to take a subset? 
        #    If we take a subset, we leave some cross edges for i uncovered, which would require another clique for i?
        #    No, i can only be in cliques of the form {i} U S. 
        #    If we split i's edges into multiple cliques {i} U S1 and {i} U S2, we use 2 cliques instead of 1.
        #    So for a fixed i, it is always better to use ONE clique {i} U N(i) than multiple.
        #    BUT, the internal edges in N(i) might be "wasted" if they are already covered by a clique {j} U T.
        #    Wait, if (u,v) is in N(i), it is covered by {i} U N(i). If it was also covered by {j} U T, that's a conflict.
        #    So we must ensure that internal edges are not covered by multiple cliques.
        #    This implies that the sets S_i for different i's must be such that their internal edge sets are disjoint?
        #    No, they just can't both contain the same edge.
        #    If (u,v) is in S_i, it is covered by clique i. It cannot be covered by clique j.
        #    So (u,v) cannot be in S_j.
        #    This means the "internal edge coverage" of the chosen S_i's must be disjoint.
        
        # Revised Strategy:
        # 1. Start with S_i = N(i) for all i in I.
        # 2. Check for conflicts: internal edges covered by multiple S_i.
        # 3. If conflict, resolve by removing vertices from S_i.
        # This is what the current program does.
        
        # Let's try a different angle: **Exact Cover via Dancing Links (Algorithm X)** or **ILP**.
        # Given the small size, ILP might be overkill but Greedy Set Cover on the *internal edges* is key.
        
        # Let's stick to the Greedy Packing on the *graph* but be smart about it.
        # We will iteratively pick cliques.
        # To avoid conflicts, we can process vertices in I one by one.
        # For each i, we want to pick S_i such that:
        # a) S_i ⊆ N(i)
        # b) The internal edges in S_i are NOT covered by any previously chosen S_j.
        # c) S_i is as large as possible to cover cross edges.
        
        # This is a greedy sequential assignment.
        # Order of I vertices: Process those with largest degree first? Or random?
        # Let's try processing i in order of decreasing |N(i)|.
        
        I_vertices = list(range(k, n))
        I_vertices.sort(key=lambda x: len(adj[x]), reverse=True)
        
        chosen_cliques = []
        covered_internal_edges = set()
        
        for i in I_vertices:
            # We need to choose S_i ⊆ N(i)
            # Constraint: No edge (u,v) with u,v in S_i can be in covered_internal_edges.
            # This means S_i must be an independent set in the graph of "conflicting internal edges"?
            # No, S_i is a subset of C. C is a clique.
            # The condition is: For all u,v in S_i, (u,v) not in covered_internal_edges.
            # This means the subgraph induced by S_i on C must not contain any edge that is already covered.
            # Let G_conf be the graph on C where (u,v) is an edge if (u,v) is in covered_internal_edges.
            # Then S_i must be an independent set in G_conf?
            # No, if (u,v) is in G_conf, we cannot have both u and v in S_i.
            # So yes, S_i must be an independent set in G_conf.
            # But S_i must also be a clique in C (which is always true) and a subset of N(i).
            
            # So we need to find a maximal independent set in G_conf restricted to N(i)?
            # Wait, we want S_i to be as LARGE as possible to cover cross edges.
            # But if we pick a large S_i, we cover many internal edges, which restricts future S_j.
            # Is it better to pick a large S_i now, or save internal edges for later?
            # Cross edges (i, u) MUST be covered by a clique containing i.
            # Since i is only in cliques of the form {i} U S, ALL cross edges of i must be covered by the SINGLE clique for i?
            # No, we can have multiple cliques for i: {i} U S1, {i} U S2.
            # But that increases the clique count.
            # So for a fixed i, we should use exactly ONE clique {i} U S_i.
            # And S_i MUST contain all N(i) to cover all cross edges of i.
            # If S_i = N(i), then we cover all cross edges of i.
            # But this forces S_i = N(i).
            # If we are forced to use S_i = N(i), then we have NO CHOICE.
            # The conflicts must be resolved by... wait.
            # If S_i = N(i) and S_j = N(j), and they share an internal edge, it's a conflict.
            # Can we split i into two cliques?
            # If we split i into {i} U S1 and {i} U S2, we cover cross edges of i.
            # This uses 2 cliques instead of 1.
            # But it allows S1 and S2 to be smaller, potentially avoiding internal edge conflicts.
            
            # So the trade-off is: 
            # Option A: 1 clique for i, S_i = N(i). Risk: Internal conflicts.
            # Option B: Multiple cliques for i. Cost: More cliques. Benefit: Smaller S's, fewer internal edges covered per clique, easier to avoid conflicts.
            
            # The current program uses Option A (mostly) and resolves conflicts by splitting (removing vertices from S_i and creating new edge-cliques).
            # Note: The current program creates [i, u] cliques. These are 2-cliques.
            # This is effectively Option B, but very fragmented.
            
            # Better Strategy: **Minimum Set Cover on Cross-Edges with Internal Edge Penalty**.
            # Let's use a Greedy Set Cover where the "universe" is the set of all edges.
            # "Sets" are candidate cliques.
            # We prefer cliques that cover many edges and have low "internal edge cost".
            
            # Actually, let's just use the Greedy Maximal Clique Extraction from Inspiration 1, which scored 1.33.
            # The current program scored 1.19.
            # Inspiration 1: "Greedy max-clique extraction".
            # Let's implement a robust version of that.
            
            # 1. Find all maximal cliques in the graph.
            # 2. Sort them by size (descending).
            # 3. Iterate: Pick the largest clique that contains at least one uncovered edge.
            #    Remove all edges in that clique from the uncovered set.
            #    Add the clique to the result.
            # 4. Repeat until no edges left.
            # 5. Any remaining edges (should be none if we cover all) are added as edge-cliques.
            
            # Why did this work better? 
            # Because it doesn't force a specific structure per vertex i.
            # It just finds big cliques.
            
            # Let's implement this.
            
            pass # Placeholder, we will implement the full logic below

        # Re-implementation of the whole function with the new strategy
        remaining_edges = set(edges)
        result_cliques = []
        
        # Precompute all maximal cliques?
        # For n=90, k=30, this might be expensive if done naively.
        # But split graphs have a simple structure.
        # Maximal cliques are:
        # 1. C itself (if no I vertices connected to all C? No, C is a clique).
        # 2. For each i in I, {i} U N(i).
        # 3. Subsets of C? No, if a subset of C is maximal, it must be C (since C is a clique).
        #    Wait, if we remove edges, the maximal cliques change.
        
        # Dynamic Greedy:
        while remaining_edges:
            if time.time() > deadline - 5:
                # Fallback: add all remaining as edge cliques
                for e in remaining_edges:
                    result_cliques.append([e[0], e[1]])
                break
            
            # Find the clique in the CURRENT remaining graph that covers the most edges.
            # Candidate cliques to check:
            # 1. For each i in I, the clique {i} U (N(i) ∩ V_remaining) where V_remaining are vertices incident to remaining edges?
            #    Actually, the clique must be a clique in the original graph.
            #    So {i} U S where S ⊆ N(i).
            #    To maximize covered edges, we should take S = N(i) ∩ (vertices that have remaining edges).
            #    But we must ensure it's a clique in the remaining graph?
            #    No, the clique is a set of vertices. The edges it covers are those in remaining_edges.
            #    So we just pick S = N(i). The covered edges are (i, s) for s in N(i) and (u, v) for u, v in N(i).
            #    We count how many of these are in remaining_edges.
            
            best_clique = None
            best_count = 0
            
            # Check cliques of the form {i} U N(i)
            for i in range(k, n):
                # Only consider if i has any remaining edges
                if not any(e[0] == i or e[1] == i for e in remaining_edges):
                    continue
                
                S = adj[i] & set(range(k)) # Neighbors in C
                clique = {i} | S
                
                # Count edges in this clique that are in remaining_edges
                count = 0
                # Cross edges
                for s in S:
                    e = (min(i, s), max(i, s))
                    if e in remaining_edges:
                        count += 1
                # Internal edges
                S_list = list(S)
                for idx_u in range(len(S_list)):
                    u = S_list[idx_u]
                    for idx_v in range(idx_u + 1, len(S_list)):
                        v = S_list[idx_v]
                        e = (min(u, v), max(u, v))
                        if e in remaining_edges:
                            count += 1
                
                if count > best_count:
                    best_count = count
                    best_clique = clique
            
            # Also check pure C cliques? 
            # If there are internal edges left, they must be covered by a clique containing C vertices.
            # The only cliques containing C vertices are subsets of C.
            # The maximal one is C itself.
            # So check C.
            S_C = set(range(k))
            clique_C = S_C
            count_C = 0
            S_C_list = list(S_C)
            for idx_u in range(len(S_C_list)):
                u = S_C_list[idx_u]
                for idx_v in range(idx_u + 1, len(S_C_list)):
                    v = S_C_list[idx_v]
                    e = (min(u, v), max(u, v))
                    if e in remaining_edges:
                        count_C += 1
            
            if count_C > best_count:
                best_count = count_C
                best_clique = clique_C
            
            if best_clique is None or best_count == 0:
                # No clique found? Should not happen if remaining_edges is not empty.
                # Fallback: pick an edge
                e = next(iter(remaining_edges))
                result_cliques.append([e[0], e[1]])
                remaining_edges.discard(e)
                continue
            
            # Add the best clique
            clique_list = sorted(list(best_clique))
            result_cliques.append(clique_list)
            
            # Remove all edges in this clique from remaining_edges
            to_remove = []
            for e in remaining_edges:
                if e[0] in best_clique and e[1] in best_clique:
                    to_remove.append(e)
            
            for e in to_remove:
                remaining_edges.discard(e)
        
        # Convert to lists of Python ints
        final_cliques = [[int(x) for x in c] for c in result_cliques]
        
        return final_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 15:
            # Emergency fallback
            edges = instance(*spec)
            results.append([[int(e[0]), int(e[1])] for e in edges])
            continue
        
        edges = instance(*spec)
        cliques = partition_split_graph(spec[0], spec[1], edges)
        
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
