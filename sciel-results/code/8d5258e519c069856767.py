"""Clique partition of split graphs using Minimum Weight Bipartite Matching structure."""
import time
from random import Random
import numpy as np
from scipy.optimize import linear_sum_assignment

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
        # Split graph: Vertices 0..k-1 form a clique C. Vertices k..n-1 form an independent set I.
        # Edges are: all within C, and some between C and I.
        
        # Identify cross edges (between C and I)
        cross_edges = []
        for u, v in edges:
            if u < k and v >= k:
                cross_edges.append((u, v))
            elif v < k and u >= k:
                cross_edges.append((v, u))
            
        # We need to cover all cross edges with cliques.
        # A clique in a split graph is of the form {c1, ..., cm, i1, ..., ij}
        # where c's are from C, i's are from I.
        # For i to be in the clique, it must be connected to ALL c's in the clique.
        # The cost of adding a clique with m clique-vertices and j independent-vertices
        # is 1 (the clique itself), but we are minimizing the NUMBER of cliques.
        # Wait, the objective is to minimize the total number of cliques.
        
        # Let's re-evaluate the objective.
        # Total edges = E_C (internal to C) + E_cross (between C and I).
        # E_C must be covered. The maximum clique is C itself (size k).
        # If we use C as a clique, it covers all E_C edges and all cross edges incident to vertices in C?
        # No, a clique {c1..ck, i} covers edges within {c1..ck} and edges between {c1..ck} and {i}.
        # It does NOT cover edges between {c1..ck} and other independent vertices.
        
        # Strategy:
        # 1. The internal edges of C (size k(k-1)/2) must be covered.
        #    The most efficient way to cover edges within C is to use C itself as a clique.
        #    If we use C as a clique, it covers all internal C edges.
        #    It also covers cross edges (c, i) where i is connected to ALL vertices in C.
        #    If i is not connected to all of C, we cannot include i in the clique C.
        #    So, we might need to break C into smaller cliques to accommodate specific independent vertices?
        #    Actually, if we use C as a clique, we have 1 clique covering all C-C edges.
        #    Any independent vertex i that is connected to all of C can be added to this clique for "free" (in terms of increasing clique count, it just enlarges the existing one).
        #    But if i is not connected to all of C, it must be covered by other cliques.
        
        # Let's assume we start with the clique C = {0, ..., k-1}.
        # This covers all edges within C.
        # It also covers cross edges (c, i) for all c in C, if i is included in the clique.
        # But we can only include i in the clique if i is connected to all c in C.
        # Let S_full = {i in I | i connected to all c in C}.
        # We can form a single large clique C_full = C ∪ S_full.
        # This covers all C-C edges, and all edges between C and S_full.
        
        # Now we have remaining cross edges between C and I \ S_full.
        # Let I_rem = I \ S_full.
        # For each i in I_rem, it is not connected to at least one vertex in C.
        # We need to cover edges (c, i) for c in C, i in I_rem.
        # Since i is not connected to all C, we cannot just add i to the main clique.
        # We must form additional cliques.
        # Any clique covering an edge (c, i) must contain c and i.
        # Since I is an independent set, a clique can contain at most one vertex from I.
        # So any clique covering edges incident to i must be of the form {c1, ..., cm, i} where i is connected to all c1..cm.
        # The edges covered are (cj, i) for all j.
        # To minimize the number of such cliques, we want each such clique to cover as many edges (cj, i) as possible.
        # This means for a fixed i, we should find the largest subset of C that is connected to i.
        # Let N(i) be the set of neighbors of i in C.
        # The largest clique containing i is N(i) ∪ {i}.
        # This clique covers all edges (c, i) for c in N(i).
        # Does it cover any other edges? It covers edges within N(i).
        # But edges within C are already covered by the main clique C_full.
        # Wait, the rule is "every edge must lie in exactly one listed clique".
        # If we use C_full to cover C-C edges, we cannot use another clique to cover C-C edges.
        # So, when we form a clique {c1, ..., cm, i}, we must ensure that the edges (ca, cb) for ca, cb in {c1..cm} are NOT covered by this clique if they were already covered by C_full.
        # But the constraint is "every edge lies in exactly one clique".
        # This implies we must partition the edge set.
        # If we use C_full = C ∪ S_full, it covers all edges within C.
        # Then we cannot use any other clique that contains two vertices from C, because that would cover C-C edges again.
        # Therefore, any other clique can contain at most ONE vertex from C.
        # If a clique contains at most one vertex from C and one from I, it is just an edge (c, i).
        # So, for I_rem, we are forced to use edge-cliques (c, i) for every remaining cross edge?
        # That would be very expensive.
        
        # Let's rethink. Can we avoid using the full C as a single clique?
        # Yes. We can partition C into smaller cliques to allow independent vertices to join.
        # Example: k=2, C={0,1}, I={2,3}.
        # Edges: (0,1), (0,2), (1,3).
        # Option A: Clique {0,1}. Covers (0,1). Remaining: (0,2), (1,3). Cliques: {0,2}, {1,3}. Total 3.
        # Option B: Clique {0,2}. Covers (0,2). Remaining: (0,1), (1,3).
        #   Next clique must cover (0,1) or (1,3).
        #   If {0,1}, covers (0,1). Remaining (1,3) -> {1,3}. Total 3.
        #   If {1,3}, covers (1,3). Remaining (0,1) -> {0,1}. Total 3.
        # Option C: Clique {1,3}. Covers (1,3). Remaining (0,1), (0,2).
        #   Next {0,1} covers (0,1). Remaining (0,2) -> {0,2}. Total 3.
        
        # What if we have a star?
        # k=3, C={0,1,2}, I={3,4,5}.
        # 3 connected to 0,1. 4 connected to 1,2. 5 connected to 0,2.
        # Edges in C: (0,1), (1,2), (0,2).
        # Cross: (0,3), (1,3), (1,4), (2,4), (0,5), (2,5).
        # If we use C={0,1,2} as a clique, it covers all C-C edges.
        # 3 is not connected to 2, so cannot join.
        # 4 is not connected to 0, so cannot join.
        # 5 is not connected to 1, so cannot join.
        # So we must cover cross edges with cliques containing at most 1 vertex from C?
        # No, if we use C as a clique, we can't use other C-C edges.
        # So we must use edges (0,3), (1,3), etc. Total 6 cross edges -> 6 cliques. Total 7.
        
        # Alternative: Don't use C as a single clique.
        # Use {0,1,3}. Covers (0,1), (0,3), (1,3).
        # Remaining C-C edges: (1,2), (0,2).
        # Remaining cross: (1,4), (2,4), (0,5), (2,5).
        # Next clique: {1,2,4}. Covers (1,2), (1,4), (2,4).
        # Remaining C-C: (0,2).
        # Remaining cross: (0,5), (2,5).
        # Next clique: {0,2,5}. Covers (0,2), (0,5), (2,5).
        # Total 3 cliques. Much better!
        
        # So the problem is: Partition the edges into cliques, where each clique is a set of vertices from C and at most one from I?
        # No, a clique can have multiple from I? No, I is independent. So at most one from I.
        # So every clique is of the form S ∪ {i} where S ⊆ C and i ∈ I, or S ⊆ C.
        # And S must be a clique in C (which it always is).
        # And i must be connected to all v in S.
        
        # This is equivalent to:
        # We have a set of "jobs" which are the cross edges.
        # Each clique can cover a set of cross edges (v, i) for v in S, provided i is connected to all v in S.
        # It also covers internal edges within S.
        # We must cover all internal edges of C and all cross edges.
        
        # This looks like a Set Cover or Partition problem.
        # Given the small size (k<=30, n<=90), we can use a greedy heuristic that is smarter than the previous one.
        # The previous greedy picked max degree vertex.
        # A better greedy:
        # 1. Start with all edges.
        # 2. Identify "rich" independent vertices (high degree to C).
        # 3. For each i in I (sorted by degree desc), find the largest S ⊆ C such that i is connected to all in S, AND S contains vertices that have "uncovered" edges to other independent vertices or internal edges.
        
        # Actually, there is a known result for split graphs.
        # The minimum clique partition size is related to the independence number of the complement?
        # Or simply:
        # Let's try a greedy that prioritizes covering cross edges with large cliques.
        
        # Algorithm:
        # 1. Build adjacency.
        # 2. While edges remain:
        #    a. Find a clique K that maximizes the number of edges it covers.
        #    b. To find such a K:
        #       - Try K = C (if all C-C edges are uncovered? No, we can't reuse edges).
        #       - We need to track which edges are covered.
        
        # Let's implement a greedy that picks a clique maximizing covered edges.
        # Since k is small, we can iterate over subsets of C? 2^30 is too big.
        # But we can iterate over i in I and find best S for each i.
        # And also consider cliques purely in C.
        
        # Heuristic:
        # 1. If there are uncovered edges in C, we must cover them.
        #    The best way is to use a large subset of C.
        # 2. If we pick a subset S of C to form a clique, we can add any i connected to all of S.
        
        # Let's try this specific greedy:
        # 1. remaining_edges = set of all edges.
        # 2. While remaining_edges:
        #    - Find the "best" clique.
        #    - Candidate cliques:
        #      - For each i in I: Let S_i = {v in C | (v,i) in remaining_edges}.
        #        The clique K_i = S_i ∪ {i} covers all edges (v,i) for v in S_i, and all edges (u,v) for u,v in S_i.
        #        Score = number of edges covered by K_i.
        #      - For subsets of C?
        #        If we just pick S ⊆ C, it covers internal edges.
        #        But we can always add an i if it's connected.
        #        So the "best" clique involving i is determined by S_i.
        #        What if no i is connected to a large S?
        #        Then we should pick a large S ⊆ C to cover internal edges.
        #        Let's say we pick S = C_remaining (vertices in C with uncovered internal edges).
        #        Then we can add any i connected to all of S.
        
        # Let's refine:
        # 1. Calculate degree of each vertex in remaining graph.
        # 2. For each i in I, let S_i be neighbors in C.
        #    Clique K_i = S_i ∪ {i}.
        #    Edges covered: all (v,i) for v in S_i, and all (u,v) for u,v in S_i.
        # 3. For C, let S_C be the set of vertices in C that have any uncovered internal edge.
        #    Actually, if we pick S_C = C, we cover all internal edges.
        #    Let K_C = C ∪ {i | i connected to all C}.
        #    Edges covered: all internal C, and all (c,i) for those i.
        # 4. Pick the clique (K_i or K_C) that covers the most edges.
        # 5. Remove covered edges.
        
        # This is O(n * E) per step?
        # n=90, E~2000. 90*2000 = 180k per step.
        # Number of steps? At least E/30 ~ 60 steps.
        # 60 * 180k = 10M ops. Very fast.
        
        edge_set = set()
        for e in edges:
            edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
        
        # Build initial adjacency for quick lookup
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            
        cliques = []
        remaining_edges = set(edge_set)
        
        # Precompute C and I
        C = list(range(k))
        I = list(range(k, n))
        
        while remaining_edges:
            if time.time() > deadline - 5:
                for (u, v) in remaining_edges:
                    cliques.append([u, v])
                break
                
            best_clique = None
            best_covered = 0
            best_covered_edges = set()
            
            # Candidate 1: Large clique from C
            # Let's try S = C.
            # Find i's connected to all of C.
            # But maybe a subset of C is better if it allows more i's?
            # No, if i is connected to subset S, it's connected to any superset? No.
            # If i is connected to all C, it's connected to any S.
            # So K_C = C ∪ {i | i ~ all C} is the maximal clique containing C.
            # Let's evaluate it.
            S_C = set(C)
            potential_i = [i for i in I if all(v in adj[i] for v in S_C)]
            K_C = list(S_C) + potential_i
            if len(K_C) >= 2:
                covered = set()
                for u in K_C:
                    for v in K_C:
                        if u < v:
                            e = (u, v)
                            if e in remaining_edges:
                                covered.add(e)
                if len(covered) > best_covered:
                    best_covered = len(covered)
                    best_clique = K_C
                    best_covered_edges = covered
            
            # Candidate 2: For each i in I, S_i = neighbors in C.
            for i in I:
                S_i = [v for v in C if (min(i,v), max(i,v)) in remaining_edges or (min(v,i), max(v,i)) in remaining_edges]
                # Actually, we need v in C such that (v,i) is an edge in the ORIGINAL graph?
                # No, the clique must be a valid clique in the graph.
                # So v must be connected to i in the original graph.
                # And we want to cover REMAINING edges.
                # So S_i should be {v in C | (v,i) in original_graph AND (v,i) in remaining_edges?}
                # No, if (v,i) is covered, we don't need it.
                # But if we include v in the clique, we cover (v,i) if it's remaining.
                # We also cover (u,v) internal edges.
                
                # Let's define S_i as neighbors of i in C that have at least one remaining edge?
                # Or just all neighbors in C?
                # If we include v in S_i, we cover (v,i) (if remaining) and (v, u) for u in S_i.
                # It is always beneficial to include v if it helps cover internal edges, even if (v,i) is already covered.
                # So S_i should be {v in C | v ~ i in original graph}.
                
                S_i = [v for v in C if v in adj[i]]
                if not S_i:
                    continue
                
                K_i = S_i + [i]
                if len(K_i) < 2:
                    continue
                    
                covered = set()
                for u in K_i:
                    for v in K_i:
                        if u < v:
                            e = (u, v)
                            if e in remaining_edges:
                                covered.add(e)
                
                if len(covered) > best_covered:
                    best_covered = len(covered)
                    best_clique = K_i
                    best_covered_edges = covered
            
            if best_clique is None:
                # Fallback: pick any edge
                u, v = next(iter(remaining_edges))
                best_clique = [u, v]
                best_covered_edges = {(min(u,v), max(u,v))}
            
            cliques.append(best_clique)
            
            # Remove covered edges
            remaining_edges -= best_covered_edges
            
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 15:
            edges = instance(*spec)
            results.append([[int(e[0]), int(e[1])] for e in edges])
            continue
        
        edges = instance(*spec)
        cliques = partition_split_graph(spec[0], spec[1], edges)
        
        # Convert to lists of Python ints
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(x) for x in c])
        
        results.append(final_cliques)
    
    return results
# EVOLVE-BLOCK-END
