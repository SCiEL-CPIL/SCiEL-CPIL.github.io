"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        # Edges within clique C
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # Edges between C and I
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def solve_split_graph(n, k, p, seed):
        if time.time() > deadline:
            edges = instance(n, k, p, seed)
            return [[int(u), int(v)] for u, v in edges]

        edges = instance(n, k, p, seed)
        if not edges:
            return []

        # 1. Parse Graph Structure
        # E_CC: edges within C (0..k-1)
        # N_C[v]: neighbors of v (v in I, i.e., v >= k) in C
        N_C = {}
        for v in range(k, n):
            N_C[v] = set()
        
        # Build N_C from edges
        for u, v in edges:
            if u < k and v < k:
                pass # All pairs in C are edges (C is a clique)
            elif u < k and v >= k:
                N_C[v].add(u)
            elif v < k and u >= k:
                N_C[u].add(v)
        
        # The set of all edges
        # 1. All pairs in C
        # 2. Edges (u, v) where u in C, v in I, u in N_C[v]
        
        # Strategy:
        # We need to partition the edge set into cliques.
        # In a split graph, any maximal clique is either a subset of C or {v} union S where v in I and S is a clique in N_C[v] (subset of C).
        # Since C is a clique, any subset of C is a clique.
        # Any set {v} U S is a clique iff S is a clique in G[C] (which is always true since C is a clique) AND for all u in S, u in N_C[v].
        # So {v} U S is a clique iff S is a subset of N_C[v].
        
        # Key Insight:
        # An edge (u, v) with v in I must be covered by a clique containing v.
        # Since v is in I, any clique containing v is of the form {v} U S where S subset N_C[v].
        # Let K_v be the set of cliques chosen that contain v.
        # The union of the C-parts of K_v must cover N_C[v].
        # Let S_v = union_{K in K_v} S_K. Then S_v must contain N_C[v].
        # The cliques in K_v are subsets of {v} U C.
        
        # An edge (u, w) with u, w in C can be covered by:
        # 1. A clique entirely in C.
        # 2. A clique {v} U S containing both u and w (i.e., u, w in S subset N_C[v]).
        
        # To minimize the number of cliques:
        # We want to "pack" C-C edges into the C-parts of the cliques that cover I-edges.
        # Let's first determine the "essential" structure for covering I-edges.
        # For each v in I, we need to cover N_C[v].
        # If we use a single clique {v} U N_C[v] for each v, we cover all I-edges and also cover the C-C edges within each N_C[v].
        # The remaining C-C edges are those pairs {u, w} such that there is no v in I with u, w in N_C[v].
        # Let's call this set of "uncovered" C-C edges E_rem.
        # We then need to partition E_rem into cliques. Since these are edges in the complete graph C, we just need to find a clique partition of the graph (C, E_rem).
        
        # Is using {v} U N_C[v] always optimal?
        # Consider two vertices v1, v2 in I.
        # N_C[v1] = {1, 2}, N_C[v2] = {2, 3}.
        # Option A: Cliques {v1, 1, 2} and {v2, 2, 3}.
        #   Edges covered in C: (1,2) by C1, (2,3) by C2.
        #   If (1,3) is an edge in C, it is NOT covered by these. We need a third clique for (1,3), e.g., {1, 3}.
        #   Total 3 cliques.
        # Option B: Clique {v1, v2, 2} is NOT a clique because v1, v2 are in I and not connected (I is independent).
        #   So we cannot combine v1 and v2 in a clique.
        #   What if we split N_C?
        #   If we use {v1, 1} and {v1, 2} and {v2, 2} and {v2, 3}? That's 4 cliques. Worse.
        #   What if we use a clique in C to cover (1,3)?
        #   {v1, 1, 2}, {v2, 2, 3}, {1, 3}. Still 3.
        #   Can we do better?
        #   Maybe {v1, 1} and {v2, 3} and {1, 2, 3}?
        #   {1, 2, 3} is a clique in C. It covers (1,2), (2,3), (1,3).
        #   {v1, 1} covers (v1,1). But (v1,2) is an edge! It is not covered by {v1,1} nor {1,2,3} (v1 not in {1,2,3}).
        #   So (v1,2) must be covered by a clique containing v1.
        #   So we must have a clique {v1} U S1 where 2 in S1.
        #   If we use {v1, 1, 2}, it covers (v1,1) and (v1,2).
        #   It seems we are forced to cover N_C[v] with cliques containing v.
        
        # Hypothesis: For each v in I, the set of cliques containing v must cover N_C[v].
        # Since any clique containing v is {v} U S with S subset N_C[v], the union of these S's must be N_C[v].
        # Let K_v be the collection of S's.
        # The cost is sum(|K_v|) over all v, PLUS the cliques used to cover the remaining C-C edges.
        
        # Note: A clique {v} U S covers C-C edges within S.
        # If we choose S = N_C[v], we cover all C-C edges within N_C[v].
        # If we choose smaller S's, we cover fewer C-C edges.
        # So choosing S = N_C[v] is "dominant" for covering C-C edges, as it covers a superset of edges for the same clique count (1 clique for v).
        # Wait, could using 2 cliques for v cover *more* C-C edges?
        # If N_C[v] is not a clique... but N_C[v] is a subset of C, and C is a clique, so N_C[v] is always a clique.
        # So {v} U N_C[v] is always a valid clique.
        # And it covers ALL C-C edges induced by N_C[v].
        # If we split N_C[v] into S1, S2, we cover edges in S1 and S2, but NOT edges between S1 and S2.
        # So {v} U N_C[v] covers strictly more C-C edges (or equal) than any partition of N_C[v] into multiple cliques, for a single clique vs multiple.
        # Actually, 1 clique covers all pairs in N_C[v].
        # 2 cliques S1, S2 cover pairs in S1 and S2. Missing pairs between S1 and S2.
        # So 1 clique {v} U N_C[v] is always better or equal in terms of C-C edge coverage per clique count for vertex v.
        
        # Therefore, the optimal strategy is:
        # 1. For each v in I, select the clique K_v = {v} U N_C[v].
        # 2. This covers all I-edges and all C-C edges that are "covered" by some N_C[v].
        # 3. Identify the remaining C-C edges that are not covered by any N_C[v].
        # 4. Partition these remaining edges into minimum number of cliques.
        
        # Step 3 details:
        # An edge (u, w) in C is "covered" if there exists v in I such that u in N_C[v] and w in N_C[v].
        # If not covered, it must be covered by a clique entirely in C.
        
        # Let's implement this.
        
        covered_cc = set()
        cliques = []
        
        # Generate cliques for I vertices
        for v in range(k, n):
            S = sorted(list(N_C[v]))
            if not S:
                continue # No edges from v to C, so no clique needed (or rather, v is isolated, but problem says every edge must be covered. If v has no edges, it's not in any clique. But wait, "Every listed set must be a clique... and every edge must lie in exactly one listed clique". If v has no edges, we don't need to list v.
            # If v has edges, N_C[v] is not empty.
            clique = [v] + S
            cliques.append(clique)
            # Mark C-C edges covered by this clique
            for i in range(len(S)):
                for j in range(i+1, len(S)):
                    covered_cc.add((min(S[i], S[j]), max(S[i], S[j])))
        
        # Identify remaining C-C edges
        remaining_cc = []
        for u in range(k):
            for w in range(u+1, k):
                e = (u, w)
                if e not in covered_cc:
                    remaining_cc.append(e)
        
        # Now we need to partition remaining_cc into minimum number of cliques.
        # The graph is (C, remaining_cc). C is a set of k vertices.
        # We need a clique partition of this graph.
        
        if not remaining_cc:
            # Done
            pass
        else:
            # Greedy approach for clique partition of remaining_cc
            # Since remaining_cc is a subgraph of C (a clique), the "maximal cliques" in the graph of remaining_cc are just the maximal sets of vertices forming a clique in remaining_cc.
            # Actually, since the vertices are C and the edges are a subset, we just need to find a partition of the edges into cliques.
            
            # Heuristic: 
            # 1. Find a maximal clique in the graph (C, remaining_cc).
            # 2. Remove its edges.
            # 3. Repeat.
            
            # For k=30, we can afford a decent search.
            # But simple greedy might be enough.
            
            rem_edges = set(remaining_cc)
            verts = set(range(k))
            
            while rem_edges:
                if time.time() > deadline:
                    # Fallback: add remaining as size-2 cliques
                    for e in rem_edges:
                        cliques.append([e[0], e[1]])
                    rem_edges.clear()
                    break
                
                # Find a maximal clique in the graph induced by rem_edges
                # To speed up, we can restrict to vertices that are endpoints of rem_edges
                active_verts = set()
                for u, v in rem_edges:
                    active_verts.add(u)
                    active_verts.add(v)
                
                # Greedy Max Clique:
                # Start with a random vertex, or the one with highest degree in rem_edges
                deg = {v: 0 for v in active_verts}
                for u, v in rem_edges:
                    deg[u] += 1
                    deg[v] += 1
                
                # Pick vertex with max degree
                if not active_verts:
                    break
                start_v = max(active_verts, key=lambda x: deg[x])
                
                clique = [start_v]
                candidates = [v for v in active_verts if v != start_v]
                
                # Greedy expansion
                while True:
                    # Filter candidates: must be connected to all in clique via rem_edges
                    new_candidates = []
                    for v in candidates:
                        ok = True
                        for u in clique:
                            e = (min(u, v), max(u, v))
                            if e not in rem_edges:
                                ok = False
                                break
                        if ok:
                            new_candidates.append(v)
                    candidates = new_candidates
                    
                    if not candidates:
                        break
                    
                    # Pick best candidate
                    # Score: number of connections to other candidates? Or just pick first?
                    # Let's pick the one with max degree in the candidate subgraph
                    cand_deg = {v: 0 for v in candidates}
                    for v in candidates:
                        for w in candidates:
                            if v == w: continue
                            e = (min(v, w), max(v, w))
                            if e in rem_edges:
                                cand_deg[v] += 1
                    
                    best_v = max(candidates, key=lambda x: cand_deg[x])
                    clique.append(best_v)
                    candidates.remove(best_v)
                
                if len(clique) >= 2:
                    cliques.append(clique)
                    # Remove edges in this clique from rem_edges
                    for i in range(len(clique)):
                        for j in range(i+1, len(clique)):
                            e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                            rem_edges.discard(e)
                
                # If no clique found (should not happen if rem_edges not empty), pick an edge
                if len(clique) < 2 and rem_edges:
                    e = next(iter(rem_edges))
                    cliques.append([e[0], e[1]])
                    rem_edges.discard(e)
            
            # Final check: if rem_edges still has stuff (shouldn't), add as edges
            for e in rem_edges:
                cliques.append([e[0], e[1]])
            
        # Convert to required format
        result = []
        for c in cliques:
            # Ensure ints
            result.append([int(x) for x in c])
            
        return result

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[int(u), int(v)] for u, v in edges])
        else:
            res = solve_split_graph(*spec)
            results.append(res)
            
    return results
# EVOLVE-BLOCK-END
