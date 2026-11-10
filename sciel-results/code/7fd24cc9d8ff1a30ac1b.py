"""Clique partition of split graphs using Global MWIS on Conflict Graph of Candidate Cliques."""
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
        # 1. Build adjacency and separate edges
        adj_c = [set() for _ in range(k)]  # Neighbors in I for each c in C
        adj_i = [set() for _ in range(n)]  # Neighbors in C for each i in I
        c_edges = []
        cross_edges = []
        
        # Precompute C-clique adjacency for fast checks
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
                adj_c[c].add(i)
                adj_i[i].add(c)

        # Identify active I vertices (those with at least one cross edge)
        active_i = [i for i in range(k, n) if adj_i[i]]
        
        if not active_i and not c_edges:
            return []

        # 2. Generate Candidate Cliques
        # A candidate clique is of the form {i} U S where i in I and S subset N(i) is a clique in C.
        # We generate candidates for each active i.
        # To keep the number manageable, we generate:
        # a) Maximal cliques in N(i) (using a simple greedy maximal clique finding)
        # b) Single edges {i, c} for all c in N(i) (fallback, ensures validity)
        
        candidates = []
        
        for i in active_i:
            neighbors = sorted(adj_i[i])
            
            # Generate maximal cliques in the subgraph of C induced by neighbors
            # We use a simple recursive backtracking or greedy approach.
            # Given small k (<=30), we can be somewhat exhaustive or use a heuristic.
            # Let's use a heuristic: Greedy Maximal Clique on the induced subgraph.
            
            # We'll generate multiple candidates per i by varying the starting vertex or using a simple local search.
            # For efficiency, let's generate:
            # 1. The greedy maximal clique starting from each vertex in neighbors (might be slow if k is large, but k=30 is small).
            #    Actually, just finding ONE good maximal clique per i might be enough, or a few.
            #    Let's find the maximum clique in N(i) using a simple branch and bound or just greedy multiple times.
            
            # Simpler: Generate all cliques of size 2 (edges) and try to extend them greedily.
            # This ensures we cover all cross edges.
            
            # Strategy: 
            # For each i, we want a set of disjoint (in terms of C-edges) cliques? No, the MWIS will handle conflicts.
            # We just need a good pool of candidates.
            
            # Let's generate:
            # - For each c in neighbors: candidate {i, c}
            # - For each pair (c1, c2) in neighbors that are connected: try to extend to maximal clique.
            
            # To limit candidates, we will generate:
            # 1. All edges {i, c}
            # 2. A few large cliques found by greedy expansion from random/high-degree starts.
            
            # Greedy Maximal Clique in N(i):
            def find_maximal_clique_in_neigh(starts, neighbors_set, adj_c_local):
                # adj_c_local is the global C adjacency, but we only care about nodes in neighbors_set
                # We want a clique S such that S subset neighbors_set and S is a clique in C.
                # Greedy: Start with a set of vertices, add vertices connected to all in set.
                
                best_clique = set()
                for start in starts:
                    if start not in neighbors_set:
                        continue
                    current = {start}
                    # Try to expand
                    # We can do a simple greedy expansion:
                    # Candidates are neighbors of all in current that are in neighbors_set
                    # Add the one with highest degree in the remaining subgraph?
                    
                    # Simple greedy:
                    while True:
                        # Find all vertices in neighbors_set connected to all in current
                        common = set(neighbors_set)
                        for v in current:
                            common &= adj_c_local[v]
                        common -= current
                        
                        if not common:
                            break
                        
                        # Pick a vertex from common. Let's pick the one with max degree in common (greedy heuristic)
                        # Or just the first one for determinism?
                        # Let's pick max degree in the subgraph induced by common.
                        best_v = None
                        max_deg = -1
                        for v in common:
                            deg = len(common & adj_c_local[v])
                            if deg > max_deg:
                                max_deg = deg
                                best_v = v
                        if best_v is None:
                            break
                        current.add(best_v)
                        
                    if len(current) > len(best_clique):
                        best_clique = current
                return best_clique

            # Generate candidates for i
            # 1. Single edges
            for c in neighbors:
                candidates.append((i, [c]))
            
            # 2. Greedy maximal cliques
            # We'll run the greedy expansion starting from each vertex in neighbors to get diverse maximal cliques.
            # To save time, we might limit this if neighbors is large.
            # k=30, neighbors <= 30. 30 starts * 30 steps * 30 checks = 27000 ops. Fast.
            
            seen_cliques = set()
            for start_c in neighbors:
                clique_set = find_maximal_clique_in_neigh([start_c], set(neighbors), c_adj)
                if len(clique_set) >= 2:
                    key = tuple(sorted(clique_set))
                    if key not in seen_cliques:
                        seen_cliques.add(key)
                        candidates.append((i, list(clique_set)))
            
            # Also, try to find a large clique by random restarts?
            # Maybe just the greedy from each start is enough.

        # 3. Build Conflict Graph for MWIS
        # Nodes: candidates
        # Edge between two candidates if they share a C-C edge.
        # Two candidates (i1, S1) and (i2, S2) conflict if:
        #   - There exists u, v in S1 such that (u, v) is a C-edge AND u, v in S2?
        #   - Wait, if i1 == i2, they can't be in the same partition because they are different cliques for the same i?
        #     Actually, if i1 == i2, the cliques are {i1} U S1 and {i2} U S2.
        #     If S1 and S2 share a vertex c, then the cross edge (i1, c) is covered by both?
        #     No, the cross edge (i, c) is covered by the clique containing i and c.
        #     If we select two candidates for the same i, say {i, c1} and {i, c2}, they don't conflict on cross edges (different c's).
        #     But do they conflict on C-edges?
        #     {i, c1} has no C-edges (size 2, one C, one I).
        #     {i, c1, c2} has C-edge (c1, c2).
        #     If we select {i, c1, c2} and {i, c1}, the edge (c1, c2) is covered by the first. The cross edge (i, c1) is covered by both?
        #     Yes! The cross edge (i, c1) is in both cliques.
        #     So any two candidates for the SAME i that share a C-vertex conflict on that cross edge.
        #     Actually, ANY two candidates for the same i that overlap in S conflict.
        #     If they are disjoint in S, e.g., {i, c1} and {i, c2}, they don't share cross edges.
        #     But do they share C-edges? No, size 2 cliques have no C-edges.
        #     So disjoint S for same i are OK?
        #     Wait, if we select {i, c1} and {i, c2}, we have two cliques.
        #     Edge (i, c1) is in clique 1. Edge (i, c2) is in clique 2.
        #     This is valid.
        #     So conflicts for same i are only if S1 and S2 intersect.
        
        #   - If i1 != i2:
        #     They conflict if they share a C-C edge.
        #     S1 induces a set of C-edges E1. S2 induces E2.
        #     If E1 ∩ E2 is not empty, they conflict.
        #     Note: They don't conflict on cross edges because i1 != i2.
        
        # To simplify, let's define conflict as:
        # 1. Same i and S1 ∩ S2 != empty.
        # 2. Different i and (C-edges in S1) ∩ (C-edges in S2) != empty.

        # We need to map candidates to their covered C-edges.
        # For a candidate (i, S), the C-edges covered are all (u, v) with u, v in S and u < v.
        
        cand_c_edges = []
        for i, S in candidates:
            edges_in_S = set()
            for idx_u in range(len(S)):
                u = S[idx_u]
                for idx_v in range(idx_u + 1, len(S)):
                    v = S[idx_v]
                    if (min(u, v), max(u, v)) in c_adj[u]: # Check if it's a real edge
                        edges_in_S.add((min(u, v), max(u, v)))
            cand_c_edges.append(edges_in_S)

        # Also, we need to ensure that if we select a candidate (i, S), ALL cross edges (i, c) for c in S are covered.
        # And no cross edge is covered twice.
        # My conflict definition above handles same-i overlap.
        # What if two different i's cover the same cross edge? Impossible, cross edge is (i, c), unique i.
        
        # So the conflict graph is valid.
        
        # 4. Greedy MWIS
        # We want to select a subset of candidates such that no two conflict.
        # Objective: Maximize total edges covered? Or minimize number of cliques?
        # The task says "Minimize the total number of cliques".
        # So we want to select as FEW cliques as possible that cover ALL edges.
        # This is a Set Cover problem with conflicts (Independent Set in Conflict Graph where we want to cover all elements).
        # Wait, if we just pick an Independent Set, we might not cover all edges.
        # We need a **Covering Independent Set**.
        
        # Greedy Approach for Covering Independent Set:
        # 1. Start with all edges uncovered.
        # 2. While uncovered edges exist:
        #    - Find a candidate that covers the most uncovered edges.
        #    - Among candidates that don't conflict with ALREADY SELECTED candidates, pick the best.
        #    - If no such candidate exists (all candidates covering remaining edges conflict with selected), 
        #      we have a problem. 
        #      Actually, if we pick greedily, we might paint ourselves into a corner.
        #      However, since we have single-edge candidates {i, c} for every cross edge, and {u, v} for every C-edge?
        #      We didn't generate {u, v} for C-edges!
        #      We only generated mixed cliques.
        #      If a C-edge (u, v) is not covered by any selected mixed clique, we MUST add a pure C-clique.
        #      So we should add pure C-cliques as candidates too?
        #      Or handle them at the end.
        
        # Let's refine the Greedy:
        selected = []
        selected_c_edges = set()
        selected_cross_edges = set() # (i, c)
        
        # Precompute uncovered edges
        all_cross_edges = set((i, c) for c, i in cross_edges)
        all_c_edges = set(c_edges)
        
        uncovered_cross = set(all_cross_edges)
        uncovered_c = set(all_c_edges)
        
        # Sort candidates by "impact" (number of edges covered) descending?
        # But impact changes as edges get covered.
        
        # To avoid corner cases, we can use a backtracking or a more careful greedy.
        # Given the size, a simple greedy might fail to find a valid cover if it blocks necessary edges.
        # But we have single-edge candidates.
        # If we pick a large clique, it covers many edges.
        # If later we need to cover an edge that conflicts with the large clique, we can't use any candidate that conflicts with the large clique.
        # But we can use single-edge candidates that DON'T conflict with the large clique.
        # A single edge {i, c} conflicts with a selected clique (i', S') if:
        #   - i == i' and c in S' (conflict on cross edge)
        #   - i != i' and ... {i, c} has no C-edges, so no conflict on C-edges.
        # So {i, c} only conflicts with selected cliques that contain c and have the same i.
        # This is very loose.
        
        # What about pure C-edges?
        # If a C-edge (u, v) is uncovered, we need a clique containing u, v.
        # This clique can be {u, v} (pure) or {i, u, v} (mixed).
        # If we didn't generate {i, u, v} in candidates, we can't use it.
        # We should ensure candidates include enough variety.
        
        # Let's add pure C-cliques as candidates too?
        # If we add {u, v} as a candidate, it has no I-vertex.
        # It conflicts with other candidates if they share C-edges.
        
        # Let's just handle C-edges at the end if they are uncovered.
        # If uncovered_c is not empty, we add minimal pure cliques.
        
        # Greedy Selection:
        # We iterate. At each step, we pick a candidate that maximizes the number of NEW edges covered.
        # New edges = (uncovered_cross ∩ CrossEdges(cand)) + (uncovered_c ∩ CEdges(cand))
        
        # To make it fast, we can precompute edge sets for each candidate.
        cand_cross_edges = []
        for i, S in candidates:
            cand_cross_edges.append(set((i, c) for c in S))
            
        # Initialize
        best_cliques = []
        
        # We'll use a priority queue or just scan each time.
        # Number of candidates is small (maybe a few hundred).
        # Number of steps is number of cliques (maybe a few hundred).
        # Scanning is O(N_cand * E) which is fine.
        
        while uncovered_cross or uncovered_c:
            if time.time() > deadline - 10:
                break # Fallback later
            
            best_idx = -1
            best_gain = 0
            
            for idx, (i, S) in enumerate(candidates):
                # Check if candidate conflicts with selected
                # Conflict if:
                # 1. Shares a cross edge with selected
                # 2. Shares a C-edge with selected
                
                # We maintain selected_cross_edges and selected_c_edges.
                
                # Check Cross Edges
                cand_cross = cand_cross_edges[idx]
                if cand_cross & selected_cross_edges:
                    continue
                    
                # Check C Edges
                cand_ce = cand_c_edges[idx]
                if cand_ce & selected_c_edges:
                    continue
                
                # Calculate gain
                gain_cross = len(cand_cross & uncovered_cross)
                gain_c = len(cand_ce & uncovered_c)
                gain = gain_cross + gain_c
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            
            if best_idx == -1:
                # No candidate improves coverage?
                # This can happen if all remaining edges are only covered by candidates that conflict with selected.
                # Or if gain is 0.
                # If gain is 0, we are stuck?
                # If uncovered_cross is not empty, there must be a candidate covering it (single edges).
                # Why would it conflict?
                # Single edge {i, c} conflicts if (i, c) in selected_cross_edges.
                # But if (i, c) is in uncovered_cross, it is NOT in selected_cross_edges.
                # So single edges should always be available if their edge is uncovered.
                # Unless... we didn't generate single edges for all?
                # We did: "for c in neighbors: candidates.append((i, [c]))"
                # So there should always be a candidate with gain > 0 if uncovered edges exist.
                
                # Maybe best_gain is 0 because all candidates covering uncovered edges conflict?
                # If so, we have a bug in conflict logic or candidate generation.
                # Let's force pick a single edge if stuck.
                found = False
                for (i, c) in list(uncovered_cross):
                    # Find candidate (i, [c])
                    # It should exist.
                    # Does it conflict?
                    # It conflicts if (i, c) in selected_cross (False, it's uncovered)
                    # Or if it has C-edges (False, size 1)
                    # So it should be valid.
                    # Let's just pick the first uncovered cross edge and form a clique.
                    best_cliques.append([i, c])
                    selected_cross_edges.add((i, c))
                    uncovered_cross.discard((i, c))
                    found = True
                    break
                
                if not found:
                    # Only C-edges left?
                    # Pick a C-edge and form a pure clique.
                    if uncovered_c:
                        u, v = next(iter(uncovered_c))
                        best_cliques.append([u, v])
                        selected_c_edges.add((u, v))
                        uncovered_c.discard((u, v))
                    else:
                        break
                continue
            
            # Select best candidate
            i, S = candidates[best_idx]
            clique = [i] + sorted(S)
            best_cliques.append(clique)
            
            # Update state
            selected_cross_edges |= cand_cross_edges[best_idx]
            selected_c_edges |= cand_c_edges[best_idx]
            
            uncovered_cross -= cand_cross_edges[best_idx]
            uncovered_c -= cand_c_edges[best_idx]
            
            # Remove this candidate from consideration?
            # We can't just remove it, but we can mark it as "used" or just let the conflict check handle it.
            # Since it conflicts with itself (shares edges), it will be skipped in future iterations.
            # Wait, does it conflict with itself?
            # cand_cross & selected_cross: Yes, all its cross edges are now selected.
            # So it will be skipped.

        # 5. Handle remaining uncovered edges (safety)
        if uncovered_cross:
            for (i, c) in uncovered_cross:
                best_cliques.append([i, c])
        if uncovered_c:
            # Group remaining C-edges into cliques
            # Simple: add each edge as a clique? Or try to merge?
            # For safety, add each edge.
            for (u, v) in uncovered_c:
                best_cliques.append([u, v])
        
        return best_cliques

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
            # Ensure distinct ints
            c_ints = list(set(int(x) for x in c))
            if len(c_ints) >= 2:
                final_cliques.append(sorted(c_ints))
        
        results.append(final_cliques)
    
    return results
# EVOLVE-BLOCK-END
