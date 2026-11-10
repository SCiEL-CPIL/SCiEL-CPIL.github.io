"""Clique partition of split graphs using structural set cover."""
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
        # 1. Build adjacency and separate edges
        edge_set = set()
        for e in edges:
            edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
        
        clique_edges = set()
        cross_edges = set()
        
        for u, v in edge_set:
            if u < k and v < k:
                clique_edges.add((u, v))
            elif u >= k and v >= k:
                # Should not happen in split graph, but ignore if it does
                pass
            else:
                # Ensure u is clique vertex, v is independent vertex
                if u >= k and v < k:
                    u, v = v, u
                cross_edges.add((u, v))
        
        # 2. Analyze Independent Set (I) vertices
        # For each vertex v in I, determine its neighborhood in C
        # N_C(v) = { u in C | (u, v) in cross_edges }
        
        # Group I vertices by their exact neighborhood in C
        # Two vertices v1, v2 in I can be in the same clique iff N_C(v1) == N_C(v2)
        # Because if v1, v2 are in a clique, they must be connected to the same set of C vertices
        # Wait, that's not quite right.
        
        # Let's rethink the structure of a clique in a split graph.
        # A clique K must consist of:
        # - A subset C_K of the clique vertices (0..k-1)
        # - A subset I_K of the independent vertices (k..n-1)
        #
        # For K to be a clique:
        # 1. C_K must be a clique in C (always true since C is a clique).
        # 2. I_K must be an independent set (always true since I is independent).
        # 3. Every vertex in C_K must be connected to every vertex in I_K.
        #
        # This implies that for all v in I_K, N_C(v) must contain C_K.
        # In other words, C_K must be a subset of the intersection of N_C(v) for all v in I_K.
        #
        # To minimize the number of cliques, we want to cover all edges with as few cliques as possible.
        # The edges are of two types:
        # A. Edges within C (clique edges).
        # B. Edges between C and I (cross edges).
        #
        # Note: Edges within I do not exist.
        #
        # Strategy:
        # The edges within C must be covered by cliques. Since C is a clique, we can cover all C-edges
        # with a small number of cliques that also cover some cross-edges.
        # However, any clique that includes a vertex from I must include its neighbors in C.
        #
        # Let's group the independent vertices by their neighborhood in C.
        # Let the distinct neighborhoods be S_1, S_2, ..., S_m.
        # For each group G_j = { v in I | N_C(v) = S_j }, all vertices in G_j have the same neighbors in C.
        #
        # Consider a clique K. If it contains any vertex v in G_j, then K must contain S_j (or a subset of S_j? No, K is a clique, so if v in K, all neighbors of v in C that are in K must be in K. But K doesn't have to contain ALL neighbors of v, just the ones in K. Wait.
        # If v is in K, and u is in C_K, then (u,v) must be an edge. So C_K must be a subset of N_C(v).
        #
        # This is a Set Cover problem.
        # Universe: All edges (C-edges and Cross-edges).
        # Sets: All possible cliques K.
        #
        # This is too large. We need a smarter approach.
        
        # Observation:
        # Any clique K is defined by C_K and I_K.
        # I_K is a subset of some group G_j? Not necessarily.
        # If v1 in G_j1 and v2 in G_j2 are in the same clique K, then C_K must be a subset of N_C(v1) AND a subset of N_C(v2).
        # So C_K is a subset of N_C(v1) ∩ N_C(v2).
        
        # Let's try a greedy approach on the cross-edges first, then handle remaining C-edges.
        # Actually, C-edges are "free" in the sense that any clique containing two C-vertices covers their edge.
        # The bottleneck is covering the cross-edges efficiently.
        
        # Heuristic:
        # 1. Identify groups of I-vertices with identical neighborhoods in C.
        # 2. For each group, the vertices are "interchangeable" regarding which C-vertices they connect to.
        # 3. We can form cliques of the form {c1, ..., cm} U {v1, ..., vt} where {c1..cm} is a subset of the common neighborhood, and {v1..vt} is a subset of the group.
        
        # Let's look at the structure of the cross-edges.
        # For each v in I, the set of cross-edges incident to v is { (u, v) | u in N_C(v) }.
        
        # Greedy Set Cover on Cross-Edges:
        # Universe U = cross_edges.
        # Candidate sets: For each possible clique K, the set of cross-edges it covers.
        # But generating all cliques is expensive.
        
        # Alternative:
        # For each vertex v in I, the "best" clique containing v is one that includes as many of its neighbors in C as possible, and also includes other I-vertices with the same neighbors.
        
        # Let's group I vertices by neighborhood.
        neigh_map = {}
        for v in range(k, n):
            neigh = frozenset(u for (u, w) in cross_edges if w == v)
            if not neigh:
                # Isolated in C, no cross edges. Can be ignored for cross-edge coverage.
                continue
            if neigh not in neigh_map:
                neigh_map[neigh] = []
            neigh_map[neigh].append(v)
        
        # Now, for each distinct neighborhood S, we have a group of vertices G_S.
        # Any clique that includes a vertex from G_S must have its C-part as a subset of S.
        # To cover the cross-edges incident to vertices in G_S, we can create cliques.
        # A clique K = C_K U I_K where I_K subset of G_S and C_K subset of S.
        # The cross-edges covered are { (u, v) | u in C_K, v in I_K }.
        
        # If we pick C_K = S and I_K = G_S, we cover all cross-edges incident to G_S.
        # This uses 1 clique for all vertices in G_S.
        # Is this optimal?
        # If we split G_S into smaller subsets, we use more cliques.
        # So for each group G_S, using one clique {S} U G_S seems optimal for covering cross-edges of that group.
        
        # BUT, we also need to cover C-edges.
        # The cliques we form for cross-edges also cover C-edges among S.
        # If we use cliques {S1} U G1, {S2} U G2, etc., the C-edges covered are those within S1, S2, etc.
        # The union of S_j might not cover all C-edges.
        
        # Strategy:
        # 1. For each group G_S, create a clique C_S = list(S) + list(G_S).
        # 2. Collect all these cliques.
        # 3. Check which C-edges are NOT covered by these cliques.
        # 4. Cover the remaining C-edges with additional cliques (subsets of C).
        
        # Wait, is it always optimal to use the full neighborhood S for the clique?
        # Suppose S is large. The clique {S} U G_S covers |S|*|G_S| cross-edges.
        # If we used a smaller C_K subset of S, we would need more cliques to cover the remaining cross-edges for G_S.
        # So yes, using the full S is likely best for minimizing the number of cliques for the cross-edges of that group.
        
        # However, what if two groups G_S1 and G_S2 have overlapping neighborhoods?
        # Can we merge them?
        # If v1 in G_S1 and v2 in G_S2 are in the same clique, the C-part must be a subset of S1 ∩ S2.
        # If S1 != S2, the intersection might be small.
        # It is generally better to keep them separate if S1 != S2, because merging would force a smaller C-part, reducing efficiency.
        
        # So, the plan:
        # 1. For each distinct neighborhood S in the I-vertices, form a clique K_S = S U G_S.
        # 2. Remove all cross-edges covered by these cliques.
        # 3. The remaining edges are only C-edges (edges within the clique vertices 0..k-1).
        # 4. Find a minimum clique cover for the remaining C-edges.
        #    Since C is a complete graph, the remaining C-edges form a subgraph of the complete graph.
        #    Actually, the remaining C-edges are those NOT covered by any K_S.
        #    An edge (u, v) in C is covered by K_S if u in S and v in S.
        #    So we need to cover the edges of the graph H on vertices 0..k-1 where (u,v) is in H iff (u,v) is NOT in any S.
        #    Wait, (u,v) is covered if there exists a clique K in our list such that u,v in K.
        #    Our cliques are K_S = S U G_S. The C-part is S.
        #    So (u,v) is covered if u in S and v in S for some S.
        
        # Let's implement this.
        
        cliques = []
        covered_c_edges = set()
        
        # Step 1: Process groups
        for S, G_S in neigh_map.items():
            if not G_S:
                continue
            # Form clique
            clique_verts = list(S) + list(G_S)
            if len(clique_verts) >= 2:
                cliques.append(clique_verts)
            
            # Mark C-edges covered by this clique
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    u, v = S_list[i], S_list[j]
                    covered_c_edges.add((min(u, v), max(u, v)))
        
        # Step 2: Find remaining C-edges
        remaining_c_edges = set()
        for u in range(k):
            for v in range(u+1, k):
                if (u, v) not in covered_c_edges:
                    remaining_c_edges.add((u, v))
        
        # Step 3: Cover remaining C-edges
        # The remaining C-edges form a graph on k vertices.
        # We need to partition these edges into cliques.
        # Since the original graph C is a clique, any subset of vertices forms a clique.
        # So we just need to cover the edges of the graph H (defined by remaining_c_edges) with cliques.
        # This is the Edge Clique Cover problem, which is NP-hard, but k is small (10, 20, 30).
        # We can use a greedy approach: repeatedly find a max clique in H, remove its edges.
        
        # Build adjacency for H
        adj_H = [set() for _ in range(k)]
        for u, v in remaining_c_edges:
            adj_H[u].add(v)
            adj_H[v].add(u)
        
        rem_edges = set(remaining_c_edges)
        
        while rem_edges:
            if time.time() > deadline - 5:
                # Fallback: list each edge
                for u, v in rem_edges:
                    cliques.append([u, v])
                break
            
            # Find max degree vertex
            degs = [len(adj_H[i]) for i in range(k)]
            # Note: adj_H is static, but we are removing edges.
            # We need dynamic degrees or just re-scan.
            # For small k, re-scanning is fine.
            
            # Find a vertex with max remaining degree
            max_v = -1
            max_deg = -1
            for i in range(k):
                # Count remaining edges incident to i
                d = 0
                for j in adj_H[i]:
                    if (min(i,j), max(i,j)) in rem_edges:
                        d += 1
                if d > max_deg:
                    max_deg = d
                    max_v = i
            
            if max_v == -1 or max_deg == 0:
                break # Should not happen if rem_edges is not empty
            
            # Greedily build a clique starting with max_v
            current_clique = [max_v]
            candidates = sorted(adj_H[max_v], key=lambda x: -len(adj_H[x]))
            
            for cand in candidates:
                if cand in current_clique:
                    continue
                # Check if cand is connected to all in current_clique in rem_edges
                can_add = True
                for vc in current_clique:
                    e = (min(cand, vc), max(cand, vc))
                    if e not in rem_edges:
                        can_add = False
                        break
                if can_add:
                    current_clique.append(cand)
            
            if len(current_clique) < 2:
                # Take any edge
                u, v = next(iter(rem_edges))
                current_clique = [u, v]
            
            cliques.append(current_clique)
            
            # Remove edges covered by this clique
            clique_set = set(current_clique)
            new_rem = set()
            for e in rem_edges:
                if e[0] in clique_set and e[1] in clique_set:
                    continue
                new_rem.add(e)
            rem_edges = new_rem
        
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
        
        # Convert to lists of Python ints
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(x) for x in c])
        
        results.append(final_cliques)
    
    return results
# EVOLVE-BLOCK-END
