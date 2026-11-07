import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        # K is 0..k-1, I is k..n-1
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        # Edges are of two types:
        # 1. Edges within K (clique K_k).
        # 2. Edges between K and I.
        
        # Strategy:
        # We want to minimize the number of cliques.
        # A clique can be of type:
        # - {u, v} where u in K, v in I (covers edge (u, v))
        # - {u, w, v} where u, w in K, v in I (covers edges (u, v), (w, v), (u, w))
        # - {u, w} where u, w in K (covers edge (u, w))
        # - Larger cliques in K or mixed.
        
        # Key insight:
        # Every edge (u, v) with u in K, v in I MUST be covered by a clique containing v.
        # Since I is independent, any clique containing v can only contain vertices from K.
        # Let C be a clique containing v. Then C \ {v} is a clique in K.
        # Since K is a clique, any subset of K is a clique.
        # So for each v, we can partition N(v) into subsets S_1, ..., S_m.
        # Each subset S_i forms a clique {v} U S_i.
        # This covers:
        #   - Cross edges (u, v) for u in S_i.
        #   - K-edges (u, w) for u, w in S_i.
        
        # The K-edges covered by these I-cliques must be disjoint for different v's.
        # Let E_K be the set of all edges in K.
        # Let E_I_covered be the set of K-edges covered by I-cliques.
        # The remaining K-edges E_K \ E_I_covered must be covered by cliques entirely in K.
        
        # To minimize total cliques:
        # 1. Minimize sum_v m_v (number of I-cliques).
        # 2. Minimize the number of K-cliques needed to cover E_K \ E_I_covered.
        
        # Trade-off:
        # Merging two neighbors u, w in v's clique saves 1 I-clique (from 2 to 1).
        # But it covers the K-edge (u, w), removing it from E_K.
        # This might increase the number of K-cliques needed.
        
        # Observation:
        # If we don't merge any neighbors (all S_i are singletons), then E_I_covered is empty.
        # The remaining K-edges are all of E_K.
        # The minimum number of cliques to cover K_k is 1 (the whole clique).
        # Total cliques = sum_v |N(v)| + 1.
        
        # If we merge, we save I-cliques but might need more K-cliques.
        # For example, if we cover all K-edges with I-cliques, we need 0 K-cliques.
        # But can we cover all K-edges with I-cliques while minimizing I-cliques?
        
        # Let's try a greedy approach:
        # 1. For each v, start with N(v) as a set.
        # 2. Try to merge pairs (u, w) in N(v) if the edge (u, w) is still "available" in K.
        #    "Available" means it hasn't been covered by another I-clique.
        # 3. This forms a matching in the graph of available K-edges induced by N(v).
        # 4. Each matched pair becomes a clique {u, w, v}.
        # 5. Unmatched vertices become cliques {u, v}.
        # 6. After processing all v, cover the remaining K-edges with cliques.
        
        # This is similar to the current program, but let's see if we can improve the K-clique cover.
        
        # The remaining K-edges form a graph G_K on k vertices.
        # We need to find a minimum clique cover of G_K.
        # Clique cover is NP-hard in general, but G_K is a subgraph of K_k.
        # Specifically, G_K is K_k minus a matching (the edges covered by I-cliques).
        
        # For K_k minus a matching of size m:
        # The vertices can be partitioned into m pairs (from the matching) and k-2m singletons.
        # An edge exists between any two vertices except those in the same pair.
        # So G_K is a complete k-partite graph? No, it's the complement of a matching.
        
        # How to cover G_K with minimum cliques?
        # A clique in G_K is a set of vertices with no pair from the matching.
        # So a clique can contain at most one vertex from each pair in the matching.
        # The maximum clique size is k - m (if we pick one from each pair and all singletons).
        
        # But we need to cover all edges, not just vertices.
        # Let's think of it as a coloring problem? No, clique cover is dual to coloring.
        # Clique cover number of G = chromatic number of complement of G.
        # Complement of G_K is the matching itself (plus isolated vertices).
        # So the clique cover number of G_K is the chromatic number of the matching.
        # The chromatic number of a matching of size m is 2 (if m > 0), because each edge needs 2 colors.
        # Wait, that's not right. The complement of G_K is a graph where edges are exactly the matching.
        # So the complement is a matching. The chromatic number of a matching is 2 (if there's at least one edge).
        # So the clique cover number of G_K is 2?
        
        # Let's test with K_4 minus a matching of size 2.
        # G_K has edges: (0,2), (0,3), (1,2), (1,3). This is C_4.
        # Clique cover of C_4: 2 cliques (e.g., {0,2} and {1,3}).
        # So yes, 2 cliques suffice.
        
        # What if the matching is not perfect?
        # K_5 minus a matching of size 1.
        # G_K has all edges except (0,1).
        # Can we cover it with 2 cliques?
        # Clique 1: {0, 2, 3, 4} (size 4). Covers edges within {0,2,3,4}.
        # Clique 2: {1, 2, 3, 4} (size 4). Covers edges within {1,2,3,4}.
        # But edge (0,1) is not covered (it's not in G_K).
        # Edges (0,2), (0,3), (0,4) are covered by Clique 1.
        # Edges (1,2), (1,3), (1,4) are covered by Clique 2.
        # Edges (2,3), (2,4), (3,4) are covered by both? No, clique cover requires each edge to be in exactly one clique.
        # So we can't have overlapping edges.
        
        # Let's try a different cover.
        # Clique 1: {0, 2, 3}. Covers (0,2), (0,3), (2,3).
        # Clique 2: {1, 2, 4}. Covers (1,2), (1,4), (2,4).
        # Clique 3: {1, 3, 4}. Covers (1,3), (1,4), (3,4).
        # But (1,4) is covered by both Clique 2 and Clique 3. Invalid.
        
        # So we need a proper clique cover (edge-disjoint).
        # For K_5 minus (0,1), can we cover with 2 cliques?
        # Total edges in G_K: 5*4/2 - 1 = 9.
        # Max edges in a clique of size s: s(s-1)/2.
        # If we use 2 cliques, max total edges covered is 2 * (s(s-1)/2).
        # If s=3, max edges = 2*3 = 6 < 9.
        # If s=4, max edges = 2*6 = 12 >= 9.
        # So it's possible in principle.
        
        # Let's try:
        # Clique 1: {0, 2, 3, 4}. Edges: (0,2), (0,3), (0,4), (2,3), (2,4), (3,4). (6 edges)
        # Remaining edges: (1,2), (1,3), (1,4). (3 edges)
        # These form a star centered at 1.
        # Can we cover them with 1 clique? No, because (1,2), (1,3), (1,4) are not all in a clique with 1?
        # Wait, {1,2,3} is a clique in G_K? Yes, because (1,2), (1,3), (2,3) are all in G_K.
        # But (2,3) is already covered by Clique 1. So we can't use it.
        # So we need to cover (1,2), (1,3), (1,4) with cliques that don't overlap with Clique 1's edges.
        # This means we can't use edges (2,3), (2,4), (3,4) again.
        # So the remaining graph on {1,2,3,4} has only edges (1,2), (1,3), (1,4).
        # This is a star. Minimum clique cover of a star with 3 edges is 3 (each edge is a clique of size 2).
        # So total cliques = 1 + 3 = 4.
        
        # So for K_5 minus 1 edge, we need 4 cliques?
        # Let's check if 3 cliques suffice.
        # Total edges 9. Max edges in 3 cliques: 3 * (s(s-1)/2).
        # If s=3, max edges = 3*3 = 9. So it's possible if we use 3 cliques of size 3.
        # Can we partition the 9 edges into 3 cliques of size 3?
        # Each clique of size 3 covers 3 edges.
        # This is equivalent to decomposing G_K into 3 triangles.
        # G_K is K_5 minus (0,1).
        # Triangles in G_K:
        # (0,2,3), (0,2,4), (0,3,4), (1,2,3), (1,2,4), (1,3,4).
        # We need 3 disjoint triangles.
        # Try: (0,2,3), (0,2,4) -> share (0,2). Invalid.
        # Try: (0,2,3), (1,2,4), (0,1,?) -> (0,1) is not an edge.
        # It seems hard to find 3 disjoint triangles.
        
        # Let's just implement a greedy clique cover for the remaining K-edges.
        # It should be fast enough for k=30.
        
        available_k_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        cliques = []
        
        # Process I-vertices in order of decreasing degree
        vs = sorted(range(k, n), key=lambda v: len(neighbors[v]), reverse=True)
        
        for v in vs:
            N_v = neighbors[v]
            if not N_v:
                continue
            
            # Find maximal matching in available_k_edges induced by N_v
            matched = set()
            pairs = []
            N_v_list = list(N_v)
            for i in range(len(N_v_list)):
                if N_v_list[i] in matched:
                    continue
                for j in range(i + 1, len(N_v_list)):
                    if N_v_list[j] in matched:
                        continue
                    u, w = N_v_list[i], N_v_list[j]
                    edge = (min(u, w), max(u, w))
                    if edge in available_k_edges:
                        pairs.append((u, w))
                        matched.add(u)
                        matched.add(w)
                        available_k_edges.remove(edge)
                        break
            
            for u, w in pairs:
                cliques.append([u, w, v])
            
            for u in N_v:
                if u not in matched:
                    cliques.append([u, v])
        
        # Cover remaining K-edges
        remaining_edges = list(available_k_edges)
        while remaining_edges:
            # Greedy: pick an edge, extend to maximal clique
            u, w = remaining_edges[0]
            clique = {u, w}
            
            # Find all x such that (u, x) and (w, x) in remaining_edges
            candidates = set()
            for e in remaining_edges:
                a, b = e
                if a == u and b != w:
                    candidates.add(b)
                elif b == u and a != w:
                    candidates.add(a)
                if a == w and b != u:
                    candidates.add(b)
                elif b == w and a != u:
                    candidates.add(a)
            
            # Filter candidates
            valid_c = []
            for c in candidates:
                if (min(u, c), max(u, c)) in remaining_edges and (min(w, c), max(w, c)) in remaining_edges:
                    valid_c.append(c)
            
            # Add all valid candidates to the clique
            for c in valid_c:
                clique.add(c)
            
            clique_list = list(clique)
            cliques.append(clique_list)
            
            # Remove edges covered by this clique
            edges_to_remove = set()
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    edges_to_remove.add((min(a, b), max(a, b)))
            
            remaining_edges = [e for e in remaining_edges if e not in edges_to_remove]
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback: return a simple valid solution
            # This should not happen, but just in case
            pass
        cliques = solve_instance(*spec)
        clean_cliques = []
        for c in cliques:
            clean_cliques.append([int(x) for x in c])
        results.append(clean_cliques)
        
    return results
