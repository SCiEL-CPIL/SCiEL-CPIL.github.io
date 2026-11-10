import time
from random import Random

def construct():
    start_time = time.time()
    deadline = 150.0
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency sets for quick lookup
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        C = list(range(k))
        I = list(range(k, n))
        
        cliques = []
        
        # 1. Cover cross edges and potentially some C-C edges
        # For each i in I, if it has neighbors in C, form clique {i} U N_C(i)
        # This covers all edges (i, c) for c in N_C(i).
        # It also covers edges (c1, c2) for c1, c2 in N_C(i).
        
        # Track which C-C edges are covered by these mixed cliques
        covered_cc_edges = set()
        
        for i in I:
            neighbors_in_C = [c for c in C if c in adj[i]]
            if neighbors_in_C:
                clique = [i] + neighbors_in_C
                cliques.append(clique)
                # Mark C-C edges within neighbors_in_C as covered
                for idx in range(len(neighbors_in_C)):
                    for jdx in range(idx + 1, len(neighbors_in_C)):
                        u = neighbors_in_C[idx]
                        v = neighbors_in_C[jdx]
                        covered_cc_edges.add((min(u, v), max(u, v)))
                        
        # 2. Find remaining C-C edges
        # All C-C edges are (u, v) with u, v in C, u < v
        # We need to cover those NOT in covered_cc_edges
        remaining_edges = []
        for u in range(k):
            for v in range(u + 1, k):
                if (u, v) not in covered_cc_edges:
                    remaining_edges.append((u, v))
                    
        # 3. Partition remaining_edges into minimum number of cliques
        # The vertices involved are in C.
        # A clique in this partition is a set of vertices in C such that all pairs are in remaining_edges?
        # No. The cliques we add must be cliques in the ORIGINAL graph.
        # Since C is a clique in the original graph, any subset of C is a clique in the original graph.
        # So we just need to partition the set of edges `remaining_edges` into groups,
        # such that each group forms a clique (which is always true for subsets of C),
        # and we minimize the number of groups.
        
        # Wait, if we add a clique K_subset of C, it covers ALL edges within K_subset.
        # So we need to partition the edge set `remaining_edges` into subsets,
        # where each subset is the edge set of a complete graph on some vertex set S_j.
        # This is equivalent to partitioning the vertices of the graph G_rem = (C, remaining_edges)
        # into cliques? No.
        
        # Let's re-evaluate.
        # We have a set of edges E_rem.
        # We want to cover them with cliques.
        # A clique K (subset of C) covers all edges inside K.
        # If we pick a clique K, it covers E(K).
        # We need E_rem = union of E(K_j) disjointly?
        # "Every edge must lie in exactly one listed clique."
        # So the edge sets E(K_j) must be disjoint and their union is E_rem.
        
        # This is the problem of partitioning the edges of G_rem into cliques.
        # Since G_rem is a subgraph of a complete graph, this is equivalent to finding
        # a set of vertex subsets S_1, ..., S_m such that:
        # 1. For each j, S_j is a clique in G_rem (i.e., all edges in S_j are in E_rem).
        #    Wait, if S_j is a clique in G_rem, it means all pairs in S_j are edges in G_rem.
        #    But we want to cover the edges in E_rem.
        #    If we pick S_j, we cover all edges within S_j.
        #    These edges must be a subset of E_rem.
        #    And the sets of edges covered by different S_j must be disjoint.
        #    This implies that the vertex sets S_j cannot share edges.
        #    If S_j and S_l share a vertex v, they don't share an edge unless they share 2 vertices.
        #    If they share 2 vertices u, v, then the edge (u, v) is covered by both.
        #    So the vertex sets S_j must be pairwise disjoint?
        #    No. If S_j = {1, 2, 3} and S_l = {3, 4, 5}.
        #    Edges in S_j: (1,2), (1,3), (2,3).
        #    Edges in S_l: (3,4), (3,5), (4,5).
        #    No shared edges.
        #    So S_j can share vertices, but not edges.
        #    This means the graph formed by the cliques S_j must be an edge-disjoint union.
        
        # Given the high probability that E_rem is empty, we can just return empty if E_rem is empty.
        # If E_rem is not empty, we can use a greedy approach.
        # Since k is small, we can try to find a partition.
        
        if remaining_edges:
            # Greedy: repeatedly find a maximal clique in the graph defined by remaining_edges
            # and remove its edges.
            # Note: A clique in G_rem is a set of vertices where all pairs are edges in G_rem.
            # But we want to cover edges.
            # If we pick a clique S in G_rem, we cover all edges in S.
            # We must ensure these edges are in remaining_edges.
            # And we remove them.
            
            # To be safe and fast, since E_rem is likely empty or very small:
            # Just group edges arbitrarily? No, they must form cliques.
            # Since any subset of C is a clique in the original graph,
            # we can just take each edge as a clique of size 2?
            # That would use |E_rem| cliques.
            # We can do better by grouping edges that share vertices.
            
            # Let's just use the trivial partition: one clique per edge.
            # This is valid and safe.
            # If E_rem is small, it's fine.
            for u, v in remaining_edges:
                cliques.append([u, v])
                
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline - 10:
            # Safety break, though unlikely to happen
            break
        res = solve_one(*spec)
        # Ensure types are correct
        clean_res = []
        for c in res:
            if len(c) >= 2:
                clean_res.append(list(c))
        results.append(clean_res)
        
    return results

