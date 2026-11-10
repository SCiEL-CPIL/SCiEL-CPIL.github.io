"""Clique partition of split graphs using internal-first structural decomposition."""
import time
from random import Random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        # 1. Separate internal and cross edges
        internal_edges = set()
        cross_edges = set()
        # Adjacency for cross edges: i -> set of c
        adj_i_to_c = defaultdict(set)
        adj_c_to_i = defaultdict(set)
        
        for u, v in edges:
            if u < k and v < k:
                internal_edges.add((u, v) if u < v else (v, u))
            elif (u < k and v >= k) or (u >= k and v < k):
                if u < k:
                    c, i = u, v
                else:
                    c, i = v, u
                cross_edges.add((c, i))
                adj_i_to_c[i].add(c)
                adj_c_to_i[c].add(i)
        
        cliques = []
        
        # 2. Partition internal clique C (vertices 0..k-1) into cliques
        # We want to cover all internal_edges with a set of cliques S_1...S_m
        # Greedy: Find max clique in remaining internal edges?
        # Since C is a complete graph, any subset is a clique.
        # We want to partition the EDGE SET of the complete graph K_k.
        # This is equivalent to finding a clique partition of K_k.
        
        # To minimize the number of cliques, we should make them as large as possible.
        # However, we also want to attach independent vertices to them.
        # Let's just generate a standard clique partition of K_k.
        # A simple partition: 
        # If k is small, we can use a known optimal or near-optimal partition.
        # For now, use a greedy approach:
        # Pick a vertex, add as many as possible? 
        # Actually, the number of cliques is determined by the partition of edges.
        # Let's use a recursive greedy:
        # 1. Pick a vertex u.
        # 2. Find the largest set S containing u such that all edges in S are uncovered?
        #    No, we are partitioning the edges of K_k.
        #    We can just iterate through edges and build cliques.
        
        # Simpler: Since C is a clique, we can just pick arbitrary subsets.
        # To allow attachment of I, we want S to be a subset of N(i) for some i.
        
        # Let's try a different tactic:
        # Iterate through I vertices. For each i, find a large clique S in C 
        # such that S subset N(i) and edges in S are not yet covered.
        # This is the "Cross-First" approach but optimized.
        
        # Let's stick to the "Internal-First" idea but make it work.
        # 1. Cover internal edges with cliques S_1...S_m.
        #    We can use a simple partition: 
        #    For k=30, k^2/6 ~ 150 cliques.
        #    We can generate these cliques arbitrarily, e.g., using a greedy edge-cover.
        #    Let's use a greedy algorithm to partition the edges of K_k.
        
        remaining_internal = set(internal_edges)
        internal_cliques = []
        
        # To speed up, we can just use pairs if k is large? No, larger is better for attaching I.
        # Let's try to make cliques of size ~ k/2?
        
        # Greedy Max Clique on the graph of remaining edges?
        # The graph of remaining edges is a subgraph of K_k.
        # We want to find a clique in this subgraph.
        
        while remaining_internal:
            if time.time() > deadline:
                break
            # Pick a vertex with highest degree in remaining_internal
            deg = defaultdict(int)
            for u, v in remaining_internal:
                deg[u] += 1
                deg[v] += 1
            
            if not deg:
                break
                
            # Start with vertex with max degree
            start_v = max(deg, key=lambda x: deg[x])
            S = {start_v}
            
            # Try to add other vertices
            # A vertex v can be added to S if all edges (v, u) for u in S are in remaining_internal
            # This ensures the new clique's edges are a subset of remaining_internal.
            # We want to maximize |S|.
            
            candidates = list(deg.keys())
            # Sort candidates by degree descending
            candidates.sort(key=lambda x: deg[x], reverse=True)
            
            for v in candidates:
                if v in S:
                    continue
                # Check if v can be added
                can_add = True
                for u in S:
                    edge = (u, v) if u < v else (v, u)
                    if edge not in remaining_internal:
                        can_add = False
                        break
                if can_add:
                    S.add(v)
            
            # Add this clique
            internal_cliques.append(list(S))
            # Remove edges covered by this clique
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    u, v = S_list[i], S_list[j]
                    edge = (u, v) if u < v else (v, u)
                    remaining_internal.discard(edge)
        
        # 3. For each internal clique S, try to attach an independent vertex i
        #    We want to choose i such that i is connected to all v in S.
        #    To minimize total cliques, we prefer i that covers many edges?
        #    Actually, each attachment reduces the number of 2-cliques needed for cross edges.
        #    So we just need ANY i.
        
        # However, if we attach i to S, we cover cross edges (i, v) for v in S.
        # If we don't, we need separate cliques for those cross edges.
        
        # Let's sort internal_cliques by size descending to attach high-degree i first?
        # Or just iterate.
        
        # We need to track which cross edges are covered.
        covered_cross = set()
        
        # For each S in internal_cliques:
        #   Find i in I such that N(i) superset S.
        #   If found, add clique {i} U S. Mark (i, v) for v in S as covered.
        #   If not, add clique S. (This covers internal edges, already accounted for).
        
        # To be efficient, for each S, we can precompute the set of i's that can attach.
        # Since k is small, S is small.
        
        for S in internal_cliques:
            if time.time() > deadline:
                break
            S_set = set(S)
            # Find i such that S_set is subset of adj_i_to_c[i]
            # We can just iterate through I vertices?
            # n-k is at most 60. S size is at most k.
            
            best_i = -1
            # Prefer i with high degree? Or just first found?
            # Let's pick the i with the largest degree to "use up" high-degree vertices?
            # Or smallest degree to save high-degree ones?
            # Let's just pick the first one found for simplicity.
            
            for i in range(k, n):
                if S_set.issubset(adj_i_to_c[i]):
                    best_i = i
                    break
            
            if best_i != -1:
                cliques.append([best_i] + S)
                for v in S:
                    covered_cross.add((v, best_i))
            else:
                # No i can attach to this S
                # Add S as a clique (covers internal edges)
                if len(S) >= 2:
                    cliques.append(S)
                # Else, S is a single vertex, no internal edges covered?
                # Wait, if S is size 1, it covers no internal edges.
                # But we generated S from remaining_internal edges, so S should be size >= 2?
                # If remaining_internal was not empty, S must contain at least one edge, so size >= 2.
        
        # 4. Cover remaining cross edges
        #    These are edges (c, i) not in covered_cross.
        
        #    We can group them into cliques?
        #    A clique can contain at most one i (since I is independent).
        #    So a clique is {i} U S' where S' subset N(i).
        #    We want to cover the remaining edges incident to i.
        #    For each i, let R_i = { c | (c, i) is a remaining cross edge }.
        #    We can form cliques {i} U S where S is a subset of R_i.
        #    To minimize cliques, we should make S as large as possible.
        #    But S must be a clique in C? Yes, C is a clique, so any S is a clique.
        #    However, we must ensure that internal edges in S are NOT covered by other cliques.
        #    Wait, the internal edges in S are already covered by the internal_cliques partition?
        #    YES. The internal_cliques partition covered ALL internal edges.
        #    So if we form a new clique {i} U S, it will cover internal edges in S AGAIN.
        #    This violates the "exactly one" rule.
        
        #    Therefore, we CANNOT form cliques {i} U S with |S| > 1 for the remaining cross edges,
        #    unless we modify the internal partition to NOT cover those edges.
        
        #    This implies the "Internal-First" approach is tricky if we want to reuse I vertices for multiple cross edges.
        #    If we use {i} U S, we cover internal edges in S.
        #    If those internal edges were already covered by an internal clique, we have a conflict.
        
        #    So, for the remaining cross edges, we MUST use cliques of size 2: {i, c}.
        #    Or, we must ensure that the internal edges in S are NOT covered by the internal partition.
        
        #    Given the complexity, let's just use 2-cliques for remaining cross edges.
        #    This is safe and valid.
        
        for c, i in cross_edges:
            if (c, i) not in covered_cross:
                if time.time() > deadline:
                    break
                cliques.append([c, i])
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = partition_split_graph(n, k, edges)
        # Ensure types are Python int and list
        cleaned_cliques = []
        for c in cliques:
            cleaned_cliques.append([int(x) for x in c])
        results.append(cleaned_cliques)
    
    return results
# EVOLVE-BLOCK-END
