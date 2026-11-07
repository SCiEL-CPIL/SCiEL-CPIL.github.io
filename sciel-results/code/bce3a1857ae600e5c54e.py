"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
import heapq

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def clique_partition(edges, n, k):
        """
        Valid clique partition for split graphs using Greedy Set Cover for cross edges.
        
        Strategy:
        1. One clique for the entire K (covers all edges within K).
        2. For cross edges, use Greedy Set Cover.
           - Elements: The set of all cross edges E_cross.
           - Sets: For each vertex v (in K or I), the set of cross edges incident to v.
             If we choose vertex v, we can form a clique consisting of v and its neighbors 
             in the OTHER part (and potentially other same-part vertices if they are connected, 
             but in split graphs, independent set vertices are only connected to K).
             
             Actually, a clique containing an independent vertex v can ONLY contain v and 
             vertices from K that are adjacent to v. It cannot contain other independent vertices.
             So the "set" of edges covered by choosing a clique centered at v is exactly 
             the set of cross edges incident to v.
             
             Similarly, a clique centered at a K vertex u can contain u and all independent 
             vertices adjacent to u. The edges covered are the cross edges incident to u.
             
             However, we must be careful: if we pick a clique C_v = {v} ∪ N_K(v), it covers 
             all edges (u, v) for u in N_K(v).
             If we pick a clique C_u = {u} ∪ N_I(u), it covers all edges (u, v) for v in N_I(u).
             
             Note: Any cross edge (u, v) is covered by C_v OR C_u.
             We want to minimize the number of such cliques.
             
             Greedy Set Cover:
             1. Start with all cross edges uncovered.
             2. While uncovered edges exist:
                a. Find the vertex (u or v) that covers the most uncovered edges.
                b. Add the corresponding clique to the partition.
                c. Mark those edges as covered.
                d. Update degrees.
        """
        cliques = []
        
        # 1. The clique K
        if k >= 2:
            cliques.append(list(range(k)))
        
        # Identify cross edges
        cross_edges = []
        for u, v in edges:
            if u < k and v >= k:
                cross_edges.append((u, v))
        
        if not cross_edges:
            return cliques
        
        # Build adjacency for cross edges
        # adj_k[u] = set of v in I connected to u
        # adj_i[v] = set of u in K connected to v
        adj_k = {u: set() for u in range(k)}
        adj_i = {v: set() for v in range(k, n)}
        
        for u, v in cross_edges:
            adj_k[u].add(v)
            adj_i[v].add(u)
        
        # Uncovered edges
        uncovered = set(cross_edges)
        
        # To speed up greedy, we can pre-calculate potential coverage.
        # But dynamic updates are needed.
        # For n=90, k=30, cross edges ~ 30*60*0.7 = 1260.
        # Vertices involved: 90.
        # Greedy steps: at most 1260.
        # Each step scans 90 vertices and counts covered edges.
        # 1260 * 90 * (avg degree) ~ 1260 * 90 * 20 ~ 2.2M ops. Very fast.
        
        while uncovered:
            best_vertex = -1
            best_count = 0
            best_is_k = False # False means independent set vertex
            
            # Check K vertices
            for u in range(k):
                # Count how many uncovered edges are incident to u
                # Edge is (u, v). u is in K.
                count = 0
                for v in adj_k[u]:
                    if (u, v) in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_vertex = u
                    best_is_k = True
            
            # Check I vertices
            for v in range(k, n):
                # Count how many uncovered edges are incident to v
                # Edge is (u, v). v is in I.
                count = 0
                for u in adj_i[v]:
                    if (u, v) in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_vertex = v
                    best_is_k = False
            
            if best_count == 0:
                # Should not happen if uncovered is not empty
                break
            
            # Add the clique
            if best_is_k:
                # Centered at u in K. Clique is {u} ∪ N_I(u)
                # But we only need to list vertices that form a clique.
                # u is connected to all v in N_I(u). 
                # Are the v's connected to each other? No, they are in I.
                # So the clique is {u} ∪ N_I(u).
                # Wait, is {u} ∪ N_I(u) a clique?
                # u is connected to all v in N_I(u).
                # v1, v2 in N_I(u) are NOT connected to each other (independent set).
                # So {u, v1, v2} is NOT a clique if v1 != v2 and v1, v2 not connected.
                # Ah! A clique cannot contain two independent vertices.
                # So a clique containing a K vertex u can ONLY contain u and AT MOST ONE independent vertex?
                # No, that's not right.
                # Definition of Clique: Every pair of vertices in the set must be connected.
                # In a split graph, I is an independent set. So no two vertices in I are connected.
                # Therefore, any clique can contain at most ONE vertex from I.
                # Consequently, any clique containing a vertex v from I is of the form {v} ∪ S_u, where S_u ⊆ N_K(v).
                # And since all u in S_u are in K, they are all connected to each other and to v.
                # So {v} ∪ N_K(v) IS a valid clique.
                
                # Now, what about cliques centered at K?
                # A clique containing u from K can contain other vertices from K (any subset) and AT MOST ONE vertex from I.
                # If it contains only K vertices, it's a subset of K. But we already have the big K clique.
                # Should we use subsets of K? 
                # The big K clique covers all edges within K.
                # Cross edges (u, v) require v to be in the clique.
                # Since v is in I, the clique MUST contain v.
                # Since it contains v, it can contain at most v from I.
                # So any clique covering a cross edge (u, v) MUST contain v.
                # Therefore, we can restrict our set cover to cliques that contain exactly one vertex from I.
                # Such a clique is {v} ∪ S, where S ⊆ N_K(v).
                # To cover edge (u, v), u must be in S.
                # So the set of edges covered by clique {v} ∪ S is {(u, v) | u ∈ S}.
                # To maximize coverage for a chosen v, we should choose S = N_K(v).
                # Then the clique is {v} ∪ N_K(v), and it covers ALL edges incident to v.
                
                # So, my previous logic was flawed for K-centered cliques.
                # We don't need K-centered cliques for cross edges, because any cross edge (u, v) 
                # must be in a clique containing v (since v is in I and can't be in a clique with other I vertices, 
                # and if the clique doesn't contain v, it doesn't cover the edge).
                # Wait, could a clique contain u and v? Yes.
                # But it MUST contain v.
                # So the "set" in set cover is always associated with a vertex v in I.
                # The set of edges covered by the clique C_v = {v} ∪ N_K(v) is all edges incident to v.
                
                # So the problem reduces to:
                # Cover all cross edges using cliques C_v for v in I.
                # This is equivalent to covering all edges in the bipartite graph using stars centered at I.
                # This is the Minimum Edge Cover problem on a bipartite graph?
                # No, Minimum Edge Cover usually covers vertices. Here we cover edges.
                # But wait, if we pick C_v, it covers ALL edges incident to v.
                # So if we pick a set of vertices V' ⊆ I, the edges covered are all edges incident to V'.
                # We need to cover ALL cross edges.
                # This means every cross edge (u, v) must have v ∈ V'.
                # So V' must contain the endpoint in I for EVERY cross edge.
                # That means V' must be ALL vertices in I that have degree > 0.
                # If that's the case, the number of cliques is just the number of non-isolated vertices in I.
                
                # Let's re-read the problem carefully.
                # "Every listed set must be a clique... every edge must lie in exactly one listed clique."
                # If I pick C_v = {v} ∪ N_K(v), it covers all edges (u, v).
                # If I pick C_w = {w} ∪ N_K(w), it covers all edges (u, w).
                # If an edge (u, v) is covered by C_v, is it covered by any other clique?
                # No, because no other clique contains v (other independent vertices are not connected to v, 
                # and K-cliques don't contain v).
                # So the cliques C_v are disjoint in terms of edges?
                # Yes, because each edge (u, v) is uniquely associated with v.
                # So if I use C_v for all v with deg(v) > 0, I cover all edges exactly once.
                # The number of cliques is |{v ∈ I | deg(v) > 0}|.
                
                # Can I do better?
                # What if I use a smaller clique? E.g., {u, v} for a specific edge (u, v).
                # This covers only (u, v).
                # This is worse than C_v which covers all edges for v.
                # So C_v is always at least as good as any subset.
                
                # So the optimal solution is simply:
                # 1. One clique for K.
                # 2. One clique C_v = {v} ∪ N_K(v) for each v in I with deg(v) > 0.
                
                # Let's verify this logic.
                # Is it possible to combine edges from different v's?
                # No, because a clique can contain at most one vertex from I.
                # So any clique covering an edge (u, v) MUST contain v.
                # So the set of cliques covering edges incident to v must all contain v.
                # But if two cliques both contain v, say C1 and C2, and they cover different edges (u1, v) and (u2, v),
                # we can merge them into C1 ∪ C2 = {v} ∪ {u1, u2} ∪ ...
                # Since u1, u2 ∈ K, they are connected to each other and to v.
                # So the union is also a clique.
                # And it covers the union of edges.
                # So it is always optimal to merge all cliques containing the same v into one big clique C_v.
                
                # Therefore, the minimum number of cliques for cross edges is exactly the number of 
                # vertices in I that are incident to at least one cross edge.
                
                # Let's check the current program's score.
                # Current: 1 (K) + E_cross.
                # New: 1 (K) + |I_non_isolated|.
                # E_cross is roughly k * (n-k) * p.
                # |I_non_isolated| is roughly (n-k) * (1 - (1-p)^k).
                # For p=0.7, k=10, n=30:
                # E_cross ~ 10 * 20 * 0.7 = 140.
                # |I_non_isolated| ~ 20 * (1 - 0.3^10) ~ 20.
                # Huge improvement!
                
                # Why did the current program use {u, v} cliques?
                # Probably because it didn't realize C_v is a valid clique or that it's optimal.
                # C_v = {v} ∪ N_K(v) is a clique because:
                # 1. v is connected to all u in N_K(v).
                # 2. All u in N_K(v) are in K, so they are connected to each other.
                # Thus, C_v is a clique.
                
                # So the strategy is:
                # 1. Add K as a clique.
                # 2. For each v in I, if deg(v) > 0, add {v} ∪ N_K(v) as a clique.
                
                pass # We will implement this simpler logic instead of greedy.
            
            else:
                # Centered at v in I. Clique is {v} ∪ N_K(v)
                clique_verts = [best_vertex] + list(adj_i[best_vertex])
                cliques.append(clique_verts)
                # Remove covered edges
                for u in adj_i[best_vertex]:
                    uncovered.discard((u, best_vertex))
            
            # If we implemented the simple logic, we wouldn't be in this loop.
            # Let's just implement the simple logic directly.
            return None # Placeholder to force rewrite

    # Rewrite clique_partition with the simple optimal logic
    def clique_partition_optimal(edges, n, k):
        cliques = []
        if k >= 2:
            cliques.append(list(range(k)))
        
        # Find non-isolated vertices in I
        adj_i = {v: set() for v in range(k, n)}
        for u, v in edges:
            if u < k and v >= k:
                adj_i[v].add(u)
        
        for v in range(k, n):
            if adj_i[v]:
                clique = [v] + list(adj_i[v])
                cliques.append(clique)
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = clique_partition_optimal(edges, n, k)
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
