"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    _deadline = time.time() + 140.0  # Safety margin

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_one(n, k, p, seed):
        # 1. Generate edges
        # Cross edges: (u, v) where u < k <= v
        # Internal K edges: (u, v) where u < v < k
        cross_edges = []
        rng = Random(seed)
        # We need to replicate the exact random sequence of instance()
        # instance() generates internal K edges first (deterministic), then cross edges.
        # The random calls are ONLY for cross edges in the order: v from k to n-1, u from 0 to k-1.
        
        # Let's verify the order in instance():
        # edges = [(u, v) for u in range(k) for v in range(u + 1, k)]  # No random
        # edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        # So random calls are for each (v, u) pair in that nested loop order.
        
        # We don't strictly need to call instance() if we generate the same data, 
        # but keeping the function is required. We can use it to get edges, 
        # but parsing them is slower. Let's generate directly to be fast, 
        # but we must ensure the random sequence matches.
        
        # Actually, let's just call instance() to be safe and parse.
        # n is small (<=90), k<=30. Edges ~ 30^2/2 + 30*60 ~ 450 + 1800 = 2250.
        # Parsing 2250 edges is fast.
        
        edges = instance(n, k, p, seed)
        
        # Separate edges
        k_edges = []
        cross = []
        for u, v in edges:
            if u < k and v < k:
                k_edges.append((u, v))
            else:
                # u < k <= v is guaranteed by generation logic
                cross.append((u, v))
        
        # Data structures for K edges
        # rem_adj[u] is a bitmask of neighbors of u in the remaining K-edge graph
        rem_adj = [0] * k
        for u, v in k_edges:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
        
        # Helper to find a maximal clique in a candidate set (bitmask) 
        # using the current rem_adj
        def find_maximal_clique(candidates_mask):
            # Greedy: repeatedly pick a vertex with high degree in the induced subgraph
            # and intersect candidates with its neighbors.
            # This finds a maximal clique.
            clique = []
            current_mask = candidates_mask
            
            while current_mask:
                # Pick a vertex. To be a bit smarter, pick one with max degree in current_mask
                # But for speed, picking the lowest set bit is O(1) amortized if we just pop.
                # Let's try to pick a vertex that connects to many others to make the clique larger?
                # Actually, for maximal clique, any order works. 
                # To get a *larger* clique, we should pick vertices with high degree.
                
                best_v = -1
                best_deg = -1
                # Iterate through set bits
                m = current_mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    # Degree of v in current_mask is popcount(current_mask & rem_adj[v])
                    deg = (current_mask & rem_adj[v]).bit_count()
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                    m ^= lsb
                
                if best_v == -1:
                    break
                    
                clique.append(best_v)
                # Intersect with neighbors of best_v
                current_mask &= rem_adj[best_v]
                # Remove best_v from current_mask (it's not in rem_adj[best_v] anyway)
                
            return clique

        result = []
        
        # 2. Process Cross Edges
        # Group cross edges by v (the vertex in I)
        neighbors_of_v = {v: [] for v in range(k, n)}
        for u, v in cross:
            neighbors_of_v[v].append(u)
        
        for v in range(k, n):
            if not neighbors_of_v[v]:
                continue
            
            # Candidates are neighbors of v in K
            # We need to form cliques {v} U S where S is a clique in G_rem
            # S must be subset of neighbors_of_v[v]
            
            # Start with all neighbors
            candidates_mask = 0
            for u in neighbors_of_v[v]:
                candidates_mask |= (1 << u)
            
            while candidates_mask:
                # Check time
                if time.time() > _deadline:
                    # Fallback: add edges individually
                    # This is unlikely to happen if we are efficient
                    for u in neighbors_of_v[v]:
                        if (1 << u) & candidates_mask:
                            result.append([v, u])
                            # Mark edge (u, u) ? No, just consume cross edge.
                            # We don't need to update rem_adj for cross edges.
                            # But we must ensure we don't double add.
                            # Since we break out, we just add remaining as pairs.
                            pass
                    break
                
                # Find maximal clique S in candidates_mask w.r.t rem_adj
                S = find_maximal_clique(candidates_mask)
                
                if not S:
                    # Should not happen if candidates_mask != 0 and rem_adj is valid
                    # But if rem_adj has no edges, S will be singletons?
                    # find_maximal_clique picks one vertex, then current_mask becomes 0.
                    # So S will have at least 1 vertex.
                    pass
                
                # Add clique {v} U S
                clique_vertices = [v] + S
                result.append(clique_vertices)
                
                # Consume internal edges of K formed by S
                # For all pairs (u, w) in S, remove from rem_adj
                for i in range(len(S)):
                    u = S[i]
                    for j in range(i + 1, len(S)):
                        w = S[j]
                        # Remove edge (u, w)
                        rem_adj[u] &= ~(1 << w)
                        rem_adj[w] &= ~(1 << u)
                
                # Remove S from candidates_mask
                for u in S:
                    candidates_mask &= ~(1 << u)
        
        # 3. Cover Remaining K Edges
        # rem_adj now represents the graph of remaining edges in K
        # We need to partition these edges into cliques.
        # Since any subset of K is a clique in the original graph, 
        # we just need to find cliques in the graph (K, rem_edges).
        
        # Build list of vertices that still have incident edges
        active_vertices = [u for u in range(k) if rem_adj[u] != 0]
        
        # We need to cover all edges.
        # Greedy: While there are edges, find a maximal clique in the remaining graph, 
        # add it, and remove its edges.
        
        # To avoid infinite loops or missing edges, we track remaining edges explicitly or 
        # rely on the adjacency matrix.
        
        # Let's iterate while any edge exists.
        # An edge (u, v) exists if rem_adj[u] has bit v set.
        
        # To find a maximal clique, we can use the same helper, but we need to ensure
        # we pick a clique that covers *some* edges.
        
        # We can just pick a vertex u with rem_adj[u] != 0, 
        # and find a maximal clique containing u?
        # Or just find a maximal clique in the whole remaining graph.
        
        # Let's collect all vertices with degree > 0
        # And repeatedly extract maximal cliques.
        
        # We need to be careful: find_maximal_clique uses rem_adj.
        # If we pass a mask of all active vertices, it will find a clique.
        
        # To ensure progress, we can just pick any edge (u, v) and force u into the clique?
        # Simpler: Just find a maximal clique in the induced subgraph of active vertices.
        
        # However, if we just call find_maximal_clique(all_active_mask), it might return a clique.
        # Then we remove its edges.
        
        # Let's do this:
        while True:
            if time.time() > _deadline:
                # Fallback: add all remaining edges as pairs
                # This is safe but suboptimal
                # We need to identify remaining edges.
                # Iterate u < v
                for u in range(k):
                    for v in range(u + 1, k):
                        if (rem_adj[u] >> v) & 1:
                            result.append([u, v])
                            rem_adj[u] &= ~(1 << v)
                            rem_adj[v] &= ~(1 << u)
                break
            
            # Find any vertex with degree > 0
            active_mask = 0
            for u in range(k):
                if rem_adj[u]:
                    active_mask |= (1 << u)
            
            if active_mask == 0:
                break
            
            # Find maximal clique in active_mask
            # To ensure we pick a clique that actually has edges, 
            # the clique must have size >= 2? 
            # A clique of size 1 has no edges.
            # find_maximal_clique returns a set of vertices that form a clique.
            # If it returns [u], it covers no edges. We must avoid this.
            
            # Modification: Find maximal clique with size >= 2.
            # Or, pick an edge (u, v) and find maximal clique containing u and v?
            
            # Let's pick an edge (u, v) from active_mask.
            # Find u such that rem_adj[u] != 0.
            u = -1
            m = active_mask
            while m:
                lsb = m & -m
                candidate = lsb.bit_length() - 1
                if rem_adj[candidate]:
                    u = candidate
                    break
                m ^= lsb
            
            if u == -1:
                break
                
            # Pick a neighbor v of u
            v_mask = rem_adj[u] & active_mask
            if not v_mask:
                # Should not happen if rem_adj[u] is non-zero and active_mask covers all
                break
            
            lsb_v = v_mask & -v_mask
            v = lsb_v.bit_length() - 1
            
            # We want a clique containing u and v.
            # Candidates must be neighbors of both u and v.
            # candidates = active_mask & rem_adj[u] & rem_adj[v]
            # Plus u and v themselves.
            
            # Let's just find a maximal clique in the intersection of neighbors, 
            # then add u and v.
            
            common_neighbors = rem_adj[u] & rem_adj[v]
            # We can add any subset of common_neighbors that forms a clique with u and v.
            # Since u and v are connected to all in common_neighbors, 
            # we just need the subset to be a clique among themselves.
            
            # Use find_maximal_clique on common_neighbors?
            # find_maximal_clique uses rem_adj.
            # If we pass common_neighbors, it will find a clique S in common_neighbors.
            # Then {u, v} U S is a clique.
            
            S = find_maximal_clique(common_neighbors)
            
            clique_vertices = [u, v] + S
            
            result.append(clique_vertices)
            
            # Remove edges in this clique
            # Edges are (u, v), (u, s), (v, s), (s1, s2)
            # (u, v)
            rem_adj[u] &= ~(1 << v)
            rem_adj[v] &= ~(1 << u)
            
            for s in S:
                # (u, s)
                rem_adj[u] &= ~(1 << s)
                rem_adj[s] &= ~(1 << u)
                # (v, s)
                rem_adj[v] &= ~(1 << s)
                rem_adj[s] &= ~(1 << v)
                # (s, s2) handled by pairs in S
                for s2 in S:
                    if s < s2:
                        rem_adj[s] &= ~(1 << s2)
                        rem_adj[s2] &= ~(1 << s)
                        
            # Note: find_maximal_clique might return empty if common_neighbors is 0.
            # In that case, clique is {u, v}. This is valid (size 2).
            
        return result

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > _deadline - 5:
            break
        results.append(solve_one(*spec))
    
    return results
# EVOLVE-BLOCK-END
