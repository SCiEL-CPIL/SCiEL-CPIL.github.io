"""Combined program with optimized split-graph clique partition."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed):
        start_time = time.time()
        deadline = start_time + 110  # Safety margin for 120s limit
        
        rng = Random(seed)
        
        # Generate cross edges and neighborhoods
        # N[v] is a set of neighbors in C for v in I
        N = [set() for _ in range(n)]
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
                    N[v].add(u)
        
        cliques = []
        
        # Step 1: Greedily select active vertices in I
        # Active v uses clique {v} U N[v]
        # Inactive v uses individual cliques {v, u}
        # Constraint: No internal edge (u1, u2) in C is covered by two active v's.
        # i.e., if v1, v2 are active, |N[v1] intersection N[v2]| <= 1
        
        covered_pairs = set()
        active_vertices = []
        inactive_vertices = []
        
        # Order vertices by degree (larger first) to maximize coverage?
        # Or just iterate.
        for v in range(k, n):
            if time.time() > deadline:
                # Fallback to simple solution if running out of time
                # Just use simple heuristic for remaining
                break
            
            # Check if N[v] conflicts with covered_pairs
            # Conflict if there exists u1, u2 in N[v] such that (min(u1,u2), max(u1,u2)) is in covered_pairs
            conflict = False
            nv_list = list(N[v])
            for i in range(len(nv_list)):
                for j in range(i+1, len(nv_list)):
                    u1, u2 = nv_list[i], nv_list[j]
                    pair = (min(u1, u2), max(u1, u2))
                    if pair in covered_pairs:
                        conflict = True
                        break
                if conflict:
                    break
            
            if not conflict:
                active_vertices.append(v)
                # Add all pairs in N[v] to covered_pairs
                for i in range(len(nv_list)):
                    for j in range(i+1, len(nv_list)):
                        u1, u2 = nv_list[i], nv_list[j]
                        covered_pairs.add((min(u1, u2), max(u1, u2)))
            else:
                inactive_vertices.append(v)
        
        # If we broke early due to time, handle remaining vertices as inactive
        # This logic is a bit loose, let's just assume we processed all or use simple fallback
        # For robustness, if we didn't process all, let's just classify unprocessed as inactive
        processed = set(active_vertices) | set(inactive_vertices)
        for v in range(k, n):
            if v not in processed:
                inactive_vertices.append(v)

        # Step 2: Add cliques for active and inactive vertices
        for v in active_vertices:
            cliques.append([v] + list(N[v]))
        
        for v in inactive_vertices:
            for u in N[v]:
                cliques.append([u, v])
        
        # Step 3: Cover remaining internal edges in C
        # Total internal edges in C
        all_internal_edges = set()
        for u in range(k):
            for w in range(u+1, k):
                all_internal_edges.add((u, w))
        
        remaining_edges = all_internal_edges - covered_pairs
        
        if remaining_edges:
            # We need to cover remaining_edges with cliques from C
            # Such that no edge in the clique is already covered (which is guaranteed 
            # if we only pick edges from remaining_edges? No.
            # If we pick a clique S, it covers ALL edges in S.
            # We must ensure that ALL edges in S are in remaining_edges.
            # This means S must be a clique in the graph G_rem = (C, remaining_edges).
            
            # Greedy clique cover on G_rem
            # G_rem is a graph on vertices 0..k-1
            # Adjacency in G_rem: u~w if (u,w) in remaining_edges
            
            rem_adj = {u: set() for u in range(k)}
            for u, w in remaining_edges:
                rem_adj[u].add(w)
                rem_adj[w].add(u)
            
            remaining_edges_copy = set(remaining_edges)
            
            while remaining_edges_copy:
                if time.time() > deadline:
                    # Emergency fallback: just add individual edges as cliques
                    # This is valid but suboptimal
                    for u, w in remaining_edges_copy:
                        cliques.append([u, w])
                    break
                
                # Find a maximal clique in G_rem induced by remaining_edges_copy
                # Simple heuristic: pick a vertex with highest degree in remaining graph, 
                # then greedily add neighbors that are connected to all previously added
                # Or just find the largest clique (k is small, up to 30)
                
                # Let's try to find a large clique
                # Start with the vertex with max degree in the remaining subgraph
                degrees = {u: len([w for w in rem_adj[u] if (min(u,w), max(u,w)) in remaining_edges_copy]) for u in range(k)}
                if not degrees or max(degrees.values()) == 0:
                    # No edges left? Should not happen if remaining_edges_copy is not empty
                    # But if it's a matching, degrees are 1.
                    # Just pick any edge
                    u, w = next(iter(remaining_edges_copy))
                    clique = [u, w]
                else:
                    start_u = max(degrees, key=degrees.get)
                    clique = [start_u]
                    candidates = list(rem_adj[start_u])
                    # Filter candidates to those connected to start_u in remaining graph
                    candidates = [c for c in candidates if (min(start_u, c), max(start_u, c)) in remaining_edges_copy]
                    
                    for c in candidates:
                        # Check if c is connected to all in clique
                        is_connected = True
                        for u_in_clique in clique:
                            if (min(c, u_in_clique), max(c, u_in_clique)) not in remaining_edges_copy:
                                is_connected = False
                                break
                        if is_connected:
                            clique.append(c)
                
                # Add this clique
                cliques.append(clique)
                
                # Remove edges covered by this clique from remaining_edges_copy
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        u, w = clique[i], clique[j]
                        remaining_edges_copy.discard((min(u, w), max(u, w)))
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = solve_split(n, k, p, seed)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
