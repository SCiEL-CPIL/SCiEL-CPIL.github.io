"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross_edges = [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges), cross_edges

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        c_edges, cross_edges = instance(n, k, p, seed)
        
        # C is 0..k-1, I is k..n-1
        # Build adjacency for cross edges
        # N(v) for v in I
        neighbors_in_C = {v: [] for v in range(k, n)}
        for u, v in cross_edges:
            # u is in C, v is in I
            neighbors_in_C[v].append(u)
        
        cliques = []
        # Track covered internal edges in C
        covered_internal = set()
        
        for v in range(k, n):
            if neighbors_in_C[v]:
                clique = [v] + neighbors_in_C[v]
                cliques.append(clique)
                # Mark internal edges covered by this clique
                nbrs = neighbors_in_C[v]
                for i in range(len(nbrs)):
                    for j in range(i + 1, len(nbrs)):
                        u1, u2 = nbrs[i], nbrs[j]
                        if u1 > u2:
                            u1, u2 = u2, u1
                        covered_internal.add((u1, u2))
        
        # Determine remaining internal edges in C
        # All edges in C are (u, v) with 0 <= u < v < k
        all_internal = set((u, v) for u in range(k) for v in range(u + 1, k))
        remaining_internal = all_internal - covered_internal
        
        if remaining_internal:
            # Need to cover remaining_internal with cliques in C
            # Greedy approach: 
            # 1. Find a clique in G_rem (subgraph of C with edges remaining_internal)
            # 2. Add it, remove its edges, repeat.
            
            # To find a clique in G_rem, we can look for triangles, etc.
            # Since k is small, we can just use a simple heuristic.
            # Actually, just adding the edges as cliques is safe but many.
            # Better: Group edges into maximal cliques.
            
            rem_edges = list(remaining_internal)
            # Sort edges to be deterministic
            rem_edges.sort()
            
            # Simple greedy: pick first edge, find maximal clique containing it.
            # Since G_rem is a subgraph of a clique, any subset is a clique if all edges present.
            # We need to find sets of vertices where ALL pairs are in rem_edges.
            
            # Heuristic: 
            # Try to form large cliques.
            # Start with a vertex, add neighbors in G_rem.
            
            # Build adjacency for G_rem
            adj_rem = {u: set() for u in range(k)}
            for u, v in rem_edges:
                adj_rem[u].add(v)
                adj_rem[v].add(u)
                
            used_edges = set()
            
            while rem_edges:
                if time.time() > deadline:
                    # Fallback: add all remaining as pairs
                    for u, v in rem_edges:
                        if (u, v) not in used_edges:
                            cliques.append([u, v])
                            used_edges.add((u, v))
                    break
                
                # Pick an edge (u, v)
                u, v = rem_edges[0]
                
                # Find maximal clique containing u and v in G_rem
                # Common neighbors of u and v in G_rem
                common = adj_rem[u] & adj_rem[v]
                
                # Try to add more vertices
                # Greedy: add vertices from common that are connected to all current clique members
                current_clique = {u, v}
                candidates = list(common)
                
                # Sort candidates by degree in G_rem to prefer high degree
                candidates.sort(key=lambda x: len(adj_rem[x]), reverse=True)
                
                for w in candidates:
                    # Check if w is connected to all in current_clique
                    if all(w in adj_rem[x] for x in current_clique):
                        current_clique.add(w)
                        # Update common? No, just check against current_clique.
                        # Note: adding w might allow adding others?
                        # For simplicity, we just do one pass.
                        
                # Convert to list
                clique_list = list(current_clique)
                if len(clique_list) < 2:
                    clique_list = [u, v]
                    
                cliques.append(clique_list)
                
                # Remove edges covered by this clique
                edges_to_remove = set()
                for i in range(len(clique_list)):
                    for j in range(i + 1, len(clique_list)):
                        a, b = clique_list[i], clique_list[j]
                        if a > b:
                            a, b = b, a
                        edges_to_remove.add((a, b))
                        
                # Filter rem_edges
                rem_edges = [e for e in rem_edges if e not in edges_to_remove]
                # Update used_edges? Not strictly needed if we filter rem_edges.
                
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
