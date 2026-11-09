"""Clique partitions of three fixed split (chordal) graphs."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random

    t0 = time.time()
    DEADLINE = t0 + 110.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def partition_cliques(n, k, p, seed):
        edges = instance(n, k, p, seed)
        cliques = []
        
        # 1. Compute neighborhoods of I vertices in C
        # N_I[u] will be the set of v in I connected to u in C
        # N_C[v] will be the set of u in C connected to v in I
        N_I = [set() for _ in range(k)]
        N_C = [set() for _ in range(n)]
        
        for v in range(k, n):
            for u in range(k):
                if (u, v) in edges:
                    N_I[u].add(v)
                    N_C[v].add(u)
        
        # 2. Create mixed cliques for each v in I
        # Using the maximal neighborhood N_C[v] ensures we cover as many C-C edges as possible in these cliques.
        for v in range(k, n):
            if N_C[v]:
                clique = sorted(list(N_C[v])) + [v]
                cliques.append(clique)
            else:
                # If v is isolated, it has no edges to cover. 
                # But the problem says "every edge must lie in exactly one listed clique".
                # If v has no edges, it doesn't need to be in any clique.
                # However, the constraint "Every listed set must be a clique with at least two vertices"
                # implies we only list cliques that exist.
                # So we skip isolated v.
                pass

        # 3. Determine remaining C-C edges
        # An edge (u, w) in C is covered by a mixed clique if there exists v in I such that u, w in N_C[v].
        # This is equivalent to N_I[u] intersect N_I[w] is not empty.
        # So remaining edges are those where N_I[u] intersect N_I[w] is empty.
        
        rem_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) in edges:
                    if not (N_I[u] & N_I[w]):
                        rem_edges.add((u, w))
        
        # 4. Cover remaining C-C edges with pure C-cliques
        # We need a minimum edge clique cover of the graph G_rem = (C, rem_edges)
        
        if rem_edges:
            # Build adjacency for G_rem
            adj = {u: set() for u in range(k)}
            for u, w in rem_edges:
                adj[u].add(w)
                adj[w].add(u)
            
            remaining_edges = set(rem_edges)
            
            while remaining_edges:
                if time.time() > DEADLINE:
                    break
                
                # Find a maximal clique in the graph defined by remaining_edges
                # Heuristic: Pick a vertex with high degree in the remaining graph
                # to help form larger cliques.
                
                # Calculate degrees in the subgraph induced by remaining_edges
                deg = {u: 0 for u in range(k)}
                for u, w in remaining_edges:
                    deg[u] += 1
                    deg[w] += 1
                
                # Pick start vertex
                # If no edges, break (should not happen)
                if not remaining_edges:
                    break
                
                # Find a vertex involved in an edge
                start_u = next(iter(remaining_edges))[0]
                
                # Greedy clique expansion
                # Start with an edge to ensure we cover at least one edge
                # Find a neighbor of start_u
                neighbors = [w for w in adj[start_u] if (min(start_u, w), max(start_u, w)) in remaining_edges]
                if not neighbors:
                    # Should not happen if start_u is from remaining_edges
                    continue
                start_w = neighbors[0]
                
                clique = {start_u, start_w}
                
                # Try to add more vertices
                # A vertex x can be added if it is connected to all vertices in clique
                # and the edges (x, y) for all y in clique are in remaining_edges
                changed = True
                while changed:
                    changed = False
                    # Candidates are vertices not in clique
                    for x in range(k):
                        if x in clique:
                            continue
                        
                        # Check if x can be added
                        can_add = True
                        for y in clique:
                            e = (min(x, y), max(x, y))
                            if e not in remaining_edges:
                                can_add = False
                                break
                        
                        if can_add:
                            clique.add(x)
                            changed = True
                
                # Add this clique to the partition
                if len(clique) >= 2:
                    cliques.append(sorted(list(clique)))
                    
                    # Remove all edges within this clique from remaining_edges
                    clique_list = sorted(list(clique))
                    for i in range(len(clique_list)):
                        for j in range(i + 1, len(clique_list)):
                            e = (clique_list[i], clique_list[j])
                            if e in remaining_edges:
                                remaining_edges.remove(e)
                
                # Fallback: if for some reason we didn't remove any edges (shouldn't happen)
                # take the first remaining edge as a 2-clique
                if remaining_edges:
                    if not any((min(u, w), max(u, w)) in remaining_edges for u, w in [(u, w) for u, w in remaining_edges]):
                         pass # Logic check
                    # Simpler safety: if clique was size 1 or failed to remove, just take one edge
                    # But the loop above ensures clique has at least 2 vertices from an existing edge.
                    # So it must remove at least that edge.

        # Final cleanup and type conversion
        cleaned = []
        for c in cliques:
            cl = [int(x) for x in c]
            if len(cl) >= 2 and len(set(cl)) == len(cl):
                cleaned.append(cl)
        
        return cleaned

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE:
            break
        cliques = partition_cliques(*spec)
        result.append(cliques)

    return result
# EVOLVE-BLOCK-END
