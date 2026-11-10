"""Clique partition of split graphs using structural properties."""
import time
from random import Random
import networkx as nx

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
        C = list(range(k))
        I = list(range(k, n))
        
        # Build adjacency sets for quick lookup
        # neighbors[u] = set of v connected to u
        neighbors = {u: set() for u in range(n)}
        edge_set = set()
        for u, v in edges:
            neighbors[u].add(v)
            neighbors[v].add(u)
            edge_set.add((min(u, v), max(u, v)))
            
        uncovered_edges = set(edge_set)
        cliques = []
        
        # Step 1: Process vertices in I
        # For each v in I, form a clique with its neighbors in C
        for v in I:
            if time.time() > deadline - 10:
                break
            N_v = [u for u in C if u in neighbors[v]]
            if not N_v:
                continue
            
            # The clique is N_v + [v]
            # We need to ensure this is a valid clique.
            # Since C is a clique, any subset of C is a clique.
            # Since N_v contains only neighbors of v, all edges (u, v) exist.
            # So N_v U {v} is a clique.
            
            clique = N_v + [v]
            
            # Only add if it covers at least one uncovered edge?
            # The task says "Every listed set must be a clique with at least two vertices".
            # If N_v is empty, size is 1 (just v), which is invalid. We skipped those.
            # If N_v has size 1, clique size is 2. Valid.
            # If N_v has size > 1, clique size > 2. Valid.
            
            # Check if this clique actually covers any *uncovered* edges.
            # If all edges in this clique are already covered, we shouldn't add it?
            # The rule: "every edge must lie in exactly one listed clique".
            # If we add a clique that only contains covered edges, it's redundant but does it violate "exactly one"?
            # No, it just means those edges are in multiple cliques. That IS a violation.
            # So we MUST only add a clique if it contains at least one edge that is currently uncovered.
            # AND we must remove ALL edges of this clique from uncovered_edges.
            # If an edge was already covered, it's already removed from uncovered_edges.
            # So, we check if there are ANY edges in this clique that are in uncovered_edges.
            
            has_uncovered = False
            for u in N_v:
                # Edge (u, v)
                e_uv = (min(u, v), max(u, v))
                if e_uv in uncovered_edges:
                    has_uncovered = True
                    break
                # Edges within N_v
                for w in N_v:
                    if u >= w: continue
                    e_uw = (min(u, w), max(u, w))
                    if e_uw in uncovered_edges:
                        has_uncovered = True
                        break
                if has_uncovered:
                    break
            
            if has_uncovered:
                cliques.append(clique)
                # Remove all edges in this clique from uncovered_edges
                for u in N_v:
                    e_uv = (min(u, v), max(u, v))
                    uncovered_edges.discard(e_uv)
                    for w in N_v:
                        if u >= w: continue
                        e_uw = (min(u, w), max(u, w))
                        uncovered_edges.discard(e_uw)
        
        # Step 2: Process remaining edges in C
        # These are edges (u, w) in C such that (u, w) is in uncovered_edges.
        # We need to partition these edges into cliques.
        # A clique here is a set of vertices S subset of C such that all pairs in S have edges in uncovered_edges.
        # We can greedily find maximal cliques in the graph defined by (C, uncovered_edges).
        
        remaining_vertices = set(C)
        
        while uncovered_edges:
            if time.time() > deadline - 2:
                # Fallback: just add each remaining edge as a 2-clique to ensure validity
                # This might be slow but ensures we finish
                for u, v in list(uncovered_edges):
                    if time.time() > deadline:
                         break
                    cliques.append([u, v])
                    uncovered_edges.discard((min(u, v), max(u, v)))
                break
            
            # Find a maximal clique in the graph (remaining_vertices, uncovered_edges restricted to remaining_vertices)
            # Since k is small, we can just try to build a clique greedily.
            # Pick a vertex with highest degree in uncovered_edges?
            # Or just pick the first available vertex.
            
            # To maximize clique size, we can try to find the maximum clique in the remaining graph.
            # For k <= 30, this is fast enough if we do it a few times.
            # But simpler: Greedy Maximal Clique.
            
            # Let's find a vertex u in remaining_vertices with max degree in uncovered_edges
            max_deg = -1
            start_v = -1
            for u in remaining_vertices:
                deg = sum(1 for w in remaining_vertices if (min(u, w), max(u, w)) in uncovered_edges)
                if deg > max_deg:
                    max_deg = deg
                    start_v = u
            
            if start_v == -1:
                break # Should not happen if uncovered_edges is not empty
            
            # Build clique starting with start_v
            clique = [start_v]
            candidates = [w for w in remaining_vertices if w != start_v and (min(start_v, w), max(start_v, w)) in uncovered_edges]
            
            # Greedily add vertices to clique
            for w in candidates:
                # Check if w is connected to all existing vertices in clique
                is_connected = True
                for u in clique:
                    if (min(u, w), max(u, w)) not in uncovered_edges:
                        is_connected = False
                        break
                if is_connected:
                    clique.append(w)
            
            if len(clique) < 2:
                # This shouldn't happen if we picked a vertex with degree > 0, 
                # but if max_deg was 0, we have isolated vertices? 
                # No, if max_deg == 0, uncovered_edges is empty? 
                # If uncovered_edges is not empty, there must be an edge, so some vertex has degree > 0.
                # Unless remaining_vertices doesn't contain the endpoints? 
                # We should update remaining_vertices.
                # Actually, if we just added a clique, we should remove those vertices from consideration?
                # No, we remove EDGES. Vertices can be in multiple cliques if they have edges to different parts.
                # But wait, if a vertex is in a clique, all its edges *within that clique* are covered.
                # It might have other uncovered edges to other vertices.
                # So we should NOT remove vertices from remaining_vertices, only edges from uncovered_edges.
                
                # If clique size is 1, it's invalid.
                # This implies start_v has no neighbors in remaining_vertices via uncovered_edges.
                # But we picked it based on degree in uncovered_edges restricted to remaining_vertices.
                # So if max_deg == 0, we break.
                if max_deg == 0:
                    break
                else:
                    # Fallback for safety
                    # Find any edge incident to start_v
                    found = False
                    for w in remaining_vertices:
                        if (min(start_v, w), max(start_v, w)) in uncovered_edges:
                            cliques.append([start_v, w])
                            uncovered_edges.discard((min(start_v, w), max(start_v, w)))
                            found = True
                            break
                    if not found:
                         # Should not happen
                         break
                    continue
            
            # Add clique
            cliques.append(clique)
            
            # Remove all edges within this clique from uncovered_edges
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, w = clique[i], clique[j]
                    uncovered_edges.discard((min(u, w), max(u, w)))
        
        # Final check: if there are still uncovered edges (due to time limit break in while loop),
        # we must cover them to be valid.
        if uncovered_edges:
            for u, v in list(uncovered_edges):
                cliques.append([u, v])
            uncovered_edges.clear()

        # Clean up types
        cleaned_cliques = []
        for c in cliques:
            cleaned_cliques.append([int(x) for x in c])
        
        return cleaned_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = partition_split_graph(n, k, edges)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
