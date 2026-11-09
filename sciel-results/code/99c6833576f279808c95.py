import time
import numpy as np
from scipy.optimize import linear_sum_assignment

def construct():
    def instance(n, p, seed):
        import random
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges, n):
        remaining = set()
        for (u, v) in edges:
            remaining.add((min(u, v), max(u, v)))
        
        pieces = []
        start_time = time.time()
        deadline = start_time + 110.0  # Safety margin

        while True:
            if time.time() > deadline:
                break
            
            # Build bipartite graph for matching
            # Left nodes: 0..n-1, Right nodes: 0..n-1
            # Cost matrix: 0 if edge exists in remaining, large value otherwise
            # We want to maximize the number of edges in the matching, 
            # which corresponds to minimizing the sum of costs (0 vs large).
            
            # Optimization: Only consider vertices that have degree > 0 in remaining
            active_vertices = set()
            for u, v in remaining:
                active_vertices.add(u)
                active_vertices.add(v)
            
            if not active_vertices:
                break
                
            # If no edges left, break
            if not remaining:
                break

            # Map active vertices to indices 0..k-1 for smaller matrix
            active_list = sorted(active_vertices)
            idx_map = {v: i for i, v in enumerate(active_list)}
            k = len(active_list)
            
            # Create cost matrix
            # Cost 0 if edge (u, v) is in remaining, else INF
            INF = 10000
            cost_matrix = np.full((k, k), INF, dtype=int)
            
            for u, v in remaining:
                if u in idx_map and v in idx_map:
                    i = idx_map[u]
                    j = idx_map[v]
                    cost_matrix[i, j] = 0
                    # Since graph is undirected, edge (u,v) exists in both directions 
                    # in the bipartite representation? 
                    # Wait, the bipartite double cover has edges (u_L, v_R) and (v_L, u_R).
                    # A perfect matching in the bipartite graph corresponds to a 2-factor 
                    # (collection of cycles) in the original graph.
                    # So we need to set both cost_matrix[i, j] and cost_matrix[j, i] to 0.
                    cost_matrix[j, i] = 0

            # Solve assignment problem
            # linear_sum_assignment returns row_ind, col_ind such that sum(cost_matrix[row_ind, col_ind]) is minimized
            try:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
            except Exception:
                break
            
            # Check if we found a valid matching (all assigned costs should be 0)
            # If any assigned cost is INF, it means we couldn't match all active nodes,
            # but linear_sum_assignment always returns a perfect matching.
            # However, if cost is INF, it means no edge existed.
            # We only want to include pairs where an edge actually exists.
            
            matched_pairs = []
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] == 0:
                    u = active_list[i]
                    v = active_list[j]
                    matched_pairs.append((u, v))
            
            if not matched_pairs:
                # No cycles possible? Just break and let remaining edges be single pieces.
                # Actually, if no pairs are matched with cost 0, it means the maximum matching
                # used only non-edges (which shouldn't happen if we set INF high enough and 
                # there are no edges, but if there are edges, it should find them).
                # If we have edges but no valid matching pairs, it implies the graph structure
                # doesn't allow a 2-factor? No, a matching in the double cover corresponds to 
                # a set of cycles. If the matching size is less than k, we still get some cycles.
                # But linear_sum_assignment gives a perfect matching of size k.
                # If cost_matrix[i,j] > 0, that pair is not a valid edge in the original graph.
                # We must discard those.
                
                # If no valid edges were found in the matching, we cannot form any cycles 
                # using this specific matching structure? 
                # Actually, if the graph has edges, there should be some 0s.
                # If we end up with no valid pairs, it means the "matching" picked non-edges.
                # This happens if the graph is sparse and the algorithm is forced to pick 
                # non-edges to complete the perfect matching. 
                # In that case, we should just stop looking for cycles via this method 
                # and treat remaining edges as singles? 
                # Or better: we only use the subset of the matching that corresponds to real edges.
                # Those real edges form disjoint cycles (or paths? No, in a 2-factor they form cycles).
                # Wait, a subset of a 2-factor is not necessarily a 2-factor.
                # If we pick a subset of edges from the matching, do they form cycles?
                # Yes, because the matching is a set of disjoint edges in the bipartite graph.
                # In the original graph, these edges form a set of disjoint cycles.
                # If we discard some "non-edge" assignments from the perfect matching, 
                # the remaining "edge" assignments still form disjoint cycles.
                
                if not matched_pairs:
                    break

            # Reconstruct cycles from matched_pairs
            # The matched pairs (u, v) represent directed edges u->v in the 2-factor.
            # Since it's a 2-factor, each vertex has in-degree 1 and out-degree 1.
            # So we can trace cycles.
            
            next_vertex = {}
            for u, v in matched_pairs:
                next_vertex[u] = v
            
            visited = set()
            cycles = []
            
            for u in next_vertex:
                if u in visited:
                    continue
                # Trace cycle
                cycle = []
                curr = u
                while curr not in visited:
                    visited.add(curr)
                    cycle.append(curr)
                    curr = next_vertex.get(curr)
                    if curr is None:
                        # Should not happen if it's a valid 2-factor subset
                        break
                if len(cycle) >= 3:
                    cycles.append(cycle)
                elif len(cycle) == 2:
                    # A 2-cycle in the bipartite matching corresponds to a single edge (u,v) 
                    # where both (u,v) and (v,u) are in the matching?
                    # No, in the bipartite graph, edge (u_L, v_R) and (v_L, u_R) are distinct.
                    # If both are selected, it forms a cycle of length 2 in the bipartite graph,
                    # which corresponds to the single edge {u,v} in the original graph being used twice?
                    # No, the original graph is simple. The edge {u,v} exists once.
                    # In the bipartite double cover, the edge {u,v} appears as (u_L, v_R) and (v_L, u_R).
                    # If the matching selects both, it means u is matched to v and v is matched to u.
                    # This corresponds to the edge {u,v} in the original graph.
                    # However, a single edge is not a cycle of length >= 3.
                    # The problem statement says "cycle given by at least three distinct vertices".
                    # So we cannot output a 2-cycle as a cycle piece.
                    # We must output it as an edge piece [u, v].
                    # But wait, if we have a 2-cycle in the matching, it consumes the edge {u,v}.
                    # We should just add it as an edge piece.
                    
                    # Actually, if we have a 2-cycle in the bipartite matching, it means
                    # the edge {u,v} is part of the "2-factor". But a 2-factor in a simple graph
                    # consists of cycles of length >= 3.
                    # How can a 2-cycle appear?
                    # It appears if the matching includes (u_L, v_R) and (v_L, u_R).
                    # This implies that in the set of selected edges, u is connected to v and v is connected to u.
                    # In the original graph, this is just the single edge {u,v}.
                    # Since a single edge is not a valid "cycle piece" (needs >= 3 vertices),
                    # we should not group it as a cycle. We should treat it as a single edge.
                    
                    # However, if we remove it from 'remaining', we must add it as a piece.
                    # The logic below handles this: if len(cycle) < 3, we don't add it to 'cycles'.
                    # But we still need to remove the edge from 'remaining' and add it as a piece.
                    
                    # Let's handle this carefully.
                    pass
            
            # We need to process the cycles found.
            # For each valid cycle (len >= 3), remove edges from remaining and add to pieces.
            # For 2-cycles (len == 2), remove the single edge from remaining and add as edge piece.
            
            new_pieces = []
            edges_to_remove = set()
            
            for cyc in cycles:
                if len(cyc) >= 3:
                    # Validate cycle edges
                    valid = True
                    for i in range(len(cyc)):
                        u = cyc[i]
                        v = cyc[(i+1) % len(cyc)]
                        e = (min(u, v), max(u, v))
                        if e not in remaining:
                            valid = False
                            break
                    if valid:
                        new_pieces.append(cyc)
                        for i in range(len(cyc)):
                            u = cyc[i]
                            v = cyc[(i+1) % len(cyc)]
                            edges_to_remove.add((min(u, v), max(u, v)))
            
            # Handle 2-cycles (single edges from matching)
            # We need to identify which pairs formed 2-cycles.
            # A pair (u, v) is in a 2-cycle if next_vertex[u] == v and next_vertex[v] == u.
            # And u != v.
            for u, v in matched_pairs:
                if next_vertex.get(v) == u and u != v:
                    # Check if this edge was already handled as part of a larger cycle?
                    # No, if it's a 2-cycle, it's isolated from larger cycles in the matching structure.
                    # But wait, if u is in a 3-cycle, next_vertex[u] is not v (unless v is the next in 3-cycle, 
                    # but then next_vertex[v] is not u).
                    # So 2-cycles are disjoint from larger cycles in the matching.
                    
                    e = (min(u, v), max(u, v))
                    if e in remaining:
                        # Ensure we don't add it twice if it was part of a larger cycle (impossible)
                        # Add as edge piece
                        new_pieces.append([u, v])
                        edges_to_remove.add(e)

            if not new_pieces:
                break

            pieces.extend(new_pieces)
            remaining.difference_update(edges_to_remove)

        # Add remaining edges as single pieces
        for (u, v) in sorted(remaining):
            pieces.append([u, v])

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n = spec[0]
        edges = instance(*spec)
        results.append(decompose(edges, n))
    return results
