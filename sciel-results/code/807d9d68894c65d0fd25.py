def _local_search_girth(edges, n, target_girth, deadline):
    # Convert to adjacency list
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((min(u,v), max(u,v)))
        
    for _ in range(1000): # Max iterations
        if time.time() > deadline:
            break
        # Find a 4-cycle
        cycle = _find_4_cycle(adj, n)
        if cycle is None:
            # Check full girth
            if _has_girth_at_least_from_adj(adj, n, target_girth):
                return n, list(edge_set)
            # If no 4-cycle but girth < target, try 6-cycle breaker or just return
            # For now, if no 4-cycle, we are close
            # Try to break 6-cycles
            cycle6 = _find_6_cycle(adj, n)
            if cycle6 is None:
                 if _has_girth_at_least_from_adj(adj, n, target_girth):
                     return n, list(edge_set)
            else:
                 # Break 6-cycle
                 pass
        else:
            # Break the 4-cycle
            u1, v1, u2, v2 = cycle
            # Edges are (u1,v1), (v1,u2), (u2,v2), (v2,u1)
            # Try switching (u1,v1) and (u2,v2) to (u1,v2) and (u2,v1) ? No, v2 is neighbor of u1.
            # The 4-cycle is u1-v1-u2-v2-u1.
            # Edges: (u1,v1), (v1,u2), (u2,v2), (v2,u1).
            # We can replace (u1,v1) and (u2,v2) with (u1,v2) and (u2,v1)?
            # (u1,v2) is already an edge in the cycle.
            # We need to disconnect the cycle.
            # Replace (u1,v1) with (u1, v_other) and (v1, u_other)?
            # This requires finding other neighbors.
            # Simple heuristic: Pick a random neighbor of u1 (not v1, v2) and a random neighbor of v1 (not u1, u2).
            # This is complex.
            # Let's just restart with a new random graph if we find a 4-cycle.
            # Given the low probability, restarting is simpler and might work.
            break

    return None
