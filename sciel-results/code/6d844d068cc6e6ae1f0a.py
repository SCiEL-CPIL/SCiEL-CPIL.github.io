<<<<<<< SEARCH
def _local_search(adj, edges, all_edges, deadline):
    """
    Local search: try to add missing edges, and if blocked, try edge swaps.
    """
    edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
    missing = [(u, v) for u, v in all_edges if (u, v) not in edge_set]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding missing edges
        random.shuffle(missing)
        for u, v in missing[:]:
            if time.time() > deadline:
                break
            if (u, v) in edge_set:
                continue
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if not _has_q3_with_edge_fast(adj, u, v):
                edges.append([u, v])
                edge_set.add((u, v))
                if (u, v) in missing:
                    missing.remove((u, v))
                improved = True
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        # Try edge swaps: remove one edge, add two
        if time.time() > deadline - 5:
            break
            
        # For each missing edge that's blocked, find a witness Q3
        # and try removing an edge from that Q3 to unblock
        blocked_missing = [(u, v) for u, v in missing if (u, v) not in edge_set]
        random.shuffle(blocked_missing)
        
        for u, v in blocked_missing[:10]:  # Limit work per iteration
            if time.time() > deadline - 3:
                break
            if (u, v) in edge_set:
                continue
            
            # Find a Q3 witness containing (u,v)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            witness = _find_q3_witness(adj, u, v)
            
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            if witness is None:
                continue
            
            # Get the edges of this Q3 in the graph
            q3_edges = []
            for i in range(8):
                for j in range(i + 1, 8):
                    if (i ^ j) in (1, 2, 4):
                        gi, gj = witness[i], witness[j]
                        if gi < gj:
                            q3_edges.append((gi, gj))
                        else:
                            q3_edges.append((gj, gi))
            
            # Try removing each edge from the Q3 (except u-v itself)
            for rem_u, rem_v in q3_edges:
                if time.time() > deadline - 2:
                    break
                if rem_u == u and rem_v == v:
                    continue
                if rem_u == v and rem_v == u:
                    continue
                
                # Check if this edge exists
                if not (adj[rem_u] >> rem_v) & 1:
                    continue
                
                # Try removing this edge and adding (u,v)
                adj[rem_u] &= ~(1 << rem_v)
                adj[rem_v] &= ~(1 << rem_u)
                
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                # Check if the new graph is Q3-free
                if not _has_q3_with_edge_fast(adj, u, v):
                    # Update edges list
                    edges = [[e[0], e[1]] for e in edges if not (
                        (e[0] == rem_u and e[1] == rem_v) or 
                        (e[0] == rem_v and e[1] == rem_u)
                    )]
                    edges.append([u, v])
                    edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
                    missing = [(uu, vv) for uu, vv in all_edges if (uu, vv) not in edge_set]
                    improved = True
                    break
                else:
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    adj[rem_u] |= (1 << rem_v)
                    adj[rem_v] |= (1 << rem_u)
    
    return edges, adj
=======
def _local_search(adj, edges, all_edges, deadline):
    """
    Local search: try to add missing edges, and if blocked, try edge swaps.
    """
    edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
    missing = [(u, v) for u, v in all_edges if (u, v) not in edge_set]
    full_mask = (1 << _N) - 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding missing edges
        random.shuffle(missing)
        for u, v in missing[:]:
            if time.time() > deadline:
                break
            if (u, v) in edge_set:
                continue
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if not _has_q3_with_edge_fast(adj, u, v):
                edges.append([u, v])
                edge_set.add((u, v))
                if (u, v) in missing:
                    missing.remove((u, v))
                improved = True
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        # Try edge swaps: remove one edge, add one edge
        if time.time() > deadline - 5:
            break
            
        # For each missing edge that's blocked, find a witness Q3
        # and try removing an edge from that Q3 to unblock
        blocked_missing = [(u, v) for u, v in missing if (u, v) not in edge_set]
        random.shuffle(blocked_missing)
        
        for u, v in blocked_missing[:15]:  # Limit work per iteration
            if time.time() > deadline - 3:
                break
            if (u, v) in edge_set:
                continue
            
            # Find a Q3 witness containing (u,v)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            witness = _find_q3_witness(adj, u, v)
            
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            if witness is None:
                continue
            
            # Get the edges of this Q3 in the graph
            q3_edges = []
            for i in range(8):
                for j in range(i + 1, 8):
                    if (i ^ j) in (1, 2, 4):
                        gi, gj = witness[i], witness[j]
                        if gi < gj:
                            q3_edges.append((gi, gj))
                        else:
                            q3_edges.append((gj, gi))
            
            # Try removing each edge from the Q3 (except u-v itself)
            for rem_u, rem_v in q3_edges:
                if time.time() > deadline - 2:
                    break
                if rem_u == u and rem_v == v:
                    continue
                if rem_u == v and rem_v == u:
                    continue
                
                # Check if this edge exists
                if not (adj[rem_u] >> rem_v) & 1:
                    continue
                
                # Try removing this edge and adding (u,v)
                adj[rem_u] &= ~(1 << rem_v)
                adj[rem_v] &= ~(1 << rem_u)
                
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                # Check if the new graph is Q3-free
                if not _has_q3_with_edge_fast(adj, u, v):
                    # Update edges list
                    edges = [[e[0], e[1]] for e in edges if not (
                        (e[0] == rem_u and e[1] == rem_v) or 
                        (e[0] == rem_v and e[1] == rem_u)
                    )]
                    edges.append([u, v])
                    edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
                    missing = [(uu, vv) for uu, vv in all_edges if (uu, vv) not in edge_set]
                    improved = True
                    break
                else:
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    adj[rem_u] |= (1 << rem_v)
                    adj[rem_v] |= (1 << rem_u)
        
        # Try 2-for-1 swaps: remove 2 edges, add 1 edge
        if time.time() > deadline - 4:
            break
        
        # For blocked missing edges, try removing two edges from a witness
        blocked_missing2 = [(u, v) for u, v in missing if (u, v) not in edge_set]
        random.shuffle(blocked_missing2)
        
        for u, v in blocked_missing2[:5]:
            if time.time() > deadline - 3:
                break
            if (u, v) in edge_set:
                continue
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            witness = _find_q3_witness(adj, u, v)
            
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            if witness is None:
                continue
            
            # Get Q3 edges
            q3_edges = []
            for i in range(8):
                for j in range(i + 1, 8):
                    if (i ^ j) in (1, 2, 4):
                        gi, gj = witness[i], witness[j]
                        if gi < gj:
                            q3_edges.append((gi, gj))
                        else:
                            q3_edges.append((gj, gi))
            
            # Try removing pairs of edges from Q3
            for i in range(len(q3_edges)):
                for j in range(i + 1, len(q3_edges)):
                    if time.time() > deadline - 2:
                        break
                    ru1, rv1 = q3_edges[i]
                    ru2, rv2 = q3_edges[j]
                    
                    # Skip if either is (u,v)
                    if (ru1 == u and rv1 == v) or (ru1 == v and rv1 == u):
                        continue
                    if (ru2 == u and rv2 == v) or (ru2 == v and rv2 == u):
                        continue
                    
                    # Check edges exist
                    if not (adj[ru1] >> rv1) & 1 or not (adj[ru2] >> rv2) & 1:
                        continue
                    
                    # Try removing both edges and adding (u,v)
                    adj[ru1] &= ~(1 << rv1)
                    adj[rv1] &= ~(1 << ru1)
                    adj[ru2] &= ~(1 << rv2)
                    adj[rv2] &= ~(1 << ru2)
                    
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
                    if not _has_q3_with_edge_fast(adj, u, v):
                        # Update edges
                        remove_set = {(min(ru1, rv1), max(ru1, rv1)), (min(ru2, rv2), max(ru2, rv2))}
                        edges = [[e[0], e[1]] for e in edges if (min(e[0], e[1]), max(e[0], e[1])) not in remove_set]
                        edges.append([u, v])
                        edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
                        missing = [(uu, vv) for uu, vv in all_edges if (uu, vv) not in edge_set]
                        improved = True
                        break
                    else:
                        # Revert
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        adj[ru1] |= (1 << rv1)
                        adj[rv1] |= (1 << ru1)
                        adj[ru2] |= (1 << rv2)
                        adj[rv2] |= (1 << ru2)
    
    return edges, adj
>>>>>>> REPLACE

