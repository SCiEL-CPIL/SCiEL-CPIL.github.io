import time
import random

def construct():
    n = 20
    deadline = time.time() + 110
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_index = {(u, v): i for i, (u, v) in enumerate(all_edges)}
    m = len(all_edges)
    
    # Build conflict graph: two edges conflict if adding both creates a K2,3
    # Conflict condition: edges (a,b) and (c,d) conflict if:
    # 1. They share a vertex and the other two vertices have >=1 common neighbor
    # 2. They are disjoint and the four vertices form a K2,3 pattern
    conflict = [0] * m  # bitmasks of conflicting edges
    
    for i in range(m):
        a, b = all_edges[i]
        for j in range(i + 1, m):
            c, d = all_edges[j]
            # Check if these two edges create a K2,3
            # K2,3 means two vertices with three common neighbors
            # If edges share a vertex, say (a,b) and (a,c), then b and c need >=1 common neighbor other than a
            # Actually, the constraint is: no two vertices have 3 common neighbors
            # Two edges conflict if their union creates a situation where some pair has >=3 common neighbors
            
            # Case 1: Edges share exactly one vertex
            if (a == c or a == d or b == c or b == d):
                # Find the shared vertex and the two non-shared vertices
                if a == c:
                    # Edges (a,b) and (a,d): b and d must not have common neighbors besides a
                    # But we need to check if adding both would make some pair have >=3 common neighbors
                    # This is complex; let's use a different approach
                    pass
                elif a == d:
                    pass
                elif b == c:
                    pass
                else:  # b == d
                    pass
            
            # For now, use a simpler conflict detection:
            # Two edges conflict if they form part of a K2,3
            # Actually, let's reconsider. The constraint is that NO two vertices have 3 common neighbors.
            # So we need to check all pairs of vertices in the graph formed by these edges.
            
            # Simpler approach: Two edges (a,b) and (c,d) conflict if:
            # - They share a vertex, and the "other" vertices would end up with too many common neighbors
            # - Or they are disjoint and create a K2,3 structure
            
            # Let's use a direct check: can we add both edges to an empty graph without creating K2,3?
            # In an empty graph, adding (a,b) and (c,d):
            # If they share a vertex, say (a,b) and (a,c):
            #   Common neighbors of b and c: {a} (only if a is neighbor of both, which it is)
            #   So |N(b) ∩ N(c)| = 1, which is <= 2, OK
            # If they are disjoint:
            #   Common neighbors of any pair: at most 1 (the shared endpoint if any)
            # So two edges never conflict in isolation!
            
            # The conflict arises when we consider the TRANSITIVE closure:
            # If we have edges (a,b), (a,c), (a,d), then b,c,d all have a as common neighbor
            # But we need THREE common neighbors for a K2,3
            # So (a,b), (a,c), (a,d) together create a K2,3 if there's another common neighbor
            # Actually no: K2,3 requires TWO vertices with THREE common neighbors
            # In star K1,3: center a, leaves b,c,d. 
            #   N(b) ∩ N(c) = {a}, size 1
            #   So no K2,3 yet.
            
            # K2,3 is: vertices x,y with |N(x) ∩ N(y)| >= 3
            # So we need three distinct vertices that are all neighbors of both x and y.
            # This means we need a K_{2,3} subgraph.
            
            # Two edges conflict if their addition (in combination with existing edges) would create K2,3.
            # But for the conflict graph formulation, we need: two edges are incompatible if BOTH being present
            # necessarily creates a K2,3, regardless of other edges.
            # This only happens if the two edges themselves form a K2,3, which requires 5 edges.
            # So no pair of edges directly conflicts!
            
            # I was overcomplicating. The K2,3 constraint is a 3-edge constraint:
            # Three edges (x,a), (x,b), (x,c) plus (y,a), (y,b), (y,c) create K2,3.
            # So the conflict is between SETS of edges, not pairs.
            
            # This means MCIS on a pairwise conflict graph won't work directly.
            # Let me revert to the local search approach but improve it.
            pass
    
    # Since pairwise conflict doesn't capture K2,3, let's stick with local search
    # but make it much more efficient with better heuristics
    
    def validate_and_add(adj, u, v):
        """Check if edge (u,v) can be added given current adj. Returns True if valid."""
        if (adj[u] & adj[v]).bit_count() > 2:
            return False
        
        mask = adj[v]
        while mask:
            b = (mask & -mask).bit_length() - 1
            if (adj[u] & adj[b]).bit_count() > 1:
                return False
            mask &= mask - 1
        
        mask = adj[u]
        while mask:
            a = (mask & -mask).bit_length() - 1
            if (adj[v] & adj[a]).bit_count() > 1:
                return False
            mask &= mask - 1
        
        return True

    def local_search(adj, current_edges, max_iter=1000):
        """Aggressive local search with multiple improvement strategies."""
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() > deadline:
                break
            
            # 1. Try adding any missing edge
            added_any = False
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        continue
                    if validate_and_add(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        current_edges.append((u, v))
                        added_any = True
                        break
                if added_any:
                    break
            
            if added_any:
                improved = True
                continue

            # 2. Try 1-for-2 swap
            swapped_any = False
            for i in range(len(current_edges)):
                u, v = current_edges[i]
                old_adj = adj[:]
                old_edges = current_edges[:]
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges[i] = current_edges[-1]
                current_edges.pop()
                
                added = 0
                for w in range(n):
                    if w == u or w == v or (adj[u] >> w) & 1:
                        continue
                    if validate_and_add(adj, u, w):
                        adj[u] |= (1 << w)
                        adj[w] |= (1 << u)
                        current_edges.append((u, w))
                        added += 1
                        if added == 2:
                            break
                if added < 2:
                    for w in range(n):
                        if w == u or w == v or (adj[v] >> w) & 1:
                            continue
                        if validate_and_add(adj, v, w):
                            adj[v] |= (1 << w)
                            adj[w] |= (1 << v)
                            current_edges.append((v, w))
                            added += 1
                            if added == 2:
                                break
                
                if added < 2:
                    adj[:] = old_adj
                    current_edges[:] = old_edges
                else:
                    swapped_any = True
                    break
            
            if swapped_any:
                improved = True
                continue

            # 3. Try 2-for-3 swap (limited for time)
            if time.time() < deadline - 5:
                swapped23 = False
                for i in range(len(current_edges)):
                    u1, v1 = current_edges[i]
                    for j in range(i + 1, len(current_edges)):
                        u2, v2 = current_edges[j]
                        if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                            continue
                        
                        old_adj = adj[:]
                        old_edges = current_edges[:]
                        
                        adj[u1] &= ~(1 << v1)
                        adj[v1] &= ~(1 << u1)
                        adj[u2] &= ~(1 << v2)
                        adj[v2] &= ~(1 << u2)
                        current_edges[i] = current_edges[-1]
                        current_edges.pop()
                        current_edges[j] = current_edges[-1]
                        current_edges.pop()
                        
                        added = 0
                        for a in range(n):
                            if added >= 3:
                                break
                            for b in range(a + 1, n):
                                if added >= 3:
                                    break
                                if (adj[a] >> b) & 1:
                                    continue
                                if validate_and_add(adj, a, b):
                                    adj[a] |= (1 << b)
                                    adj[b] |= (1 << a)
                                    current_edges.append((a, b))
                                    added += 1
                                    if added == 3:
                                        break
                        
                        if added < 3:
                            adj[:] = old_adj
                            current_edges[:] = old_edges
                        else:
                            swapped23 = True
                            break
                    if swapped23:
                        break
                
                if swapped23:
                    improved = True
                    continue
            
        return current_edges

    best_edges = []
    best_count = 0
    
    # Multiple restarts with different random orderings
    while time.time() < deadline - 15:
        edges_copy = all_edges[:]
        random.shuffle(edges_copy)
        
        adj = [0] * n
        current_edges = []
        
        for u, v in edges_copy:
            if validate_and_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.append((u, v))
        
        current_edges = local_search(adj, current_edges, max_iter=500)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
            if time.time() < deadline - 10:
                adj2 = [0] * n
                for u, v in best_edges:
                    adj2[u] |= (1 << v)
                    adj2[v] |= (1 << u)
                best_edges = local_search(adj2, best_edges, max_iter=2000)
                if len(best_edges) > best_count:
                    best_count = len(best_edges)

    # Final polish
    if best_edges and time.time() < deadline - 2:
        adj_final = [0] * n
        for u, v in best_edges:
            adj_final[u] |= (1 << v)
            adj_final[v] |= (1 << u)
        
        # Intensive final local search
        best_edges = local_search(adj_final, best_edges, max_iter=5000)
        
        # Try to find any remaining addable edges
        for _ in range(100):
            if time.time() > deadline - 1:
                break
            added_any = False
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj_final[u] >> v) & 1:
                        continue
                    if validate_and_add(adj_final, u, v):
                        adj_final[u] |= (1 << v)
                        adj_final[v] |= (1 << u)
                        best_edges.append((u, v))
                        added_any = True
                        break
                if added_any:
                    break
            if not added_any:
                break
        
        best_count = len(best_edges)

    return [[u, v] for u, v in best_edges]
