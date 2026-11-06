import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Canonical Q3 structure
    # Vertices 0-7
    # Edges:
    # 0-1, 0-2, 0-4
    # 1-3, 1-5
    # 2-3, 2-6
    # 3-7
    # 4-5, 4-6
    # 5-7
    # 6-7
    
    # Adjacency lists for canonical Q3
    q3_adj = {
        0: [1, 2, 4],
        1: [0, 3, 5],
        2: [0, 3, 6],
        3: [1, 2, 7],
        4: [0, 5, 6],
        5: [1, 4, 7],
        6: [2, 4, 7],
        7: [3, 5, 6]
    }
    
    # For the "find_q3_with_edge" check, we fix edge (0,1) in canonical Q3
    # to the candidate edge (u,v) in the graph.
    # We need to map canonical vertices 2,3,4,5,6,7 to graph vertices.
    
    # Pre-compute constraints for the remaining 6 vertices based on fixed edge (0,1)
    # Vertex 2: N(2) in Q3 = {0, 3, 6}. Must be in N(u) in Graph.
    # Vertex 4: N(4) in Q3 = {0, 5, 6}. Must be in N(u) in Graph.
    # Vertex 3: N(3) in Q3 = {1, 2, 7}. Must be in N(v) in Graph.
    # Vertex 5: N(5) in Q3 = {1, 4, 7}. Must be in N(v) in Graph.
    # Vertex 6: N(6) in Q3 = {2, 4, 7}. Must be in N(assigned_2) and N(assigned_4).
    # Vertex 7: N(7) in Q3 = {3, 5, 6}. Must be in N(assigned_3) and N(assigned_5).
    
    # Also, 2 and 4 must be distinct and not u, v.
    # 3 and 5 must be distinct and not u, v.
    # 2,4,3,5,6,7 must all be distinct.
    
    def find_q3_with_edge(adj, u, v):
        """
        Check if adding edge (u,v) to graph adj creates a Q3 containing (u,v).
        adj is a list of 16 integers representing adjacency bitmasks.
        Returns True if a Q3 is found, False otherwise.
        """
        # Candidates for 2 and 4: Neighbors of u, excluding v
        # Note: In the current graph, (u,v) is NOT yet added.
        # So N(u) does not contain v.
        # But we are checking if adding (u,v) creates a Q3.
        # So we assume (u,v) exists for the purpose of the check.
        
        # Neighbors of u in current graph
        n_u = adj[u]
        # Neighbors of v in current graph
        n_v = adj[v]
        
        # Candidates for canonical vertex 2: Must be in N(u)
        # Candidates for canonical vertex 4: Must be in N(u)
        # Candidates for canonical vertex 3: Must be in N(v)
        # Candidates for canonical vertex 5: Must be in N(v)
        
        # Let's iterate over possible assignments for 2 and 4 first.
        # They must be distinct neighbors of u.
        # To prune, we can collect bits.
        
        # Get list of neighbors of u
        neigh_u = [i for i in range(16) if (n_u >> i) & 1]
        # Get list of neighbors of v
        neigh_v = [i for i in range(16) if (n_v >> i) & 1]
        
        # If u and v are not connected yet, they are not in each other's neighbor lists.
        # So neigh_u does not contain v, and neigh_v does not contain u.
        
        # We need to pick 2 and 4 from neigh_u.
        # We need to pick 3 and 5 from neigh_v.
        
        # To reduce search space, we can sort candidates or just iterate.
        # Since degree is likely high, this might be slow if not pruned.
        
        # Let's use backtracking.
        # Order: 2, 4, 3, 5, 6, 7
        
        used = [False] * 16
        used[u] = True
        used[v] = True
        
        # Assign 2
        for v2 in neigh_u:
            if used[v2]: continue
            used[v2] = True
            # Assign 4
            for v4 in neigh_u:
                if used[v4]: continue
                used[v4] = True
                # Assign 3
                # 3 must be in N(v) AND N(2) (since 2-3 is edge in Q3)
                # Check intersection of N(v) and N(v2)
                # N(v) is n_v. N(v2) is adj[v2].
                # Candidates for 3: in neigh_v AND in adj[v2]
                # Also 3 must be distinct from 2, 4, u, v
                for v3 in neigh_v:
                    if used[v3]: continue
                    if not ((adj[v2] >> v3) & 1): continue
                    used[v3] = True
                    # Assign 5
                    # 5 must be in N(v) AND N(4) (since 4-5 is edge in Q3)
                    # Candidates for 5: in neigh_v AND in adj[v4]
                    for v5 in neigh_v:
                        if used[v5]: continue
                        if not ((adj[v4] >> v5) & 1): continue
                        used[v5] = True
                        # Assign 6
                        # 6 must be in N(2) AND N(4) AND N(7) -- wait, 6-7 is edge.
                        # 6 must be in N(2) AND N(4).
                        # Also 6 must be distinct from 2,4,3,5,u,v
                        # Candidates for 6: in adj[v2] AND in adj[v4]
                        for v6 in range(16):
                            if used[v6]: continue
                            if not ((adj[v2] >> v6) & 1): continue
                            if not ((adj[v4] >> v6) & 1): continue
                            used[v6] = True
                            # Assign 7
                            # 7 must be in N(3) AND N(5) AND N(6)
                            # Candidates for 7: in adj[v3] AND in adj[v5] AND in adj[v6]
                            for v7 in range(16):
                                if used[v7]: continue
                                if not ((adj[v3] >> v7) & 1): continue
                                if not ((adj[v5] >> v7) & 1): continue
                                if not ((adj[v6] >> v7) & 1): continue
                                # Found a Q3
                                used[v7] = False
                                used[v6] = False
                                used[v5] = False
                                used[v3] = False
                                used[v4] = False
                                used[v2] = False
                                return True
                            used[v6] = False
                        used[v5] = False
                    used[v3] = False
                used[v4] = False
            used[v2] = False
        return False

    best_edges = set()
    best_count = 0
    
    random.seed(42)
    
    # Strategy: Multiple restarts of Greedy Add
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
            
        adj = [0] * 16
        edges = set()
        
        # Shuffle edges
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if time.time() > deadline - 5:
                break
            # Check if adding (u,v) creates a Q3
            # We need to temporarily consider (u,v) as existing for the check?
            # No, find_q3_with_edge checks if there is a Q3 *using* (u,v).
            # The check relies on the OTHER edges of the Q3 being present.
            # So we just call find_q3_with_edge on the current adj.
            # If it returns True, it means there is a Q3 in (G U {(u,v)}) that uses (u,v).
            # This is exactly what we want to avoid.
            
            if not find_q3_with_edge(adj, u, v):
                # Safe to add
                edges.add((u, v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = set(edges)
            
        # Local Search: Try to add more edges
        # We might have missed some edges due to the random order.
        # Try all missing edges again.
        missing = [e for e in all_edges if e not in edges]
        random.shuffle(missing)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for u, v in missing:
                if time.time() > deadline - 2:
                    break
                if (u, v) in edges:
                    continue
                if not find_q3_with_edge(adj, u, v):
                    edges.add((u, v))
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
                    if len(edges) > best_count:
                        best_count = len(edges)
                        best_edges = set(edges)
                    
    result = [[u, v] for (u, v) in best_edges]
    return result
