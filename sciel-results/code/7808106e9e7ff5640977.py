import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Try n from 47 down to 39
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
            
        min_deg = n - 39
        if min_deg < 0:
            min_deg = 0
        
        # For n=39, min_deg=0, empty graph works
        if min_deg == 0:
            return {'n': n, 'edges': []}
        
        # Try multiple restarts for this n
        max_restarts = 5000
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Bitmask adjacency
            adj = [0] * n
            degrees = [0] * n
            edges = []
            
            # Helper: check if adding (u,v) creates a C4
            # C4 exists iff ∃w∈N(u)∩N(v) (common neighbor) — wait no
            # C4 is u-a-b-v where a~u, a~b, b~v, u~v (new)
            # Actually: adding edge (u,v) creates C4 iff there exists a path u-w-x-v 
            # of length 3 in the current graph, i.e., ∃w such that w~u and ∃x such that x~w and x~v
            # This is equivalent to: ∃w∈N(u) such that N(w)∩N(v) ≠ ∅
            # Which means: adj[w] & adj[v] ≠ 0 for some w in N(u)
            
            def can_add(u, v):
                if u == v:
                    return False
                if (adj[u] >> v) & 1:
                    return False
                # Check all neighbors of u
                m = adj[u]
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    if adj[w] & adj[v]:
                        return False
                    m ^= lsb
                return True
            
            # Greedy construction with local search
            # Phase 1: Greedy edge addition
            added_any = True
            while added_any and time.time() < deadline:
                added_any = False
                
                # Find vertices below target degree
                low_verts = [i for i in range(n) if degrees[i] < min_deg]
                
                if not low_verts:
                    break
                
                # Most constrained first (fewest non-neighbors)
                u = min(low_verts, key=lambda x: n - 1 - degrees[x])
                
                # Try non-neighbors, prefer lower-degree vertices
                non_neighbors = [v for v in range(n) if v != u and not ((adj[u] >> v) & 1)]
                if not non_neighbors:
                    break
                non_neighbors.sort(key=lambda v: degrees[v])
                
                for v in non_neighbors:
                    if can_add(u, v):
                        # Add edge
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        degrees[u] += 1
                        degrees[v] += 1
                        edges.append([u, v])
                        added_any = True
                        break
            
            # Check if we met the requirement
            if min(degrees) >= min_deg:
                # Verify C4-free
                c4_free = True
                for i in range(n):
                    if not c4_free:
                        break
                    for j in range(i + 1, n):
                        if not c4_free:
                            break
                        if (adj[i] >> j) & 1:
                            ai = adj[i] & ~(1 << j)
                            aj = adj[j] & ~(1 << i)
                            m = ai
                            while m:
                                lsb = m & -m
                                a = lsb.bit_length() - 1
                                if adj[a] & aj:
                                    c4_free = False
                                    break
                                m ^= lsb
                if c4_free:
                    return {'n': n, 'edges': edges}
            
            # Phase 2: Local search - try to fix deficient vertices
            if time.time() < deadline:
                for ls_iter in range(50):
                    if time.time() > deadline:
                        break
                    
                    # Find deficient vertices
                    deficient = [i for i in range(n) if degrees[i] < min_deg]
                    if not deficient:
                        break
                    
                    # Most constrained deficient vertex
                    u = min(deficient, key=lambda x: n - 1 - degrees[x])
                    
                    # Find potential partners for u
                    non_neighbors = [v for v in range(n) if v != u and not ((adj[u] >> v) & 1)]
                    non_neighbors.sort(key=lambda v: degrees[v])
                    
                    improved = False
                    for v in non_neighbors:
                        if can_add(u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            degrees[u] += 1
                            degrees[v] += 1
                            edges.append([u, v])
                            improved = True
                            break
                    
                    if not improved:
                        break
            
            # Final check
            if min(degrees) >= min_deg:
                c4_free = True
                for i in range(n):
                    if not c4_free:
                        break
                    for j in range(i + 1, n):
                        if not c4_free:
                            break
                        if (adj[i] >> j) & 1:
                            ai = adj[i] & ~(1 << j)
                            aj = adj[j] & ~(1 << i)
                            m = ai
                            while m:
                                lsb = m & -m
                                a = lsb.bit_length() - 1
                                if adj[a] & aj:
                                    c4_free = False
                                    break
                                m ^= lsb
                if c4_free:
                    return {'n': n, 'edges': edges}
        
        # If we exhausted restarts for this n, try smaller n
    
    # Fallback: return the smallest valid graph
    # For n=39, min_deg=0, empty graph works
    return {'n': 39, 'edges': []}
