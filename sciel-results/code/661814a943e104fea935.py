import time
import random

def construct():
    start_time = time.time()
    n = 16
    # Edge indices
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            edges.append((u, v))
    m = len(edges) # 120
    
    # Map edge to index
    edge_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i
        
    # Initialize graph with all edges? No, start with empty or K_3,13?
    # Let's start with all edges and remove?
    # Or start with empty and add?
    # Adding is safer for Q3 check.
    
    # Bitmasks for neighbors
    # neighbors[u] is an int with bit v set if (u,v) in E
    neighbors = [0] * n
    
    # Current edges as a set of indices for fast lookup?
    # Or just use neighbors to check existence.
    
    # We will use a recursive search.
    # To avoid deep recursion limits, we might need to be careful, but 120 depth is fine.
    
    best_count = 0
    best_mask = 0 # Bitmask of edges kept? Or just count and reconstruct?
    # Reconstructing from count is hard.
    # Let's store the list of kept edges.
    
    # To speed up, we maintain a list of candidate edges to add.
    # And a list of edges to skip?
    
    # Let's use the "Add" strategy.
    # Initial state: K_3,13.
    init_edges = set()
    for u in range(3):
        for v in range(3, 16):
            init_edges.add((u, v))
            
    for u, v in init_edges:
        neighbors[u] |= (1 << v)
        neighbors[v] |= (1 << u)
        
    best_count = len(init_edges)
    best_edges = list(init_edges)
    
    # Candidates: all other edges
    candidates = [e for e in edges if e not in init_edges]
    
    # Heuristic sort: 
    # Edges that connect high-degree nodes?
    # Or edges that are "critical"?
    
    def creates_q3(u, v):
        # Check if adding (u,v) creates a Q3
        # u, v are not connected yet.
        # n_u = neighbors[u]
        # n_v = neighbors[v]
        
        # We need x2, x3, x4, x5, x6, x7
        # x3, x4 in n_u
        # x2, x5 in n_v
        # (x2, x3) in E
        # (x4, x5) in E
        # x6 in n(x2) & n(x5)
        # x7 in n(x3) & n(x4) & n(x6)
        # (x6, x7) in E
        
        n_u = neighbors[u]
        n_v = neighbors[v]
        
        # Iterate x3 in n_u
        # To iterate bits in a mask:
        # while mask:
        #   lsb = mask & -mask
        #   i = lsb.bit_length() - 1
        #   mask ^= lsb
        
        # We can optimize by precomputing lists of neighbors?
        # But neighbors change.
        
        # Let's just use the loop with bit operations.
        
        # Get list of neighbors for x3, x4 etc?
        # That's expensive.
        
        # Alternative:
        # Iterate x2 in n_v
        # Iterate x3 in n_u
        # If (x2, x3) not in E, continue
        # Iterate x5 in n_v
        # If (x5, x4) ... wait, we need x4 too.
        
        # Let's iterate x2, x3, x4, x5.
        # This is O(d^4).
        
        # Can we reduce?
        # For a fixed x2, x3, x5, we need x4 such that (x4, x5) in E and x4 in n_u.
        # And then check x6, x7.
        
        # Let's try to optimize the inner check.
        
        # Precompute common neighbors for pairs?
        # Too slow to update.
        
        # We'll stick to the loop but use bitmasks for intersection.
        
        n_u_list = []
        temp = n_u
        while temp:
            lsb = temp & -temp
            n_u_list.append(lsb.bit_length() - 1)
            temp ^= lsb
            
        n_v_list = []
        temp = n_v
        while temp:
            lsb = temp & -temp
            n_v_list.append(lsb.bit_length() - 1)
            temp ^= lsb
            
        # This list generation is O(d).
        
        for x2 in n_v_list:
            for x3 in n_u_list:
                if x2 == x3: continue
                # Check (x2, x3) in E
                if not (neighbors[x2] & (1 << x3)): continue
                
                for x5 in n_v_list:
                    if x5 == x2 or x5 == x3: continue
                    # We need x4 in n_u such that (x4, x5) in E
                    # And x4 != x3
                    # And x4 != x2, x5 (implied)
                    
                    # Find candidates for x4
                    # x4 must be in n_u AND n(x5)
                    # And x4 != x3
                    common_x4 = n_u & neighbors[x5] & ~(1 << x3)
                    
                    temp_x4 = common_x4
                    while temp_x4:
                        lsb = temp_x4 & -temp_x4
                        x4 = lsb.bit_length() - 1
                        temp_x4 ^= lsb
                        
                        # Check x4 != x2, x5
                        if x4 == x2 or x4 == x5: continue
                        
                        # Now we have x2, x3, x4, x5
                        # Check distinctness:
                        # x2, x3, x4, x5 are distinct?
                        # x2, x5 in n_v. x3, x4 in n_u.
                        # n_u, n_v might overlap.
                        # We checked x2!=x3, x5!=x2, x5!=x3, x4!=x3, x4!=x2, x4!=x5.
                        # What about x2==x4? Checked.
                        # What about x5==x3? Checked.
                        # So distinct.
                        
                        # Now find x6, x7
                        # x6 in n(x2) & n(x5)
                        # x7 in n(x3) & n(x4) & n(x6)
                        # (x6, x7) in E
                        
                        common_x6 = neighbors[x2] & neighbors[x5]
                        # Exclude u, v, x2, x3, x4, x5
                        forbidden = (1 << u) | (1 << v) | (1 << x2) | (1 << x3) | (1 << x4) | (1 << x5)
                        common_x6 &= ~forbidden
                        
                        temp_x6 = common_x6
                        while temp_x6:
                            lsb = temp_x6 & -temp_x6
                            x6 = lsb.bit_length() - 1
                            temp_x6 ^= lsb
                            
                            # x7 in n(x3) & n(x4) & n(x6)
                            common_x7 = neighbors[x3] & neighbors[x4] & neighbors[x6]
                            # Exclude u, v, x2, x3, x4, x5, x6
                            forbidden7 = forbidden | (1 << x6)
                            common_x7 &= ~forbidden7
                            
                            if common_x7:
                                # There is at least one x7
                                # We just need to verify (x6, x7) in E?
                                # x7 is in n(x6) by definition of common_x7 (since n(x6) is in the intersection? No, n(x6) is neighbors[x6].
                                # Yes, common_x7 = n(x3) & n(x4) & n(x6).
                                # So (x6, x7) is in E.
                                return True
        return False

    def search(candidates, count):
        nonlocal best_count, best_edges
        
        if time.time() - start_time > 115:
            return
            
        if count + len(candidates) <= best_count:
            return
            
        if not candidates:
            if count > best_count:
                best_count = count
                # Reconstruct edges?
                # We didn't track the edges.
                # We need to track the edges.
                # This is a problem.
                # Let's pass the list of edges.
            return
            
        e = candidates[0]
        rest = candidates[1:]
        u, v = e
        
        # Try adding
        if not creates_q3(u, v):
            # Add
            neighbors[u] |= (1 << v)
            neighbors[v] |= (1 << u)
            
            search(rest, count + 1)
            
            # Undo
            neighbors[u] ^= (1 << v)
            neighbors[v] ^= (1 << u)
            
        # Try skipping
        search(rest, count)
        
    # We need to track best_edges.
    # Let's modify search to take a list of edges.
    
    def search2(candidates, count, current_edges):
        nonlocal best_count, best_edges
        
        if time.time() - start_time > 115:
            return
            
        if count + len(candidates) <= best_count:
            return
            
        if not candidates:
            if count > best_count:
                best_count = count
                best_edges = list(current_edges)
            return
            
        e = candidates[0]
        rest = candidates[1:]
        u, v = e
        
        # Try adding
        if not creates_q3(u, v):
            current_edges.append(e)
            neighbors[u] |= (1 << v)
            neighbors[v] |= (1 << u)
            
            search2(rest, count + 1, current_edges)
            
            # Undo
            current_edges.pop()
            neighbors[u] ^= (1 << v)
            neighbors[v] ^= (1 << u)
            
        # Try skipping
        search2(rest, count, current_edges)
        
    # Initial edges
    init_edge_list = [e for e in init_edges]
    search2(candidates, best_count, init_edge_list)
    
    return [list(e) for e in best_edges]
