import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_n = 39
    best_edges = []
    
    # Helper to convert bitmask to list of indices
    def bits_to_list(mask):
        lst = []
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            lst.append(idx)
            mask ^= lsb
        return lst

    # Check if adding edge (u, v) creates a C4
    # A C4 is u-w-x-v-u. 
    # This exists if there is w in N(u) and x in N(v) such that (w,x) is an edge.
    # Note: w cannot be v, x cannot be u.
    # Since (u,v) is not yet an edge, v is not in N(u) and u is not in N(v).
    # So we just need to check if there exists w in N(u) such that N(w) intersects N(v).
    def creates_c4(adj, u, v):
        au = adj[u]
        av = adj[v]
        
        if au == 0 or av == 0:
            return False
            
        # Iterate over neighbors of u
        # We can optimize by checking if the intersection of N(w) and N(v) is non-empty
        # for any w in N(u).
        # Actually, a C4 is formed if there is a path of length 2 from u to v that is 
        # part of a 4-cycle? No.
        # A C4 involving u and v as opposite vertices? No, u and v are adjacent in the new edge.
        # The C4 would be u - w - x - v - u.
        # So w must be neighbor of u, x must be neighbor of v, and (w,x) must be an edge.
        # Also w != v, x != u (automatic since not edges yet).
        # And w != x.
        
        # Let's iterate over w in N(u)
        temp = au
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            # Check if any neighbor of w is a neighbor of v
            # Exclude u and v from the check? 
            # adj[w] & adj[v] gives common neighbors of w and v.
            # If common neighbor is x, then we have u-w-x-v.
            # We need to ensure x != u. Since u is not neighbor of v, u is not in adj[v].
            # We need to ensure x != v. v is not in adj[v] (no self loops).
            # We need to ensure x != w. w is not in adj[w].
            # So simply checking if (adj[w] & adj[v]) != 0 is sufficient?
            # Wait, if x = w, then adj[w] has bit w? No, no self loops.
            # So yes, if adj[w] & adj[v] is non-zero, there is an x.
            if adj[w] & av:
                return True
            temp ^= lsb
            
        return False

    def try_construct(n, max_attempts=200, time_limit=None):
        min_deg_req = n - 39
        if min_deg_req <= 0:
            return True, []
        
        for attempt in range(max_attempts):
            if time_limit and time.time() > time_limit:
                return False, None
            
            adj = [0] * n
            
            # 1. Greedy Phase
            # Create list of all potential edges
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                if time_limit and time.time() > time_limit:
                    return False, None
                
                if not creates_c4(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Calculate current min degree
            degrees = [bin(adj[i]).count('1') for i in range(n)]
            min_deg = min(degrees)
            
            # 2. Local Search Phase (Polish)
            # If min degree is not met, try to add more edges to low-degree vertices
            # We will iterate a number of times, picking a random vertex below target,
            # and trying to add edges to its non-neighbors.
            
            if min_deg < min_deg_req:
                # Limit local search time
                ls_deadline = time.time() + 2.0 # 2 seconds per attempt for local search
                max_ls_steps = 5000
                
                for step in range(max_ls_steps):
                    if time.time() > ls_deadline or time_limit and time.time() > time_limit:
                        break
                    
                    # Find vertices with degree < min_deg_req
                    # To be efficient, we can just pick a random vertex and check
                    # Or maintain a list. Given n is small, picking random is fine.
                    # Let's pick the vertex with the lowest degree to prioritize repair.
                    # Actually, randomizing helps escape local minima.
                    # Let's pick a random vertex that is below requirement.
                    
                    # Simple approach: pick random u
                    u = random.randint(0, n-1)
                    if degrees[u] >= min_deg_req:
                        continue
                        
                    # Find non-neighbors of u
                    # non_neigh = [v for v in range(n) if v != u and not (adj[u] >> v) & 1]
                    # Shuffled to try different orders
                    non_neigh = []
                    for v in range(n):
                        if v != u and not (adj[u] & (1 << v)):
                            non_neigh.append(v)
                            
                    if not non_neigh:
                        continue
                        
                    random.shuffle(non_neigh)
                    
                    added_any = False
                    for v in non_neigh:
                        if creates_c4(adj, u, v):
                            continue
                        # Add edge
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        degrees[u] += 1
                        degrees[v] += 1
                        added_any = True
                        # Stop adding to u if we reached requirement? 
                        # Or keep going to help others? 
                        # Let's stop for this u to balance degrees.
                        if degrees[u] >= min_deg_req:
                            break
                            
                    if not added_any:
                        # If we couldn't add any edge to u, maybe try swapping?
                        # For now, simple addition is the primary driver.
                        # If stuck, we might need to remove an edge to free up space?
                        # That's complex. Let's rely on multiple restarts (attempts).
                        pass
            
            # Final Check
            degrees = [bin(adj[i]).count('1') for i in range(n)]
            min_deg = min(degrees)
            
            if min_deg >= min_deg_req:
                # Valid graph found
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return True, edges
                
        return False, None

    # Try from n=47 down to n=40
    for target_n in [47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline - 10:
            break
            
        success, edges = try_construct(target_n, max_attempts=100, time_limit=deadline - 2)
        if success:
            best_n = target_n
            best_edges = edges
            break
            
    if best_n > 39:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
