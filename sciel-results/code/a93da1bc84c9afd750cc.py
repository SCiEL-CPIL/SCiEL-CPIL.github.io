import time
import random

def _has_k222_bitset(colors, n):
    """
    Check if the coloring contains a monochromatic K2,2,2 using bitsets.
    """
    # adj[c][u] is an integer bitmask of vertices connected to u with color c
    adj = [[0] * n for _ in range(2)]
    
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
            
    # Check for K2,2,2
    # A K2,2,2 consists of 6 vertices {u,v,a,b,c,d} such that:
    # 1. u and v are connected by color c.
    # 2. a,b,c,d are common neighbors of u and v in color c.
    # 3. Among {a,b,c,d}, there is a K2,2 in color c.
    #    A K2,2 in {a,b,c,d} means there exist two disjoint pairs {x,y} and {z,w}
    #    such that all edges between {x,y} and {z,w} are color c.
    #    This is equivalent to: 
    #    Let S be the set of common neighbors.
    #    We need to find 4 vertices in S that form a K2,2.
    #    A set of 4 vertices {p,q,r,s} forms a K2,2 if and only if
    #    the number of edges in the induced subgraph is 4 and it's bipartite?
    #    Actually, K2,2 is a cycle of length 4.
    #    So we need a 4-cycle in the induced subgraph on S.
    #    
    #    Efficient check for K2,2 in a set S:
    #    Iterate over pairs (a,b) in S.
    #    Let N_a = adj[c][a] & S
    #    Let N_b = adj[c][b] & S
    #    Common = N_a & N_b
    #    Remove a and b from Common.
    #    If |Common| >= 2, then we have a K2,2.
    
    for u in range(n):
        for v in range(u+1, n):
            for c in [0, 1]:
                if not (adj[c][u] & (1 << v)):
                    continue
                common = adj[c][u] & adj[c][v]
                # Remove u and v from common (they are not neighbors of themselves)
                # common already excludes u and v because adj[c][u] doesn't have bit u.
                # Wait, adj[c][u] has bit v if edge (u,v) is color c.
                # We want neighbors w != u, v.
                # common includes v? No, adj[c][u] has v, adj[c][v] has u.
                # common = adj[c][u] & adj[c][v].
                # If edge (u,v) is color c, then v is in adj[c][u] and u is in adj[c][v].
                # So common contains u and v?
                # adj[c][u] contains v. adj[c][v] contains u.
                # So common contains u and v.
                # We must remove u and v from common to get the set of other common neighbors.
                common &= ~(1 << u)
                common &= ~(1 << v)
                
                # Now common is the set of vertices w such that (u,w) and (v,w) are color c.
                # We need to find a K2,2 in the subgraph induced by common.
                # This requires 4 vertices a,b,c,d in common such that
                # edges (a,c), (a,d), (b,c), (b,d) are color c.
                # This is equivalent to finding a pair (a,b) in common such that
                # the number of common neighbors of a and b (within common) is at least 2.
                
                # Get list of vertices in common for iteration
                # Bit iteration
                verts = []
                mask = common
                while mask:
                    lsb = mask & -mask
                    idx_v = lsb.bit_length() - 1
                    verts.append(idx_v)
                    mask ^= lsb
                    
                if len(verts) < 4:
                    continue
                    
                found = False
                for i_idx in range(len(verts)):
                    a = verts[i_idx]
                    n_a = adj[c][a] & common
                    for j_idx in range(i_idx+1, len(verts)):
                        b = verts[j_idx]
                        n_b = adj[c][b] & common
                        common2 = n_a & n_b
                        # Remove a and b
                        common2 &= ~(1 << a)
                        common2 &= ~(1 << b)
                        # Count bits
                        if common2 & (common2 - 1): # at least 2 bits
                            return True
    return False

def _count_violations_bitset(colors, n):
    """
    Count the number of 'bad' configurations to guide local search.
    """
    adj = [[0] * n for _ in range(2)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
            
    count = 0
    for u in range(n):
        for v in range(u+1, n):
            for c in [0, 1]:
                if not (adj[c][u] & (1 << v)):
                    continue
                common = adj[c][u] & adj[c][v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                
                verts = []
                mask = common
                while mask:
                    lsb = mask & -mask
                    idx_v = lsb.bit_length() - 1
                    verts.append(idx_v)
                    mask ^= lsb
                    
                if len(verts) < 4:
                    continue
                    
                for i_idx in range(len(verts)):
                    a = verts[i_idx]
                    n_a = adj[c][a] & common
                    for j_idx in range(i_idx+1, len(verts)):
                        b = verts[j_idx]
                        n_b = adj[c][b] & common
                        common2 = n_a & n_b
                        common2 &= ~(1 << a)
                        common2 &= ~(1 << b)
                        if common2 & (common2 - 1):
                            count += 1
                            break # Count per (u,v,c) pair to keep it simple, or break out entirely?
                            # Breaking inner loops to count 1 per (u,v,c) is faster.
                            # Let's break all the way out for this (u,v,c)
                            break
                    else:
                        continue
                    break
    return count

def _paley_coloring(n):
    """
    Generate Paley coloring for prime n = 4k+1.
    """
    if n % 4 != 1:
        return None
    squares = {x*x % n for x in range(1, n)}
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            if (j - i) % n in squares:
                colors.append(0)
            else:
                colors.append(1)
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s margin for safety
    
    # 1. Try to find n=30 coloring using local search
    # This is the "improvement" attempt.
    n_target = 30
    num_edges = n_target * (n_target - 1) // 2
    
    # Initialize with random or heuristic
    # Random restarts are usually effective for small n
    best_cost = float('inf')
    best_colors = None
    
    # Run a few local searches for n=30
    # Budget: 60 seconds
    search_deadline = start_time + 60
    
    for restart in range(100):
        if time.time() > search_deadline:
            break
            
        if restart == 0:
            # Start with a structured random? Or just random.
            curr = [random.choice([0,1]) for _ in range(num_edges)]
        else:
            curr = [random.choice([0,1]) for _ in range(num_edges)]
            
        curr_cost = _count_violations_bitset(curr, n_target)
        
        if curr_cost < best_cost:
            best_cost = curr_cost
            best_colors = curr[:]
            
        if curr_cost == 0:
            return {'n': n_target, 'colors': curr}
            
        # Local Search (Simulated Annealing)
        temp = 1.0
        no_improve_count = 0
        max_no_improve = 2000
        
        for step in range(50000):
            if time.time() > search_deadline:
                break
                
            edge_idx = random.randint(0, num_edges-1)
            old_c = curr[edge_idx]
            new_c = 1 - old_c
            curr[edge_idx] = new_c
            
            new_cost = _count_violations_bitset(curr, n_target)
            
            if new_cost < curr_cost:
                curr_cost = new_cost
                no_improve_count = 0
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_colors = curr[:]
                if new_cost == 0:
                    return {'n': n_target, 'colors': curr}
            else:
                delta = new_cost - curr_cost
                if delta == 0:
                    curr_cost = new_cost
                    no_improve_count += 1
                else:
                    prob = pow(2.71828, -delta/temp)
                    if random.random() > prob:
                        # Reject
                        curr[edge_idx] = old_c
                        no_improve_count += 1
                    else:
                        curr_cost = new_cost
                        no_improve_count += 1
                        
            temp *= 0.999
            if temp < 0.01:
                temp = 0.01
                
            if no_improve_count > max_no_improve:
                break
                
    # 2. Fall back to n=29 Paley coloring
    # This is the guaranteed valid solution for the current record.
    n_fallback = 29
    colors_29 = _paley_coloring(n_fallback)
    
    if colors_29 is not None:
        # Verify it (just in case)
        if not _has_k222_bitset(colors_29, n_fallback):
            return {'n': n_fallback, 'colors': colors_29}
            
    # 3. Fallback to n=17 if even that fails (should not happen)
    n_fallback2 = 17
    colors_17 = _paley_coloring(n_fallback2)
    if colors_17 is not None and not _has_k222_bitset(colors_17, n_fallback2):
        return {'n': n_fallback2, 'colors': colors_17}
        
    return {'n': 1, 'colors': []}
