import time
import random

def construct():
    """
    Return a C4-free graph with alpha < 11, maximizing n.
    Strategy: 
    1. Construct the polarity graph of the projective plane of order 5 (n=31).
    2. Iteratively add vertices by finding valid neighbor sets S.
    3. Check independence number < 11 using a fast branch and bound.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    # --- Helper: Fast Alpha Solver using Bitmasks ---
    def max_independent_set_size(n, adj):
        """
        Find the size of the maximum independent set using branch and bound.
        adj is a list of integers (bitmasks).
        """
        best = [0]
        
        def popcount(x):
            return bin(x).count('1')

        def branch(candidates_mask, current_size):
            if time.time() > deadline - 5:
                return
            if candidates_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Upper bound
            if current_size + popcount(candidates_mask) <= best[0]:
                return
            
            # Heuristic: Pick vertex with highest degree in the candidate subgraph
            best_v = -1
            best_deg = -1
            m = candidates_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                # Degree in candidate subgraph
                deg = popcount(candidates_mask & adj[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                m -= lsb
            
            if best_v == -1:
                return

            v_bit = 1 << best_v
            
            # Option 1: Include best_v
            # New candidates: those in candidates_mask not adjacent to best_v
            new_candidates = candidates_mask & ~adj[best_v] & ~v_bit
            branch(new_candidates, current_size + 1)
            
            # Option 2: Exclude best_v
            branch(candidates_mask & ~v_bit, current_size)

        branch((1 << n) - 1, 0)
        return best[0]

    def is_valid_neighbor_set(S_mask, adj, n):
        """
        Check if S_mask is a valid set of neighbors for a new vertex.
        Condition: For any u, v in S_mask, if u ~ v, then N(u) & N(v) == 0.
        """
        # Iterate over pairs in S_mask
        m = S_mask
        while m:
            lsb_u = m & -m
            u = lsb_u.bit_length() - 1
            m -= lsb_u
            
            # Check against other vertices in S_mask
            m2 = S_mask & ~lsb_u
            while m2:
                lsb_v = m2 & -m2
                v = lsb_v.bit_length() - 1
                m2 -= lsb_v
                
                # If u and v are adjacent
                if adj[u] & (1 << v):
                    # They must have no common neighbors
                    if adj[u] & adj[v]:
                        return False
        return True

    # --- Step 1: Construct Polarity Graph of PG(2,5) ---
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make first non-zero component 1
                if x != 0:
                    inv_x = pow(x, -1, 5)
                    points.append((1, y * inv_x % 5, z * inv_x % 5))
                elif y != 0:
                    inv_y = pow(y, -1, 5)
                    points.append((0, 1, z * inv_y % 5))
                else:
                    inv_z = pow(z, -1, 5)
                    points.append((0, 0, 1))
    
    points = list(set(points))
    n_base = len(points) # Should be 31
    
    def bilinear(u, v):
        return (u[0]*v[2] + u[1]*v[1] + u[2]*v[0]) % 5

    adj = [0] * n_base
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if bilinear(points[i], points[j]) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Verify base graph
    if max_independent_set_size(n_base, adj) >= 11:
        return {'n': 30, 'edges': []}

    best_n = n_base
    best_adj = adj[:]

    # --- Step 2: Iterative Growth ---
    # We try to add vertices until we can't or time runs out.
    # To be robust, we try multiple strategies for picking S.
    
    current_n = n_base
    current_adj = adj[:]
    
    for target_n in range(n_base + 1, 44):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Try to find a valid S for the new vertex
        # Strategy: Greedy selection of vertices for S.
        # We want S to be large to keep alpha low? No, we want S to be such that
        # the new vertex doesn't create an independent set of size 11.
        # Actually, adding a vertex with many neighbors tends to reduce alpha or keep it same.
        # Adding a vertex with few neighbors might increase alpha.
        # So we prefer large S.
        
        # Let's try to build S greedily.
        # Start with all vertices, remove those that conflict.
        # Conflict: u, v cannot both be in S if u~v and N(u)&N(v) != 0.
        
        # We can use a greedy set packing / independent set approach on a "conflict graph"
        # where u, v are connected if they cannot be in S together.
        # u, v conflict if (u~v and N(u)&N(v) != 0).
        
        # To maximize flexibility, we try different orderings.
        
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Build conflict graph for S-selection
            # conflict[i] bitmask of vertices that cannot be in S with i
            conflict = [0] * current_n
            for i in range(current_n):
                for j in range(i + 1, current_n):
                    if current_adj[i] & (1 << j): # i and j are adjacent
                        if current_adj[i] & current_adj[j]: # and have common neighbor
                            conflict[i] |= (1 << j)
                            conflict[j] |= (1 << i)
            
            # Find a large independent set in this conflict graph
            # This S will be the neighbor set of the new vertex.
            # Greedy with randomization
            random.seed(12345 + attempt)
            candidates = (1 << current_n) - 1
            S_mask = 0
            
            while candidates:
                best_v = -1
                min_conflicts = current_n
                m = candidates
                verts = []
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    verts.append(v)
                    m -= lsb
                random.shuffle(verts)
                
                for v in verts:
                    if not (candidates & (1 << v)):
                        continue
                    conf_count = bin(conflict[v] & candidates).count('1')
                    if conf_count < min_conflicts:
                        min_conflicts = conf_count
                        best_v = v
                        if min_conflicts == 0:
                            break
                
                if best_v == -1:
                    break
                    
                S_mask |= (1 << best_v)
                candidates &= ~(1 << best_v)
                candidates &= ~conflict[best_v]
            
            # S_mask is the set of neighbors for the new vertex
            # Check if S_mask is valid (it should be by construction, but double check)
            if not is_valid_neighbor_set(S_mask, current_adj, current_n):
                continue
            
            # Add vertex
            new_adj = current_adj[:] + [0]
            new_adj[current_n] = S_mask
            for i in range(current_n):
                if S_mask & (1 << i):
                    new_adj[i] |= (1 << current_n)
            
            # Check Alpha
            alpha = max_independent_set_size(current_n + 1, new_adj)
            if alpha < 11:
                current_adj = new_adj
                current_n += 1
                found = True
                break
        
        if found:
            best_n = current_n
            best_adj = current_adj[:]
        else:
            # If we couldn't add a vertex with this strategy, try another?
            # Or break?
            # Let's try a few more random seeds before breaking
            # But if we've tried 20 and failed, it's likely stuck.
            # However, maybe a smaller S works?
            # The greedy above tries to find a *large* S.
            # Sometimes a smaller S allows the graph to grow further?
            # Unlikely, larger S usually helps reduce alpha.
            break

    # Convert best_adj to edge list
    edge_list = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edge_list.append([i, j])

    return {'n': best_n, 'edges': edge_list}
