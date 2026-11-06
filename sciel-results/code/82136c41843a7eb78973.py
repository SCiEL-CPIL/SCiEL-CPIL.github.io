import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Try n from 17 down to 13
    for n in range(17, 12, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Allocate time based on n
        # 17 is hardest, give it more time
        time_limit = 40.0 if n == 17 else (20.0 if n == 16 else 10.0)
        target_end = time.time() + time_limit
        
        result = _local_search(n, target_end)
        if result:
            return result
            
    # Fallback: return a known valid solution for n=13 if search failed
    # Or just n=9 seed
    return _get_fallback_solution()

def _local_search(n, deadline):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Initialize random colors
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # adj[c][u] is bitmask of neighbors of u in color c
    adj = [[0] * n for _ in range(4)]
    
    # common_neighbors[c][i][j] is number of common neighbors of i and j in color c
    # This is O(4 * n * n) memory
    common_neighbors = [[[0] * n for _ in range(n)] for _ in range(4)]
    
    # Helper to update common_neighbors when adding/removing edge (u,v) to/from color c
    def update_common_neighbors(c, u, v, delta):
        # delta is +1 for add, -1 for remove
        # For each z, if z is connected to u, then z and v gain/lose a common neighbor u? No.
        # If we add (u,v), then for any z connected to u, z and v now share common neighbor u.
        # So common_neighbors[z][v] += 1 and common_neighbors[v][z] += 1.
        # Similarly for z connected to v.
        nu = adj[c][u]
        nv = adj[c][v]
        
        # Iterate neighbors of u
        z_mask = nu
        while z_mask:
            lsb = z_mask & -z_mask
            z = lsb.bit_length() - 1
            z_mask ^= lsb
            common_neighbors[c][z][v] += delta
            common_neighbors[c][v][z] += delta
            
        # Iterate neighbors of v
        z_mask = nv
        while z_mask:
            lsb = z_mask & -z_mask
            z = lsb.bit_length() - 1
            z_mask ^= lsb
            common_neighbors[c][u][z] += delta
            common_neighbors[c][z][u] += delta
            
        # Also, u and v become common neighbors of each other? No, they are connected.
        # Do u and v share common neighbors?
        # If z is connected to both u and v, then z is a common neighbor of u and v.
        # Adding (u,v) doesn't change the set of common neighbors of u and v, 
        # but it changes the count of paths of length 2 between other pairs.
        # Wait, common_neighbors[i][j] is count of k such that (i,k) and (j,k) exist.
        # Adding (u,v) affects:
        # 1. Pairs (z, v) where (z,u) exists. z and v now share u.
        # 2. Pairs (z, u) where (z,v) exists. z and u now share v.
        # 3. Does it affect (u,v)? No, (u,v) is not a common neighbor of itself.
        # 4. Does it affect (u, z) where z is connected to v? Yes, covered by 2.
        
        # What about the pair (u,v) themselves?
        # common_neighbors[u][v] is count of k connected to both u and v.
        # Adding edge (u,v) does NOT change which vertices are connected to u or v.
        # So common_neighbors[u][v] is unchanged.
        
        # So the update above is correct.

    # Initialize adj and common_neighbors
    for idx, (u, v) in enumerate(edges):
        c = colors[idx]
        adj[c][u] |= (1 << v)
        adj[c][v] |= (1 << u)
        
    # Initialize common_neighbors by scanning all edges
    # This is O(E * n) which is fine for init
    for c in range(4):
        for u in range(n):
            # Get neighbors of u in color c
            nu = adj[c][u]
            # For each pair of neighbors v1, v2, they share u as common neighbor
            # This is inefficient. Better:
            # For each edge (u,v) in color c, update common_neighbors for neighbors of u and v.
            pass
            
    # Let's just compute common_neighbors from scratch using bitsets for speed
    for c in range(4):
        for i in range(n):
            for j in range(i + 1, n):
                # Count common neighbors of i and j in color c
                mask = adj[c][i] & adj[c][j]
                cnt = bin(mask).count('1')
                common_neighbors[c][i][j] = cnt
                common_neighbors[c][j][i] = cnt

    # Function to count C6s involving edge (u,v) in color c
    # Path u-a-b-d-v? No, u-a-b-c-d-v.
    # We established: Sum over a in N(u), d in N(v) of valid_paths(a,d)
    # valid_paths(a,d) = common_neighbors[a][d] - (1 if u in N(d) else 0) - (1 if v in N(a) else 0)
    # Note: u is in N(a) by definition. v is in N(d) by definition.
    # So we subtract 1 if u is connected to d in color c.
    # And subtract 1 if v is connected to a in color c.
    
    def count_c6(u, v, c):
        nu = adj[c][u] & ~(1 << v)
        nv = adj[c][v] & ~(1 << u)
        
        count = 0
        # Iterate a in nu
        a_mask = nu
        while a_mask:
            lsb_a = a_mask & -a_mask
            a = lsb_a.bit_length() - 1
            a_mask ^= lsb_a
            
            # Iterate d in nv
            d_mask = nv
            while d_mask:
                lsb_d = d_mask & -d_mask
                d = lsb_d.bit_length() - 1
                d_mask ^= lsb_d
                
                if a == d:
                    continue
                    
                paths = common_neighbors[c][a][d]
                
                # Subtract if u is a common neighbor of a and d
                # i.e., if (u,d) is an edge in color c
                if adj[c][u] & (1 << d):
                    paths -= 1
                    
                # Subtract if v is a common neighbor of a and d
                # i.e., if (v,a) is an edge in color c
                if adj[c][v] & (1 << a):
                    paths -= 1
                    
                if paths > 0:
                    count += paths
                    
        return count

    # Calculate initial total violations
    total_violations = 0
    for idx, (u, v) in enumerate(edges):
        c = colors[idx]
        total_violations += count_c6(u, v, c)
        
    # If 0, we are done
    if total_violations == 0:
        return {'n': n, 'colors': colors}

    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    best_violations = total_violations
    best_colors = colors[:]
    
    # Precompute edge index map for fast access? We have edges list.
    # We need to map (u,v) to index.
    edge_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx
        
    for step in range(100000):
        if time.time() > deadline:
            break
            
        # Pick random edge
        idx = random.randint(0, num_edges - 1)
        u, v = edges[idx]
        old_c = colors[idx]
        
        # Pick new color
        new_c = random.randint(0, 3)
        if new_c == old_c:
            continue
            
        # Calculate delta violations
        # Remove (u,v) from old_c
        # Count C6s in G_old_c that use (u,v)
        # This is count_c6(u, v, old_c) BUT we must remove the edge first from adj/common_neighbors?
        # No, count_c6 uses current adj. The edge (u,v) is present.
        # The C6s counted INCLUDE the edge (u,v)?
        # My count_c6 function counts paths of length 5 between u and v.
        # Adding (u,v) creates a C6 for each such path.
        # So yes, count_c6(u, v, old_c) gives the number of C6s that will be REMOVED.
        
        remove_count = count_c6(u, v, old_c)
        
        # Temporarily remove (u,v) from old_c
        adj[old_c][u] &= ~(1 << v)
        adj[old_c][v] &= ~(1 << u)
        update_common_neighbors(old_c, u, v, -1)
        
        # Add (u,v) to new_c
        # Count C6s in G_new_c that use (u,v)
        add_count = count_c6(u, v, new_c)
        
        # Temporarily add (u,v) to new_c
        adj[new_c][u] |= (1 << v)
        adj[new_c][v] |= (1 << u)
        update_common_neighbors(new_c, u, v, 1)
        
        delta_v = add_count - remove_count
        total_violations += delta_v
        
        # Accept?
        if delta_v <= 0 or random.random() < pow(2.718, -delta_v / T):
            colors[idx] = new_c
            if total_violations < best_violations:
                best_violations = total_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return {'n': n, 'colors': best_colors}
        else:
            # Reject: revert
            # Remove from new_c
            adj[new_c][u] &= ~(1 << v)
            adj[new_c][v] &= ~(1 << u)
            update_common_neighbors(new_c, u, v, -1)
            
            # Add to old_c
            adj[old_c][u] |= (1 << v)
            adj[old_c][v] |= (1 << u)
            update_common_neighbors(old_c, u, v, 1)
            
            total_violations -= delta_v
            colors[idx] = old_c
            
        T *= cooling
        
    if best_violations == 0:
        return {'n': n, 'colors': best_colors}
        
    return None

def _get_fallback_solution():
    # Return a valid solution for n=13
    # I will use a hardcoded pattern or a quick greedy
    # For now, return n=9 seed to be safe, but I expect local search to find n>=13
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}
