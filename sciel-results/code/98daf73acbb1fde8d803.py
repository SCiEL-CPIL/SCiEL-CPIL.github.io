import time
import random
import math

def _paley_37_edges():
    """Paley graph on 37 vertices."""
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    return q, edges

def _build_masks(n, edges):
    """Build bitmask adjacency list."""
    masks = [0] * n
    for i, j in edges:
        masks[i] |= (1 << j)
        masks[j] |= (1 << i)
    return masks

def _count_triangles_in_mask(mask, masks):
    """
    Count the number of triangles in the subgraph induced by the vertices in 'mask'.
    mask is a bitmask of vertices.
    masks is the global adjacency list.
    """
    # Extract vertices
    verts = []
    m = mask
    while m:
        lsb = m & (-m)
        idx = lsb.bit_length() - 1
        verts.append(idx)
        m ^= lsb
    
    count = 0
    n_verts = len(verts)
    if n_verts < 3:
        return 0
        
    # For each pair (i, j) in verts, check if they share a neighbor in verts
    # To avoid double counting, we can iterate i < j and count common neighbors k > j?
    # Or just count all triangles and divide by 3? No, standard triangle counting:
    # Sum over i < j of |N(i) & N(j) & verts_mask| / 3 ?
    # Actually, each triangle is counted 3 times if we sum over all pairs.
    # Let's just iterate i < j < k for small n_verts? 
    # n_verts can be up to 42. 42 choose 3 is 11480. This is fast enough for an update?
    # We do this for every SA step. 11k ops * 100k steps = 1B ops. Might be too slow in Python.
    
    # Optimized triangle count for a set of vertices using bitmasks:
    # Iterate i in verts.
    # For each i, look at neighbors of i within the mask.
    # This is still potentially heavy.
    
    # Alternative: Precompute? No, the graph changes.
    
    # Let's use the property:
    # Triangles in S = (1/3) * sum_{u in S} |N(u) & S & N(v) for v in N(u) & S ...|
    
    # Simpler:
    # For each i in range(n_verts):
    #   v_i = verts[i]
    #   neighbors_in_s = masks[v_i] & mask
    #   # Remove v_i itself from neighbors_in_s (it's not a self loop)
    #   neighbors_in_s &= ~(1 << v_i)
    #   # Iterate through neighbors j in neighbors_in_s where j > v_i?
    #   # If we just count edges in the induced subgraph, that's not triangles.
    
    # Let's stick to the O(n^3) but with bit operations.
    # For i in range(n_verts):
    #   vi = verts[i]
    #   ni = masks[vi] & mask
    #   ni &= ~(1 << vi)
    #   # For each j in ni:
    #   m_j = ni
    #   while m_j:
    #     lsb = m_j & (-m_j)
    #     vj = lsb.bit_length() - 1
    #     m_j ^= lsb
    #     if vj > vi: # Only count once
    #       common = ni & masks[vj]
    #       # common contains vertices connected to both vi and vj
    #       # We need to count how many of these are in S (they are, by construction of ni)
    #       # But we need to exclude vi and vj? 
    #       # The triangle is (vi, vj, vk). vk must be in common.
    #       # Count bits in common.
    #       # But this counts each triangle 3 times?
    #       # If we enforce vj > vi and vk > vj, we count each once.
    #       # So: common &= ~(1 << vi) & ~(1 << vj)
    #       # Then iterate k in common where k > vj?
    #       # This is getting complicated.
        
    # Let's use a simpler heuristic for the SA energy:
    # Instead of exact triangle count, maybe we can use a different metric?
    # No, we need exact to know when E=0.
    
    # Let's optimize the triangle count.
    # n_verts is small (degree of u).
    # Average degree in 42-vertex graph is ~20.
    # 20 choose 3 is 1140.
    # 1140 * 100,000 steps = 114M operations. Python might handle this in ~30-60s?
    # Risky.
    
    # Better: Use the formula:
    # Triangles in S = sum_{i < j} popcount(N(i) & N(j) & S) / 3
    # This is O(n^2) bit operations.
    # n=20. 20^2 = 400 bit ops.
    # 400 * 100,000 = 40M bit ops. This is fast.
    
    total = 0
    for i in range(n_verts):
        vi = verts[i]
        # Only consider j > i to avoid double counting pairs?
        # If we sum over all i < j, each triangle (a,b,c) is counted 3 times:
        # Pair (a,b) sees c. Pair (a,c) sees b. Pair (b,c) sees a.
        # So we divide by 3 at the end.
        for j in range(i + 1, n_verts):
            vj = verts[j]
            # Common neighbors of vi and vj within S
            common = masks[vi] & masks[vj] & mask
            # Exclude vi and vj themselves (they are not their own neighbors)
            common &= ~(1 << vi)
            common &= ~(1 << vj)
            total += common.bit_count()
            
    return total // 3

def _energy(masks, n):
    """Calculate total energy: count of K5 + count of Ind5."""
    # K5 count
    k5_count = 0
    all_mask = (1 << n) - 1
    
    for v in range(n):
        nv = masks[v] & all_mask
        # Count triangles in N(v)
        # Each K5 containing v corresponds to a triangle in N(v) that is part of a K4?
        # No. A K5 is a set of 5 vertices.
        # If we fix v, the other 4 must form a K4 in N(v).
        # So we need to count K4's in N(v).
        # Counting K4's is harder.
        
        # Alternative:
        # A K5 is a triangle (a,b,c) such that a,b,c have a common neighbor d, 
        # and d has a common neighbor e with a,b,c?
        
        # Let's go back to the local update idea.
        # It is much faster to calculate Delta E than total E.
        # But for initialization, we need total E.
        # We can calculate total E by summing local contributions?
        # Each K5 is counted 5 times if we sum over vertices? No.
        # Each K5 is counted 10 times if we sum over edges?
        
        # Let's just calculate total E by brute force for initialization, 
        # and use Delta E for the SA loop.
        pass
    return 0 # Placeholder

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    # Target n
    target_n = 42
    
    # Initialize with Paley 37 + 5 random
    n_base, base_edges = _paley_37_edges()
    masks = _build_masks(n_base, base_edges)
    
    # Extend to 42
    for v in range(n_base, target_n):
        masks.append(0)
        # Randomly connect to previous vertices
        for u in range(v):
            if random.random() < 0.5:
                masks[v] |= (1 << u)
                masks[u] |= (1 << v)
    
    n = target_n
    all_mask = (1 << n) - 1
    
    # Precompute complement masks?
    # comp_masks[i] = all_mask ^ masks[i] ^ (1 << i)
    
    def get_delta_energy(u, v, masks, comp_masks, n):
        """
        Calculate change in energy if we FLIP edge (u,v).
        Returns delta_K5, delta_ind5.
        """
        # Current state: edge exists if masks[u] & (1 << v)
        has_edge = bool(masks[u] & (1 << v))
        
        # Common neighbors in G
        common_g = masks[u] & masks[v]
        common_g &= ~(1 << u) & ~(1 << v)
        
        # Common non-neighbors in G (neighbors in Complement)
        # comp_masks[u] are non-neighbors of u in G (excluding u)
        common_c = comp_masks[u] & comp_masks[v]
        common_c &= ~(1 << u) & ~(1 << v)
        
        # Delta K5:
        # If we ADD edge (u,v) (has_edge is False):
        # We create K5s that contain (u,v).
        # A K5 containing (u,v) requires 3 other vertices that form a triangle 
        # in the common neighborhood of u and v.
        # So delta_K5 = + count_triangles(common_g)
        
        # If we REMOVE edge (u,v) (has_edge is True):
        # We destroy K5s that contain (u,v).
        # delta_K5 = - count_triangles(common_g)
        
        # Delta Ind5 (K5 in Complement):
        # If we ADD edge (u,v) in G:
        # We REMOVE edge (u,v) in Comp.
        # This destroys Ind5s that contain (u,v) in Comp.
        # An Ind5 in G is a K5 in Comp.
        # So we destroy triangles in common_c.
        # delta_ind5 = - count_triangles(common_c)
        
        # If we REMOVE edge (u,v) in G:
        # We ADD edge (u,v) in Comp.
        # This creates Ind5s that contain (u,v) in Comp.
        # delta_ind5 = + count_triangles(common_c)
        
        tri_g = _count_triangles_in_mask(common_g, masks)
        tri_c = _count_triangles_in_mask(common_c, comp_masks)
        
        if has_edge:
            # Removing edge
            delta_k5 = -tri_g
            delta_ind5 = tri_c
        else:
            # Adding edge
            delta_k5 = tri_g
            delta_ind5 = -tri_c
            
        return delta_k5, delta_ind5

    # Initial Energy Calculation (Brute Force for simplicity, or sum of deltas from empty graph?)
    # Let's just run SA and check E=0 periodically.
    # To check E=0, we can maintain a running total.
    
    # Initialize running total
    # E = K5_count + Ind5_count
    # We can calculate initial E by summing contributions?
    # Or just start E=0 and add? No, we start with a random graph.
    # Let's calculate initial E using a fast approximation or just start SA and check validity at the end?
    # No, we need E to guide SA.
    
    # Fast Initial E:
    # Sum over all vertices v:
    # K5s containing v = K4s in N(v).
    # Ind5s containing v = K4s in Comp(N(v)).
    # Counting K4s is O(n^4) per vertex? Too slow.
    
    # Alternative:
    # Just use the SA with a "validity check" at the end of each cooling step?
    # No, we need the gradient.
    
    # Let's assume the initial E is high and just run SA.
    # We need to know if we hit 0.
    # We can maintain `current_k5` and `current_ind5`.
    # How to initialize?
    # We can pick a random edge, "remove" it (conceptually) and calculate delta?
    # This is tedious.
    
    # Let's just do a few random restarts of SA, and at the end, verify.
    # But SA needs a cost function.
    
    # Let's use a simpler cost: Number of K5s + Number of Ind5s.
    # We can estimate the change in cost locally.
    # To get the absolute cost, we can just compute it once at the start.
    # Computing K5 count:
    # For each 5-combination? 850k * 10 checks = 8.5M ops. Fast.
    
    def calculate_total_energy(masks, n):
        k5_count = 0
        ind5_count = 0
        all_mask = (1 << n) - 1
        comp_masks = [all_mask ^ masks[i] ^ (1 << i) for i in range(n)]
        
        # Count K5s
        # Iterate over all 5-combinations?
        # Or:
        # For each vertex v, count K4s in N(v).
        # A K5 is a K4 in N(v) for each of its 5 vertices?
        # No, a K5 is a K4 in N(v) for each v in the K5.
        # So sum of K4s in N(v) over all v = 5 * (# K5s).
        
        # Count K4s in a set S:
        # Sum over i<j in S of |N(i)&N(j)&S| / 3 ? No, that's triangles.
        # K4s in S:
        # For each i in S, count triangles in N(i) & S.
        # Sum / 4?
        # Each K4 (a,b,c,d) is a triangle in N(a), N(b), N(c), N(d).
        # So sum of triangles in N(v) for v in S = 4 * (# K4s in S).
        
        def count_k4_in_mask(mask, masks):
            verts = []
            m = mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                verts.append(idx)
                m ^= lsb
            if len(verts) < 4: return 0
            
            total_tris = 0
            for v in verts:
                nv_in_s = masks[v] & mask
                nv_in_s &= ~(1 << v)
                total_tris += _count_triangles_in_mask(nv_in_s, masks)
            return total_tris // 4
        
        for v in range(n):
            nv = masks[v] & all_mask
            k5_count += count_k4_in_mask(nv, masks)
            
            comp_nv = comp_masks[v] & all_mask
            ind5_count += count_k4_in_mask(comp_nv, comp_masks)
            
        return k5_count // 5 + ind5_count // 5

    # This is O(n * n^3) roughly. 42 * 42^3 = 3M. Fast enough for init.
    
    comp_masks = [all_mask ^ masks[i] ^ (1 << i) for i in range(n)]
    current_energy = calculate_total_energy(masks, n)
    
    best_energy = current_energy
    best_masks = masks.copy()
    
    T = 1.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        if current_energy == 0:
            break
            
        # Pick random edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
            
        delta_k5, delta_ind5 = get_delta_energy(u, v, masks, comp_masks, n)
        delta_e = delta_k5 + delta_ind5
        
        if delta_e < 0 or random.random() < math.exp(-delta_e / T):
            # Flip edge
            if masks[u] & (1 << v):
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
                # Update comp_masks
                comp_masks[u] |= (1 << v)
                comp_masks[v] |= (1 << u)
            else:
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                comp_masks[u] &= ~(1 << v)
                comp_masks[v] &= ~(1 << u)
                
            current_energy += delta_e
            if current_energy < best_energy:
                best_energy = current_energy
                best_masks = masks.copy()
                
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
    # Check if best is valid
    # Convert best_masks to edges
    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_masks[i] & (1 << j):
                final_edges.append((i, j))
                
    return n, final_edges

