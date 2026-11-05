import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    def generate_paley(q):
        """Generate Paley graph of order q (q ≡ 1 mod 4)."""
        residues = set()
        for x in range(1, q):
            residues.add((x * x) % q)
        
        adj = np.zeros((q, q), dtype=bool)
        for i in range(q):
            for j in range(i + 1, q):
                diff = (j - i) % q
                if diff in residues:
                    adj[i, j] = True
                    adj[j, i] = True
        return adj

    def build_masks(adj, n):
        """Build bitmask representation of adjacency."""
        masks = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    masks[i] |= (1 << j)
                    masks[j] |= (1 << i)
        return masks

    def count_k222_violations_fast(adj, n, masks):
        """
        Count K2,2,2 violations in both color classes using bitmasks.
        Returns (violations_color0, violations_color1).
        """
        full_mask = (1 << n) - 1
        # masks1[i] = neighbors of i in color 1
        masks1 = [0] * n
        for i in range(n):
            masks1[i] = (full_mask ^ (1 << i)) & ~masks[i]
        
        def count_violations_for_color(masks_c, adj_c):
            """
            Count K2,2,2 in color class c.
            For each pair (a,b) NOT in color c (i.e., in other color),
            find common neighbors in color c.
            For each pair (c,d) of common neighbors NOT in color c,
            count pairs (e,f) in common neighbors of c,d in color c that are NOT in color c.
            """
            violations = 0
            for a in range(n):
                for b in range(a + 1, n):
                    # (a,b) must be in the OTHER color
                    if adj_c[a, b]:
                        continue
                    
                    # Common neighbors of a and b in color c
                    cn_ab = masks_c[a] & masks_c[b]
                    
                    # Get vertices in cn_ab
                    verts = []
                    temp = cn_ab
                    while temp:
                        lsb = temp & -temp
                        verts.append(lsb.bit_length() - 1)
                        temp ^= lsb
                    
                    # For each pair (c,d) in verts that is in other color
                    for i_c in range(len(verts)):
                        c = verts[i_c]
                        for i_d in range(i_c + 1, len(verts)):
                            d = verts[i_d]
                            if adj_c[c, d]:
                                continue
                            
                            # Common neighbors of c and d in color c, within cn_ab
                            common_4 = cn_ab & masks_c[c] & masks_c[d]
                            
                            # Count pairs (e,f) in common_4 that are in other color
                            verts_common = []
                            temp = common_4
                            while temp:
                                lsb = temp & -temp
                                verts_common.append(lsb.bit_length() - 1)
                                temp ^= lsb
                            
                            for i_e in range(len(verts_common)):
                                e = verts_common[i_e]
                                for i_f in range(i_e + 1, len(verts_common)):
                                    f = verts_common[i_f]
                                    if not adj_c[e, f]:
                                        violations += 1
            return violations
        
        # Count violations in color 0
        viol0 = count_violations_for_color(masks, adj)
        # Count violations in color 1
        viol1 = count_violations_for_color(masks1, ~adj)
        
        return viol0 + viol1

    def is_valid_adj(n, adj):
        """Check if coloring is valid (no K2,2,2 in either color)."""
        masks = build_masks(adj, n)
        return count_k222_violations_fast(adj, n, masks) == 0

    def adj_to_colors(n, adj):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(1 if adj[i, j] else 0)
        return colors

    def colors_to_adj(n, colors):
        adj = np.zeros((n, n), dtype=bool)
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[k]:
                    adj[i, j] = True
                    adj[j, i] = True
                k += 1
        return adj

    # Step 1: Generate Paley(29) as baseline
    n_base = 29
    adj_base = generate_paley(n_base)
    
    if not is_valid_adj(n_base, adj_base):
        # Fallback: try random search for valid 29-vertex coloring
        found = False
        for _ in range(100):
            if time.time() > deadline - 30:
                break
            adj_try = np.zeros((n_base, n_base), dtype=bool)
            for i in range(n_base):
                for j in range(i + 1, n_base):
                    adj_try[i, j] = random.random() < 0.5
                    adj_try[j, i] = adj_try[i, j]
            if is_valid_adj(n_base, adj_try):
                adj_base = adj_try
                found = True
                break
        if not found:
            # Last resort: n=17
            n_base = 17
            adj_base = generate_paley(n_base)
    
    best_adj = adj_base.copy()
    best_n = n_base
    
    # Step 2: Try to extend to n=30
    n_target = 30
    if best_n == 29 and time.time() < deadline - 20:
        masks_base = build_masks(adj_base, n_base)
        
        # Precompute for each vertex i: the set of vertices that form K2,2,2 with i
        # When we add a new vertex v with connection pattern conn,
        # a violation occurs if there exist a,b,c,d in {0,...,28} such that
        # the 6 edges among {v,a,b,c,d} form K2,2,2 with v in one of the parts.
        
        # Actually, let's use a different approach:
        # For each possible connection pattern conn (bitmask of length 29),
        # count violations in the extended graph.
        # We can precompute which subsets of vertices are "dangerous".
        
        # A K2,2,2 involving the new vertex v must have v in one of the three parts.
        # Case 1: v is in part A with another vertex a.
        #   Then we need b,c,d such that:
        #   - (v,a) is in other color (not in conn for v-a, and not in adj for a)
        #   Wait, this is getting complex. Let me just use the violation counter.
        
        # Strategy: local search on conn (29-bit integer)
        # Cost = number of K2,2,2 violations in extended graph
        
        def count_violations_extended(conn):
            """Count K2,2,2 violations when adding vertex 29 with connection pattern conn."""
            n = n_base + 1
            adj_new = np.zeros((n, n), dtype=bool)
            adj_new[:n_base, :n_base] = adj_base
            for i in range(n_base):
                if conn & (1 << i):
                    adj_new[i, n_base] = True
                    adj_new[n_base, i] = True
            
            masks_new = build_masks(adj_new, n)
            return count_k222_violations_fast(adj_new, n, masks_new)
        
        # Try to find a conn with 0 violations
        found_30 = False
        best_conn = None
        best_viol = float('inf')
        
        # Random restarts with local search
        for restart in range(200):
            if time.time() > deadline - 10:
                break
            
            # Initialize conn randomly
            conn = random.getrandbits(n_base)
            
            # Local search: try flipping individual bits
            improved = True
            while improved:
                if time.time() > deadline - 10:
                    break
                improved = False
                current_viol = count_violations_extended(conn)
                if current_viol == 0:
                    found_30 = True
                    best_conn = conn
                    break
                if current_viol < best_viol:
                    best_viol = current_viol
                    best_conn = conn
                
                # Try flipping each bit
                for i in range(n_base):
                    new_conn = conn ^ (1 << i)
                    new_viol = count_violations_extended(new_conn)
                    if new_viol < current_viol:
                        conn = new_conn
                        current_viol = new_viol
                        improved = True
                        if current_viol == 0:
                            found_30 = True
                            best_conn = conn
                            break
                    if time.time() > deadline - 10:
                        break
                if found_30:
                    break
            
            if found_30:
                break
            
            # If stuck, perturb
            if best_viol > 0 and restart > 20:
                for _ in range(3):
                    i = random.randint(0, n_base - 1)
                    conn ^= (1 << i)
        
        if found_30:
            # Build full adjacency
            adj_full = np.zeros((n_target, n_target), dtype=bool)
            adj_full[:n_base, :n_base] = adj_base
            for i in range(n_base):
                if best_conn & (1 << i):
                    adj_full[i, n_base] = True
                    adj_full[n_base, i] = True
            
            if is_valid_adj(n_target, adj_full):
                best_adj = adj_full
                best_n = n_target
    
    # Final verification
    if not is_valid_adj(best_n, best_adj):
        best_n = 17
        best_adj = generate_paley(17)
    
    colors = adj_to_colors(best_n, best_adj)
    return {'n': best_n, 'colors': colors}
