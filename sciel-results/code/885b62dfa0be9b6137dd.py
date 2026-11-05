import time
import random

def construct():
    start = time.time()
    deadline = start + 140  # Safety margin before 150s limit
    
    def generate_paley(n):
        squares = set((x*x) % n for x in range(n))
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                d = (j - i) % n
                colors.append(1 if d in squares else 0)
        return colors
    
    def build_adj(n, colors):
        """Build adjacency bitmask list for color-1 edges."""
        adj = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if colors[idx] == 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                idx += 1
        return adj
    
    def find_k222(adj, n):
        """Return True if adj (color-1 graph) contains K_{2,2,2} as a subgraph.
        K_{2,2,2} = octahedron: 6 vertices partitioned into 3 pairs,
        all cross-pair edges present, within-pair edges absent."""
        for u in range(n):
            for v in range(u+1, n):
                # (u,v) must be a non-edge in G1
                if (adj[u] >> v) & 1:
                    continue
                S = adj[u] & adj[v]
                if S.bit_count() < 4:
                    continue
                s_list = [k for k in range(n) if (S >> k) & 1]
                m = len(s_list)
                for i in range(m):
                    x = s_list[i]
                    for j in range(i+1, m):
                        y = s_list[j]
                        # (x,y) must be a non-edge in G1
                        if (adj[x] >> y) & 1:
                            continue
                        # Need a non-edge in G1 within C = S & N1(x) & N1(y)
                        C = S & adj[x] & adj[y]
                        if C.bit_count() < 2:
                            continue
                        c_list = [k for k in range(n) if (C >> k) & 1]
                        # Check if G1[C] is NOT a clique (i.e., has a non-edge)
                        for a in range(len(c_list)):
                            z = c_list[a]
                            for b in range(a+1, len(c_list)):
                                w = c_list[b]
                                if not ((adj[z] >> w) & 1):
                                    return True
        return False
    
    def is_valid_full(n, colors):
        """Check both color classes are K_{2,2,2}-free."""
        adj1 = build_adj(n, colors)
        if find_k222(adj1, n):
            return False
        # Complement: non-edges in G1 are edges in G0
        adj0 = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if not ((adj1[i] >> j) & 1):
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
        if find_k222(adj0, n):
            return False
        return True
    
    # Generate base Paley(29)
    paley_29 = generate_paley(29)
    base_adj = build_adj(29, paley_29)
    
    # Precompute for the 29-vertex base:
    # For each pair (u,v) that is a non-edge in G1, compute S_uv = N1(u) & N1(v)
    # This helps us quickly check violations involving vertex 29.
    
    # Strategy: try random masks for vertex 29's connections,
    # then do local search (flip bits) to fix violations.
    
    # Fast check for a given mask m (29 bits, bit i = color of edge (29,i)):
    # Build full 30-vertex adj quickly.
    
    def check_mask(m, base_adj, n_base=29, n_total=30):
        """Check if adding vertex 29 with connection mask m creates K_{2,2,2}.
        Returns (valid, violation_type) where violation_type is 0 or 1 (which color)."""
        # Build adj for vertex 29
        # adj29[i] = 1 if bit i of m is set (edge (29,i) is color 1)
        
        # Check color 1:
        # New K_{2,2,2} in G1 must involve vertex 29 (base is valid).
        # Vertex 29 participates as:
        #   Case A: 29 is in a "non-edge pair" (with some u), and the other 4 vertices are in S
        #   Case B: 29 is in S (common neighbor), and the non-edge pair is (u,v) not involving 29
        #   Case C: 29 is in S, and one of the other non-edge pairs involves 29... 
        #           but 29 can't be in both a non-edge pair AND in S simultaneously
        #           (if 29 is in S=N1(u)&N1(v), then edges (29,u) and (29,v) are color 1,
        #            so 29 is NOT in a non-edge pair with u or v)
        
        # Let me enumerate more carefully.
        # K_{2,2,2} vertices: {a,b,c,d,e,f} with non-edges (a,b),(c,d),(e,f)
        # and all 12 cross edges present.
        # 
        # If vertex 29 is in this K_{2,2,2}, it's in exactly one of the three non-edge pairs.
        # WLOG, 29 is paired with vertex u (non-edge: (29,u) is color 0).
        # Then the other 4 vertices {c,d,e,f} form two non-edge pairs (c,d) and (e,f),
        # and all 12 cross edges among {29,u,c,d,e,f} except (29,u),(c,d),(e,f) are color 1.
        # 
        # Constraints:
        #   - (29,u) is color 0: bit u of m is 0
        #   - (29,c),(29,d),(29,e),(29,f) are color 1: bits c,d,e,f of m are 1
        #   - (u,c),(u,d),(u,e),(u,f) are color 1: c,d,e,f in N1(u) in base
        #   - (c,d) is color 0 in base: non-edge in base G1
        #   - (e,f) is color 0 in base: non-edge in base G1
        #   - (c,e),(c,f),(d,e),(d,f) are color 1 in base
        #   - (29,c) etc already covered
        #   - Also (u,c) etc covered
        #   - Cross edges between {c,d} and {e,f}: (c,e),(c,f),(d,e),(d,f) all color 1 in base
        #   - Cross edges between {29,u} and {c,d}: (29,c),(29,d) color 1, (u,c),(u,d) color 1
        #   - Cross edges between {29,u} and {e,f}: (29,e),(29,f) color 1, (u,e),(u,f) color 1
        #   - Cross edges between {c,d} and {e,f}: all 4 color 1
        #
        # So: c,d,e,f must be in S = N1(u) (base), with (c,d) non-edge, (e,f) non-edge,
        # and all cross edges between {c,d} and {e,f} present in base G1.
        # Also bits c,d,e,f of m must be 1, and bit u of m must be 0.
        
        # For color 1 violations:
        for u in range(n_base):
            if (m >> u) & 1:
                continue  # (29,u) must be color 0
            S = base_adj[u]  # N1(u) in base
            # Find two disjoint non-edge pairs in G1[S] with all cross edges present
            s_list = [k for k in range(n_base) if (S >> k) & 1]
            m_s = len(s_list)
            for i in range(m_s):
                c = s_list[i]
                for j in range(i+1, m_s):
                    d = s_list[j]
                    if (base_adj[c] >> d) & 1:
                        continue  # (c,d) must be non-edge
                    # Need (e,f) in S \ {c,d} with (e,f) non-edge and cross edges to {c,d}
                    for k in range(m_s):
                        e = s_list[k]
                        if e == c or e == d:
                            continue
                        for l in range(k+1, m_s):
                            f = s_list[l]
                            if f == c or f == d or f == e:
                                continue
                            if (base_adj[e] >> f) & 1:
                                continue  # (e,f) must be non-edge
                            # Check cross edges
                            if not ((base_adj[c] >> e) & 1): continue
                            if not ((base_adj[c] >> f) & 1): continue
                            if not ((base_adj[d] >> e) & 1): continue
                            if not ((base_adj[d] >> f) & 1): continue
                            # All base constraints satisfied. Check m bits.
                            if ((m >> c) & 1) and ((m >> d) & 1) and \
                               ((m >> e) & 1) and ((m >> f) & 1):
                                return (False, 1)
        
        # Check color 0 violations (complement):
        # Same logic but with roles of color 0 and 1 swapped.
        # In G0, edges are the non-edges of G1.
        # (29,u) is color 1 in G0 iff bit u of m is 1.
        # Non-edge in G0 = edge in G1.
        # S in G0 = N0(u) = complement of N1(u) in {0..28} \ {u}
        
        for u in range(n_base):
            if not ((m >> u) & 1):
                continue  # (29,u) must be color 1 in G0, i.e., bit u of m is 1
            # N0(u) in base = all vertices except u that are NOT in N1(u)
            S0 = 0
            for k in range(n_base):
                if k != u and not ((base_adj[u] >> k) & 1):
                    S0 |= (1 << k)
            if S0.bit_count() < 4:
                continue
            s_list = [k for k in range(n_base) if (S0 >> k) & 1]
            m_s = len(s_list)
            for i in range(m_s):
                c = s_list[i]
                for j in range(i+1, m_s):
                    d = s_list[j]
                    # (c,d) must be non-edge in G0 = edge in G1
                    if not ((base_adj[c] >> d) & 1):
                        continue
                    for k in range(m_s):
                        e = s_list[k]
                        if e == c or e == d:
                            continue
                        for l in range(k+1, m_s):
                            f = s_list[l]
                            if f == c or f == d or f == e:
                                continue
                            # (e,f) must be non-edge in G0 = edge in G1
                            if not ((base_adj[e] >> f) & 1):
                                continue
                            # Cross edges in G0: (c,e),(c,f),(d,e),(d,f) must be color 0 in G1
                            if (base_adj[c] >> e) & 1: continue
                            if (base_adj[c] >> f) & 1: continue
                            if (base_adj[d] >> e) & 1: continue
                            if (base_adj[d] >> f) & 1: continue
                            # Check m bits: c,d,e,f must be color 0 in G0 = color 1 in G1
                            if ((m >> c) & 1) and ((m >> d) & 1) and \
                               ((m >> e) & 1) and ((m >> f) & 1):
                                return (False, 0)
        
        return (True, 0)
    
    # Local search: start with random mask, flip bits to reduce violations
    def count_violations(m, base_adj, n_base=29):
        """Count total violations (for both colors) to guide local search."""
        count = 0
        # Color 1
        for u in range(n_base):
            if (m >> u) & 1:
                continue
            S = base_adj[u]
            s_list = [k for k in range(n_base) if (S >> k) & 1]
            m_s = len(s_list)
            for i in range(m_s):
                c = s_list[i]
                for j in range(i+1, m_s):
                    d = s_list[j]
                    if (base_adj[c] >> d) & 1:
                        continue
                    for k in range(m_s):
                        e = s_list[k]
                        if e == c or e == d:
                            continue
                        for l in range(k+1, m_s):
                            f = s_list[l]
                            if f == c or f == d or f == e:
                                continue
                            if (base_adj[e] >> f) & 1:
                                continue
                            if not ((base_adj[c] >> e) & 1): continue
                            if not ((base_adj[c] >> f) & 1): continue
                            if not ((base_adj[d] >> e) & 1): continue
                            if not ((base_adj[d] >> f) & 1): continue
                            if ((m >> c) & 1) and ((m >> d) & 1) and \
                               ((m >> e) & 1) and ((m >> f) & 1):
                                count += 1
        # Color 0
        for u in range(n_base):
            if not ((m >> u) & 1):
                continue
            S0 = 0
            for k in range(n_base):
                if k != u and not ((base_adj[u] >> k) & 1):
                    S0 |= (1 << k)
            if S0.bit_count() < 4:
                continue
            s_list = [k for k in range(n_base) if (S0 >> k) & 1]
            m_s = len(s_list)
            for i in range(m_s):
                c = s_list[i]
                for j in range(i+1, m_s):
                    d = s_list[j]
                    if not ((base_adj[c] >> d) & 1):
                        continue
                    for k in range(m_s):
                        e = s_list[k]
                        if e == c or e == d:
                            continue
                        for l in range(k+1, m_s):
                            f = s_list[l]
                            if f == c or f == d or f == e:
                                continue
                            if not ((base_adj[e] >> f) & 1):
                                continue
                            if (base_adj[c] >> e) & 1: continue
                            if (base_adj[c] >> f) & 1: continue
                            if (base_adj[d] >> e) & 1: continue
                            if (base_adj[d] >> f) & 1: continue
                            if ((m >> c) & 1) and ((m >> d) & 1) and \
                               ((m >> e) & 1) and ((m >> f) & 1):
                                count += 1
        return count
    
    # Try to find a valid mask
    found = False
    best_mask = None
    
    # Phase 1: Random search
    max_random = 5000
    for attempt in range(max_random):
        if time.time() > deadline:
            break
        m = random.getrandbits(29)
        valid, _ = check_mask(m, base_adj)
        if valid:
            found = True
            best_mask = m
            break
    
    # Phase 2: Local search if not found
    if not found and time.time() < deadline - 5:
        # Start from several random masks and do hill climbing
        for restart in range(20):
            if time.time() > deadline - 2:
                break
            m = random.getrandbits(29)
            cur_count = count_violations(m, base_adj)
            
            improved = True
            max_iters = 500
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                if cur_count == 0:
                    found = True
                    best_mask = m
                    improved = False
                    break
                
                best_neighbor = None
                best_neighbor_count = cur_count
                
                # Try flipping each bit
                for bit in range(29):
                    new_m = m ^ (1 << bit)
                    new_count = count_violations(new_m, base_adj)
                    if new_count < best_neighbor_count:
                        best_neighbor_count = new_count
                        best_neighbor = new_m
                
                if best_neighbor is not None:
                    m = best_neighbor
                    cur_count = best_neighbor_count
                else:
                    # Stuck: try random perturbation
                    m = m ^ (1 << random.randint(0, 28))
                    cur_count = count_violations(m, base_adj)
            
            if found:
                break
    
    # Build final coloring
    if found:
        m = best_mask
        colors = [0] * 435
        idx = 0
        base_idx = 0
        for i in range(30):
            for j in range(i+1, 30):
                if i == 29:
                    colors[idx] = (m >> j) & 1
                elif j == 29:
                    colors[idx] = (m >> i) & 1
                else:
                    colors[idx] = paley_29[base_idx]
                    base_idx += 1
                idx += 1
        
        # Verify the full coloring
        if is_valid_full(30, colors):
            return {'n': 30, 'colors': colors}
        else:
            # Shouldn't happen if check_mask is correct, but be safe
            pass
    
    # Fallback: return Paley(29)
    return {'n': 29, 'colors': paley_29}
