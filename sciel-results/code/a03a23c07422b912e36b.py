import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Generate Paley coloring for n=29
    def generate_paley(n, prime):
        res = set()
        for x in range(1, prime):
            res.add((x * x) % prime)
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % prime
                if diff in res:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    # Build adjacency bitmasks from colors
    def build_adj(colors, n):
        adj = [[0] * n for _ in range(2)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return adj

    # Count K2,2,2 violations using adjacency bitmasks
    def count_violations(adj, n):
        total_viol = 0
        for c in [0, 1]:
            A = adj[c]
            for u1 in range(n):
                for u2 in range(u1 + 1, n):
                    S = A[u1] & A[u2]
                    if S == 0:
                        continue
                    # Extract set bits
                    s_bits = []
                    temp = S
                    while temp:
                        lsb = temp & -temp
                        s_bits.append(lsb.bit_length() - 1)
                        temp ^= lsb
                    k = len(s_bits)
                    if k < 4:
                        continue
                    for a in range(k):
                        v1 = s_bits[a]
                        for b in range(a + 1, k):
                            v2 = s_bits[b]
                            C = A[v1] & A[v2]
                            common_in_S = C & S
                            if common_in_S & (common_in_S - 1):
                                total_viol += 1
        return total_viol

    # Incremental violation delta when flipping edge (i,j) from color old_c to new_c
    # Only counts violations that involve vertices i or j in the K2,2,2 structure
    def violation_delta(adj, n, i, j, old_c):
        """
        Compute change in violation count when flipping edge (i,j) from old_c to (1-old_c).
        A K2,2,2 in color c requires: two "left" vertices sharing >=2 common neighbors in c,
        and those common neighbors also sharing >=2 common neighbors in c (within the left set).
        
        We only need to check violations where i or j participates as one of the 6 vertices.
        """
        new_c = 1 - old_c
        delta = 0
        
        for c in [old_c, new_c]:
            A = adj[c]
            # After flip: if c == old_c, edge (i,j) is removed; if c == new_c, edge (i,j) is added
            # We compute violations BEFORE and AFTER for structures involving i or j
            
            # For color c, check all pairs (u1,u2) where at least one is i or j
            # S = A[u1] & A[u2] must have >=4 bits, and among those bits, some pair (v1,v2) 
            # must have A[v1]&A[v2]&S having >=2 bits
            
            # Before flip (current adj state has old_c for edge (i,j))
            # After flip: adj[old_c] loses (i,j), adj[new_c] gains (i,j)
            
            # We'll compute the delta by checking relevant structures
            # A violation is a 6-tuple {u1,u2,v1,v2,w1,w2} where:
            #   A[u1]&A[u2] contains {v1,v2,w1,w2} (at least these 4)
            #   A[v1]&A[v2]&S contains at least 2 of {w1,w2} (actually >=2 from S)
            # Wait, let me re-read the check:
            #   S = A[u1] & A[u2]
            #   For pair (v1,v2) in S: C = A[v1] & A[v2], common_in_S = C & S
            #   If common_in_S has >=2 bits, that's a violation.
            # So the 6 vertices are u1,u2,v1,v2 and two more from common_in_S.
            
            # The edge (i,j) being in color c means i and j are adjacent in A.
            # This affects:
            #   - A[i] and A[j] (they gain/lose each other)
            #   - Any S that includes both i and j (i.e., A[u1]&A[u2] contains both)
            #   - Any C that includes both i and j
            #   - Any common_in_S that includes both i and j
            
            # This is complex. Let me just do a targeted recount.
            
            # Approach: for each pair (u1,u2) where min(u1,u2) <= max(i,j) and the pair involves i or j,
            # recompute whether it contributes violations, before and after.
            
            pass  # Too complex for inline; use full recount but optimized
        
        return delta  # placeholder

    # Fast validity check
    def is_valid_fast(colors, n):
        adj = build_adj(colors, n)
        return count_violations(adj, n) == 0

    # Count violations but only for edges involving a specific vertex set (for incremental updates)
    def count_violations_restricted(adj, n, vertex_set):
        """Count violations where at least one of the 6 K2,2,2 vertices is in vertex_set."""
        total_viol = 0
        vs = set(vertex_set)
        for c in [0, 1]:
            A = adj[c]
            for u1 in range(n):
                for u2 in range(u1 + 1, n):
                    S = A[u1] & A[u2]
                    if S == 0:
                        continue
                    s_bits = []
                    temp = S
                    while temp:
                        lsb = temp & -temp
                        s_bits.append(lsb.bit_length() - 1)
                        temp ^= lsb
                    k = len(s_bits)
                    if k < 4:
                        continue
                    # Check if any of u1, u2, or s_bits are in vertex_set
                    relevant = (u1 in vs) or (u2 in vs) or any(v in vs for v in s_bits)
                    if not relevant:
                        continue
                    for a in range(k):
                        v1 = s_bits[a]
                        for b in range(a + 1, k):
                            v2 = s_bits[b]
                            C = A[v1] & A[v2]
                            common_in_S = C & S
                            if common_in_S & (common_in_S - 1):
                                total_viol += 1
        return total_viol

    # Local search for n=30 with smart initialization and faster search
    def local_search_30(colors, n):
        # Build initial adjacency
        adj = build_adj(colors, n)
        current_viol = count_violations(adj, n)
        best_colors = colors[:]
        best_viol = current_viol
        
        # Precompute edge index to (i,j) mapping
        num_edges = n * (n - 1) // 2
        edge_map = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map.append((i, j))
                idx += 1
        
        # Edge index lookup: edge_idx_from[i][j] for i<j
        edge_idx_from = [[-1]*n for _ in range(n)]
        for idx2, (i, j) in enumerate(edge_map):
            edge_idx_from[i][j] = idx2
            edge_idx_from[j][i] = idx2
        
        temp = 3.0
        cooling = 0.9998
        iter_count = 0
        stall_count = 0
        stall_threshold = 3000
        no_improve_count = 0
        no_improve_threshold = 20000
        
        while time.time() < deadline - 5:
            iter_count += 1
            
            # Pick a random edge
            edge_idx = random.randint(0, num_edges - 1)
            i, j = edge_map[edge_idx]
            old_val = colors[edge_idx]
            new_val = 1 - old_val
            
            # Update adjacency incrementally
            adj[old_val][i] &= ~(1 << j)
            adj[old_val][j] &= ~(1 << i)
            adj[new_val][i] |= (1 << j)
            adj[new_val][j] |= (1 << i)
            
            new_viol = count_violations(adj, n)
            colors[edge_idx] = new_val
            
            delta = new_viol - current_viol
            if delta < 0 or (delta == 0 and random.random() < 0.1) or random.random() < pow(2.71828, -delta / temp):
                current_viol = new_viol
                if new_viol < best_viol:
                    best_viol = new_viol
                    best_colors = colors[:]
                    stall_count = 0
                    no_improve_count = 0
                    if best_viol == 0:
                        return best_colors
                else:
                    no_improve_count += 1
            else:
                # Revert
                adj[new_val][i] &= ~(1 << j)
                adj[new_val][j] &= ~(1 << i)
                adj[old_val][i] |= (1 << j)
                adj[old_val][j] |= (1 << i)
                colors[edge_idx] = old_val
                stall_count += 1
                no_improve_count += 1
            
            temp *= cooling
            
            # Multi-flip restart when stuck
            if stall_count > stall_threshold:
                # Try flipping 5 random edges together
                num_flips = 5
                flipped_edges = []
                for _ in range(num_flips):
                    e1 = random.randint(0, num_edges - 1)
                    i1, j1 = edge_map[e1]
                    old1 = colors[e1]
                    new1 = 1 - old1
                    adj[old1][i1] &= ~(1 << j1)
                    adj[old1][j1] &= ~(1 << i1)
                    adj[new1][i1] |= (1 << j1)
                    adj[new1][j1] |= (1 << i1)
                    colors[e1] = new1
                    flipped_edges.append((e1, i1, j1, old1, new1))
                
                nv = count_violations(adj, n)
                if nv < current_viol:
                    current_viol = nv
                    if nv < best_viol:
                        best_viol = nv
                        best_colors = colors[:]
                        no_improve_count = 0
                else:
                    for (e1, i1, j1, old1, new1) in flipped_edges:
                        adj[new1][i1] &= ~(1 << j1)
                        adj[new1][j1] &= ~(1 << i1)
                        adj[old1][i1] |= (1 << j1)
                        adj[old1][j1] |= (1 << i1)
                        colors[e1] = old1
                stall_count = 0
                temp = 3.0  # Reheat
            
            # If no improvement for a long time, do a larger perturbation
            if no_improve_count > no_improve_threshold:
                # Flip 10 random edges
                num_flips = 10
                flipped_edges = []
                for _ in range(num_flips):
                    e1 = random.randint(0, num_edges - 1)
                    i1, j1 = edge_map[e1]
                    old1 = colors[e1]
                    new1 = 1 - old1
                    adj[old1][i1] &= ~(1 << j1)
                    adj[old1][j1] &= ~(1 << i1)
                    adj[new1][i1] |= (1 << j1)
                    adj[new1][j1] |= (1 << i1)
                    colors[e1] = new1
                    flipped_edges.append((e1, i1, j1, old1, new1))
                
                nv = count_violations(adj, n)
                if nv < current_viol:
                    current_viol = nv
                    if nv < best_viol:
                        best_viol = nv
                        best_colors = colors[:]
                        no_improve_count = 0
                else:
                    for (e1, i1, j1, old1, new1) in flipped_edges:
                        adj[new1][i1] &= ~(1 << j1)
                        adj[new1][j1] &= ~(1 << i1)
                        adj[old1][i1] |= (1 << j1)
                        adj[old1][j1] |= (1 << i1)
                        colors[e1] = old1
                no_improve_count = 0
                temp = 4.0
        
        return best_colors

    # Smart vertex extension: try multiple deterministic patterns for vertex 29
    def try_extend_to_30(colors29, n29):
        n30 = 30
        best_result = None
        best_viol = float('inf')
        
        # Pattern 1: Connect vertex 29 to a subset based on Paley structure
        # Try connecting to vertices that form a "balanced" set
        for attempt in range(20):
            colors30 = colors29[:]
            if attempt == 0:
                # Connect to even-indexed vertices
                for i in range(n29):
                    colors30.append(0 if i % 2 == 0 else 1)
            elif attempt == 1:
                # Connect based on Paley quadratic residues
                res = set()
                for x in range(1, 29):
                    res.add((x * x) % 29)
                for i in range(n29):
                    colors30.append(0 if i in res else 1)
            elif attempt == 2:
                # Connect to first 15 vertices
                for i in range(n29):
                    colors30.append(0 if i < 15 else 1)
            elif attempt == 3:
                # Connect to vertices congruent to 0,1 mod 3
                for i in range(n29):
                    colors30.append(0 if i % 3 < 2 else 1)
            else:
                # Random
                for i in range(n29):
                    colors30.append(random.randint(0, 1))
            
            colors30_opt = local_search_30(colors30, n30)
            adj = build_adj(colors30_opt, n30)
            viol = count_violations(adj, n30)
            
            if viol < best_viol:
                best_viol = viol
                best_result = colors30_opt
            
            if best_viol == 0:
                return best_result
        
        return best_result

    # 1. Generate and verify Paley(29)
    n29 = 29
    colors29 = generate_paley(n29, 29)
    
    if is_valid_fast(colors29, n29):
        # We have a valid n=29 solution.
        # Try to extend to n=30
        colors30_optimized = try_extend_to_30(colors29, n29)
        
        if colors30_optimized is not None:
            adj = build_adj(colors30_optimized, 30)
            if count_violations(adj, 30) == 0:
                return {'n': 30, 'colors': colors30_optimized}
        
        return {'n': 29, 'colors': colors29}
    else:
        return {'n': 17, 'colors': generate_paley(17, 17)}
