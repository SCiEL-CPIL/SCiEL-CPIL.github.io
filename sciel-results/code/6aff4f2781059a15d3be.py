import time
import random
import sys

def construct():
    """
    Construct a pentagon-free (C5-free) graph with no independent 9-set.
    Strategy:
    1. Base: 8 disjoint K4s (n=32, alpha=8, C5-free). Verified valid.
    2. Try to extend to n=33 by adding vertex 32 with strategic connections.
    3. Use fast C5 and IS checks with bitmasks.
    4. Fall back to n=32 if extension fails.
    """
    
    def has_c5_fast(n, adj_mask):
        """Fast C5 detection: check if there exists u-v-w-x-y-u cycle."""
        # For each pair (u,v) of adjacent vertices, check if there's a path of length 3 from u to v
        # through distinct vertices, which would form a C5 with edge u-v.
        # Actually, C5 is u-v-w-x-y-u where all edges exist and all vertices distinct.
        # Equivalent: for each edge (u,v), is there a path u-a-b-c-v with a,b,c distinct from u,v and from each other?
        # That's a path of length 4 from u to v avoiding... no, that would be C5 if we add edge v-u.
        # Wait: C5 = 5 vertices in a cycle. So we need u-v-w-x-y-u.
        # For each ordered pair (u,v) with u<v and edge exists,
        # check if there exist w,x,y distinct from u,v, all distinct,
        # such that edges: v-w, w-x, x-y, y-u all exist.
        # That's a path of length 3 from v to u avoiding u and v internally.
        
        for u in range(n):
            for v in range(u+1, n):
                if not (adj_mask[u] & (1 << v)):
                    continue
                # Find all w in N(v) \ {u, v}
                w_mask = adj_mask[v] & ~(1 << u) & ~(1 << v)
                if not w_mask:
                    continue
                for w in range(n):
                    if not (w_mask & (1 << w)):
                        continue
                    # Find all x in N(w) \ {u, v, w}
                    x_mask = adj_mask[w] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                    if not x_mask:
                        continue
                    for x in range(n):
                        if not (x_mask & (1 << x)):
                            continue
                        # Check if x is connected to some y in N(x) ∩ N(u) \ {u,v,w,x}
                        y_mask = adj_mask[x] & adj_mask[u] & ~(1 << v) & ~(1 << w) & ~(1 << x)
                        if y_mask:
                            return True
        return False
    
    def has_is_k_fast(n, adj_mask, k, deadline=None):
        """Fast check for independent set of size k using bitmask DFS with pruning."""
        
        def greedy_is_size(cand):
            """Greedy upper bound on independent set size in cand."""
            count = 0
            c = cand
            while c:
                # Pick vertex with min degree in remaining cand
                v = -1
                min_deg = n + 1
                cc = c
                while cc:
                    lsb = cc & -cc
                    vi = lsb.bit_length() - 1
                    deg = (adj_mask[vi] & c).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        v = vi
                    cc -= lsb
                if v == -1:
                    break
                count += 1
                # Remove v and its neighbors
                c &= ~adj_mask[v] & ~(1 << v)
            return count
        
        def dfs(cand, rem):
            if rem == 0:
                return True
            if cand == 0:
                return False
            if cand.bit_count() < rem:
                return False
            # Upper bound check
            if greedy_is_size(cand) < rem:
                return False
            
            # Pick vertex with max degree in cand
            v = -1
            max_deg = -1
            c = cand
            while c:
                lsb = c & -c
                vi = lsb.bit_length() - 1
                deg = (adj_mask[vi] & cand).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = vi
                c -= lsb
            
            if v == -1:
                return False
                
            lsb_v = 1 << v
            
            # Branch 1: Include v
            new_cand_incl = cand & ~adj_mask[v] & ~lsb_v
            if new_cand_incl.bit_count() >= rem - 1:
                if dfs(new_cand_incl, rem - 1):
                    return True
            
            # Branch 2: Exclude v
            new_cand_excl = cand & ~lsb_v
            if new_cand_excl.bit_count() >= rem:
                if dfs(new_cand_excl, rem):
                    return True
            
            return False
        
        all_mask = (1 << n) - 1
        return dfs(all_mask, k)
    
    def is_valid(n, adj_mask):
        if has_c5_fast(n, adj_mask):
            return False
        if has_is_k_fast(n, adj_mask, 9):
            return False
        return True
    
    def adj_to_edges(n, adj_mask):
        edges = []
        for u in range(n):
            mask = adj_mask[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                mask -= lsb
        return edges
    
    # Base 32-vertex graph: 8 disjoint K4s
    def get_base_32():
        n = 32
        adj_mask = [0] * n
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    adj_mask[base + i] |= (1 << (base + j))
                    adj_mask[base + j] |= (1 << (base + i))
        return n, adj_mask
    
    # Try to construct n=33 graph
    def try_n33(time_budget):
        start_time = time.time()
        deadline = start_time + time_budget
        
        n_base, adj_base = get_base_32()
        
        # Try many random configurations for vertex 32
        # Vertex 32 connects to some subset of 0..31
        # To avoid C5: 32's neighbors must form an independent set in the base graph
        # (otherwise two neighbors of 32 that are adjacent create a triangle, 
        #  but that's fine - C5 needs a 5-cycle. Actually, if 32 connects to 
        #  adjacent vertices u,v in a K4, then 32-u-v-32 is a triangle, not C5.
        #  C5 would need: 32-a-b-c-d-32 where all are distinct and edges exist.
        #  Since base has no edges between K4s, any path from 32 goes into one K4,
        #  stays in it (K4 is complete, so max path length in K4 is 3: a-b-c),
        #  then back to 32. So 32-a-b-c-32 is only 4 vertices + 32 = 5? 
        #  No: 32-a, a-b, b-c, c-32. That's a 4-cycle if c is also neighbor of 32.
        #  For C5: 32-a-b-c-d-32. Need d neighbor of 32 and c. 
        #  In a K4: a,b,c,d all mutually adjacent. So 32-a-b-c-d-32 IS a C5
        #  if 32 connects to all 4 vertices of a K4!
        #  So: 32 must NOT connect to all 4 vertices of any K4.
        #  If 32 connects to 3 or fewer vertices in each K4, can we still get C5?
        #  32-a-b-c-d-32: need 4 distinct neighbors of 32 in the same K4.
        #  So if 32 connects to at most 3 per K4, no C5 through that K4.
        #  Can C5 go through multiple K4s? No edges between K4s, so no.
        #  Therefore: 32 must connect to at most 3 vertices in each K4 block.
        
        # For alpha: base has alpha=8 (one per K4). 
        # Adding 32: if 32 is isolated, alpha=9. Bad.
        # 32 must connect to at least one vertex in each "group" to reduce alpha.
        # Actually: alpha of new graph. If 32 connects to vertex in block i, 
        # then in an IS, we can't have both 32 and that vertex.
        # Max IS: either include 32 (then can't include its neighbors, 
        #  can include 1 from each other block, and from blocks where 32 has neighbors,
        #  can include 1 from non-neighbor vertices in that block) 
        #  or exclude 32 (then alpha = 8 from base, but wait, base alpha is 8,
        #  and we might be able to do better... no, base IS is 8, one per K4).
        # 
        # Let's think: if 32 connects to vertices in blocks S (subset of {0..7}),
        # then:
        # - If we include 32: for blocks in S, we can pick 1 from non-neighbors of 32 
        #   in that block (up to 4 - deg_32_in_block vertices available, but still at most 1 
        #   from the K4 minus neighbors... actually in K4, if 32 connects to d vertices,
        #   the remaining 4-d vertices form an independent set in the K4? No! In K4 all 
        #   pairs are adjacent. So the remaining 4-d vertices are still mutually adjacent.
        #   We can pick at most 1 from them.
        #   For blocks not in S: pick 1 from the 4 vertices.
        #   Total: 1 (for 32) + |S| (one from each block in S, from non-neighbors) 
        #           + (8-|S|) (one from each block not in S) = 1 + 8 = 9. 
        #   Wait, that's always 9? No: for blocks in S, if 32 connects to all 4, 
        #   we can pick 0 from that block. But we said at most 3 per block.
        #   If 32 connects to d vertices in block i (d <= 3), we can still pick 1 
        #   from the remaining 4-d >= 1 vertices. So we get 1 from that block.
        #   So including 32 gives: 1 + sum over blocks of 1 = 1 + 8 = 9. Always 9!
        #   That means if we include 32, alpha >= 9 always (as long as each block 
        #   has at least 1 non-neighbor of 32, which is true since d<=3).
        #
        # - If we exclude 32: alpha = alpha(base) = 8.
        #
        # So alpha(new graph) = max(9, 8) = 9 if 32 has any configuration with d<=3 per block.
        # This means n=33 with this structure ALWAYS has alpha >= 9!
        # 
        # Unless... we also add edges between blocks? But that might create C5.
        # Or modify the base structure.
        
        # Alternative: modify the base. Instead of 8 K4s, use a different structure.
        # What if we use 7 K4s + 1 K5? No, K5 has C5.
        # What about Turan graph T(32,4)? T(n,r) is complete r-partite with balanced parts.
        # T(32,4): 4 parts of 8, all edges between parts. Alpha = 8 (one part).
        # C5-free? T(n,r) is C5-free iff... actually complete multipartite graphs 
        # contain C5 if there are at least 5 parts or if a part has size >= 2 and 
        # there are at least 3 parts. Wait: in T(n,4), take one vertex from each of 
        # 4 parts: that's K4, no C5. To get C5, need 5 vertices. But only 4 parts.
        # A C5 in complete multipartite: vertices must be in distinct parts (since 
        # same part = no edge). So need 5 distinct parts for C5. With 4 parts, no C5!
        # So T(32,4) is C5-free and alpha=8. Same as 8 K4s essentially.
        
        # For n=33: T(33,4) has parts of sizes 9,8,8,8. Alpha=9. Bad.
        # T(33,5): parts ~7 each. Alpha=7. C5-free? Need 5 parts for C5, have 5.
        #   Can we form C5? Pick one from each of 5 parts: that's K5, which contains C5!
        #   So T(33,5) has C5. Bad.
        
        # What about a different approach for n=33?
        # Start with T(32,4) and add vertex 33 with careful connections.
        # Or: use the fact that we need alpha < 9, so alpha <= 8.
        # By Turan's theorem, if alpha <= 8, then the complement has no K9.
        # The complement of our graph must be K9-free.
        
        # Let me try a completely different strategy:
        # Use simulated annealing on n=33 with a smarter initialization.
        # Start from T(32,4) + vertex 32, and optimize.
        
        # Actually, let me just try random n=33 graphs with SA, but with 
        # much faster cost evaluation.
        
        def cost_fast(n, adj_mask):
            """Return 0 if valid, else penalty."""
            if has_c5_fast(n, adj_mask):
                return 100
            if has_is_k_fast(n, adj_mask, 9):
                return 10
            return 0
        
        # Initialize from T(32,4) + random vertex 32
        n = 33
        adj_mask = [0] * n
        
        # T(32,4): 4 parts of 8 vertices each
        part_sizes = [8, 8, 8, 8]
        vertex = 0
        parts = []
        for size in part_sizes:
            part = list(range(vertex, vertex + size))
            parts.append(part)
            vertex += size
        
        # Add all edges between different parts
        for i in range(4):
            for j in range(i + 1, 4):
                for u in parts[i]:
                    for v in parts[j]:
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
        
        # Vertex 32: connect to random subset, at most 3 per part to avoid C5
        for part in parts:
            # Choose 0-3 random vertices from this part to connect to
            k = random.randint(0, 3)
            chosen = random.sample(part, k)
            for v in chosen:
                adj_mask[32] |= (1 << v)
                adj_mask[v] |= (1 << 32)
        
        # Simulated Annealing with smarter moves
        T = 1.0
        cooling = 0.9995
        best_adj = adj_mask[:]
        best_cost = cost_fast(n, best_adj)
        current_cost = best_cost
        
        while time.time() < deadline:
            # Smarter move: try to fix violations
            if current_cost > 0:
                # If C5 exists, try removing an edge from a detected C5
                # If IS exists, try adding an edge within the IS
                if current_cost >= 100:
                    # C5 violation: find a C5 and remove a random edge from it
                    # For simplicity, just flip a random edge but bias toward removal
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u == v:
                        continue
                    mask_u = 1 << u
                    mask_v = 1 << v
                    if adj_mask[u] & mask_v:
                        # Remove edge
                        adj_mask[u] &= ~mask_v
                        adj_mask[v] &= ~mask_u
                    else:
                        # Add edge (less likely when C5 exists)
                        if random.random() < 0.3:
                            adj_mask[u] |= mask_v
                            adj_mask[v] |= mask_u
                        else:
                            # Try another edge to remove
                            u2 = random.randint(0, n - 1)
                            v2 = random.randint(0, n - 1)
                            if u2 == v2:
                                continue
                            mu2 = 1 << u2
                            mv2 = 1 << v2
                            if adj_mask[u2] & mv2:
                                adj_mask[u2] &= ~mv2
                                adj_mask[v2] &= ~mu2
                else:
                    # IS violation: try adding an edge
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u == v:
                        continue
                    mask_u = 1 << u
                    mask_v = 1 << v
                    if not (adj_mask[u] & mask_v):
                        adj_mask[u] |= mask_v
                        adj_mask[v] |= mask_u
                    else:
                        # Remove edge
                        adj_mask[u] &= ~mask_v
                        adj_mask[v] &= ~mask_u
            else:
                # Valid: explore neighborhood
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                mask_u = 1 << u
                mask_v = 1 << v
                if adj_mask[u] & mask_v:
                    adj_mask[u] &= ~mask_v
                    adj_mask[v] &= ~mask_u
                else:
                    adj_mask[u] |= mask_v
                    adj_mask[v] |= mask_u
            
            new_cost = cost_fast(n, adj_mask)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = adj_mask[:]
            else:
                delta = new_cost - current_cost
                if delta < 0 or random.random() < pow(2.71828, -delta / max(T, 0.01)):
                    current_cost = new_cost
                else:
                    # Revert
                    if adj_mask[u] & mask_v:
                        adj_mask[u] &= ~mask_v
                        adj_mask[v] &= ~mask_u
                    else:
                        adj_mask[u] |= mask_v
                        adj_mask[v] |= mask_u
            
            T *= cooling
            
            if best_cost == 0:
                return best_adj
        
        if best_cost == 0:
            return best_adj
        return None
    
    # Main logic with time management
    total_time = 115.0
    start_total = time.time()
    
    # Try n=33 with multiple random seeds
    time_per_attempt = 15.0
    max_attempts = int((total_time - 5) / time_per_attempt)
    
    for attempt in range(max_attempts):
        if time.time() - start_total > total_time - 5:
            break
        adj_mask = try_n33(time_per_attempt)
        if adj_mask is not None:
            if is_valid(33, adj_mask):
                edges = adj_to_edges(33, adj_mask)
                return {'n': 33, 'edges': edges}
    
    # Fallback to verified n=32
    n, adj_mask = get_base_32()
    edges = adj_to_edges(n, adj_mask)
    return {'n': n, 'edges': edges}
