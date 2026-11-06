import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Fallback: 30 vertices = 10 disjoint triangles (C4-free, alpha=10)
    def get_fallback():
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges.append([a, b])
            edges.append([a, c])
            edges.append([b, c])
        return {'n': 30, 'edges': edges}

    # Build polarity graph of PG(2, q): points are (x,y,1) plus (x,y,0) with x^2 + y^2 + 1 = 0
    # Two points are adjacent if x1*x2 + y1*y2 + z1*z2 = 0
    def build_polarity_graph(q):
        # Points of PG(2,q): (x, y, z) with not all zero, mod q, up to scaling
        # Use normalized: z=1 (affine) plus z=0 (line at infinity)
        points = []
        for x in range(q):
            for y in range(q):
                points.append((x, y, 1))
        # Line at infinity: (x, y, 0) with x^2 + y^2 + 0 = 0? 
        # Actually for polarity graph, we include all points of PG(2,q)
        # Line at infinity points: (1, 0, 0), (0, 1, 0), (1, k, 0) for k in F_q
        # But we need to be careful with representation
        # Let's use: all (x,y,z) with z=1, plus (1,0,0), (0,1,0), and (1,k,0) for k in F_q
        # Actually standard: PG(2,q) has q^2 + q + 1 points
        # Affine: (x,y,1) for x,y in F_q -> q^2 points
        # Infinity: (1,0,0), (0,1,0), (1,k,0) for k in F_q -> q+1 points
        # Total: q^2 + q + 1
        
        # But for the polarity graph, we use the quadratic form x^2 + y^2 + z^2 = 0
        # Two points are adjacent if they are orthogonal under this form
        
        all_points = []
        for x in range(q):
            for y in range(q):
                all_points.append((x, y, 1))
        all_points.append((1, 0, 0))
        all_points.append((0, 1, 0))
        for k in range(q):
            all_points.append((1, k, 0))
        
        n = len(all_points)
        adj = [0] * n
        for i in range(n):
            xi, yi, zi = all_points[i]
            for j in range(i+1, n):
                xj, yj, zj = all_points[j]
                # Check orthogonality: xi*xj + yi*yj + zi*zj ≡ 0 mod q
                val = (xi * xj + yi * yj + zi * zj) % q
                if val == 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Convert to edge list
        edges = []
        for i in range(n):
            mask = adj[i] & ((1 << (i+1)) - 1)  # only j < i
            while mask:
                lsb = mask & (-mask)
                j = lsb.bit_length() - 1
                edges.append([j, i])
                mask ^= lsb
        return n, edges

    # Check C4-free using adjacency masks
    def check_c4_free(edges, n):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        for u in range(n):
            for v in range(u+1, n):
                common = adj[u] & adj[v]
                if common & (common - 1):
                    return False, adj
        return True, adj

    # Check if independent set of size k exists (exact via DFS with pruning)
    def has_independent_set(adj, n, k, time_limit):
        start = time.time()
        target = k
        
        # Greedy first: try to find independent set of size k
        candidates = (1 << n) - 1
        found_size = 0
        while candidates:
            if time.time() - start > time_limit:
                return None
            min_deg = n
            chosen = -1
            c = candidates
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj[v] & candidates).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    chosen = v
            if chosen == -1:
                break
            found_size += 1
            if found_size >= target:
                return True
            candidates &= ~(adj[chosen] | (1 << chosen))
        
        if found_size >= target:
            return True
        
        # Exact DFS
        def dfs(cand, size):
            if time.time() - start > time_limit:
                return None
            if size >= target:
                return True
            if cand == 0:
                return False
            if size + cand.bit_count() < target:
                return False
            
            # Pick vertex with max degree in cand
            c = cand
            max_deg = -1
            chosen = -1
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj[v] & cand).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    chosen = v
            if chosen == -1:
                return False
            
            # Include
            nc = cand & ~(adj[chosen] | (1 << chosen))
            res = dfs(nc, size + 1)
            if res is True:
                return True
            if res is None:
                return None
            
            # Exclude
            nc = cand & ~(1 << chosen)
            res = dfs(nc, size)
            if res is True:
                return True
            if res is None:
                return None
            
            return False
        
        return dfs((1 << n) - 1, 0)

    # Try polarity graphs for various q values
    # PG(2,q) has q^2+q+1 vertices
    # q=6: 43 vertices - but PG(2,6) doesn't exist since 6 is not prime power
    # q=7: 57 vertices - too many, alpha might be >= 11
    # q=5: 31 vertices
    # q=4: 21 vertices
    
    # Actually, let's think about this differently.
    # The known result is R(C4, K11) > 39, meaning there exists a C4-free graph on 39 vertices with alpha <= 10.
    # We need to find such a graph.
    
    # Approach: Use a polarity graph of a projective plane.
    # For PG(2, q) where q is a prime power, the polarity graph is C4-free.
    # q=7: 57 vertices, but we need to check alpha.
    # Let's try q=7 and see if we can find a subgraph or if the full graph works.
    
    # Actually, let me try a different approach: random greedy C4-free graph construction
    # with careful edge selection to keep alpha low.
    
    def generate_greedy_c4_free(n, seed, time_limit=5.0):
        random.seed(seed)
        adj = [0] * n
        edges = []
        
        # Shuffle all possible edges
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() - start_time > deadline - 10:
                break
            nu = adj[u]
            nv = adj[v]
            
            # Check if adding (u,v) creates a C4
            # C4 exists if there's a vertex w adjacent to both u and v where... 
            # Actually C4-free means no two vertices share 2+ common neighbors
            # So when adding (u,v), we need: |N(u) ∩ N(v)| = 0 
            # Wait, that's not right. After adding (u,v), N(u) contains v and N(v) contains u.
            # The condition is: for any pair of vertices, they share at most 1 common neighbor.
            # When we add edge (u,v), we create new common neighbors:
            # Any w that is neighbor of u becomes a common neighbor of (w,v) if w is also neighbor of v... no.
            
            # Let me think again. C4: a-b-c-d-a. This means b and d share neighbors a and c.
            # So the condition is: for any two vertices x, y, |N(x) ∩ N(y)| <= 1.
            
            # When adding edge (u,v):
            # New pairs that could violate: 
            # For any w in N(u), w and v now share u as common neighbor. If w is also in N(v), then w and v share both u and... wait no.
            # If w is in N(u) and we add (u,v), then v gains u as neighbor. 
            # For pair (w, v) where w in N(u): they share u. They also share any other common neighbor.
            # For pair (u, x) where x in N(v): they share v.
            
            # The key check: after adding (u,v), for every pair (a,b), |N(a) ∩ N(b)| <= 1.
            # The only new edges involving u or v are (u,v) itself.
            # So we need to check:
            # 1. For any w in N(u) ∩ N(v): w shares u and v with... no.
            
            # Let me use the standard check: adding (u,v) is safe iff 
            # N(u) and N(v) have no common neighbors... no that's not right either.
            
            # Correct condition: Adding edge (u,v) creates a C4 iff there exists w such that 
            # w is adjacent to both u and v... no, that would just mean u,v,w form a triangle-ish thing.
            
            # C4 is a cycle of length 4: u-v-w-x-u. So u and w share neighbors v and x.
            # When adding (u,v): if there exists x such that x is neighbor of u and neighbor of v, 
            # then u and v share neighbor x. That's fine, sharing 1 neighbor is OK.
            # But if there exists x,y both neighbors of u and both neighbors of v... 
            # No wait. Let me be precise.
            
            # After adding (u,v), the new common neighbors:
            # - v's neighbors gain u. So for any w in N(v) (before adding), w and u now share v.
            #   If w and u already shared another neighbor, that's a C4.
            # - Similarly, for any w in N(u) (before adding), w and v now share u.
            #   If w and v already shared another neighbor, that's a C4.
            
            # So the check is:
            # For all w in N(u): |N(w) ∩ N(v) \ {u}| + (1 if w==v else 0) ... this is getting complex.
            
            # Simpler: just check that N(u) ∩ N(v) has size 0 before adding? No.
            # Actually: adding (u,v) is C4-safe iff N(u) ∩ N(v) = ∅.
            # Why? If there's w in N(u) ∩ N(v), then u-w-v forms a path of length 2.
            # That's not a C4 by itself. A C4 needs two different paths of length 2 between the same pair.
            
            # OK let me just do the naive check: after tentatively adding, verify no C4.
            # For efficiency, check:
            # For each w in N(u): if w != v, check that w and v share at most 1 common neighbor (which would be u plus any others)
            # For each w in N(v): if w != u, check that w and u share at most 1 common neighbor
            
            # Simpler approach: N(u) and N(v) must be disjoint for C4-freeness when adding (u,v)? 
            # No, that's too strict. Consider a star: center connected to all leaves. No C4.
            
            # Let me just check directly: the condition for no C4 after adding (u,v) is:
            # (N(u) ∩ N(v)) has at most... hmm.
            
            # You know what, let me just do the O(n^2) check per edge addition.
            # For n up to 43, and ~900 edges, that's manageable.
            
            tent_adj_u = adj[u] | (1 << v)
            tent_adj_v = adj[v] | (1 << u)
            
            safe = True
            # Check all pairs involving u or v
            for w in range(n):
                if w == u or w == v:
                    continue
                # Check pair (u, w)
                if adj[w] & (1 << u):  # w is neighbor of u
                    common = (tent_adj_u & adj[w])
                    if common & (common - 1):
                        safe = False
                        break
                # Check pair (v, w)
                if safe and adj[w] & (1 << v):
                    common = (tent_adj_v & adj[w])
                    if common & (common - 1):
                        safe = False
                        break
            
            if safe:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
        
        return adj, edges

    # Try to find valid graphs from n=42 down to 31
    best_result = None
    
    for n in range(42, 30, -1):
        if time.time() > deadline - 15:
            break
        
        # Try polarity graph if n matches q^2+q+1 for some q
        # q=7: 57 (too big), q=5: 31, q=4: 21
        if n == 31:
            pg_n, pg_edges = build_polarity_graph(5)
            if pg_n == n:
                is_c4, adj = check_c4_free(pg_edges, n)
                if is_c4:
                    res = has_independent_set(adj, n, 11, 5.0)
                    if res is False:
                        return {'n': n, 'edges': pg_edges}
        
        # Try greedy construction
        max_seeds = 30 if n <= 38 else 15
        for seed in range(max_seeds):
            if time.time() > deadline - 10:
                break
            
            adj, edges = generate_greedy_c4_free(n, seed)
            
            is_c4, _ = check_c4_free(edges, n)
            if not is_c4:
                continue
            
            res = has_independent_set(adj, n, 11, 3.0)
            if res is False:
                best_result = {'n': n, 'edges': edges}
                return best_result
            # res is True: invalid, res is None: timeout
    
    # Try n=31 with polarity graph more carefully
    if time.time() < deadline - 10:
        pg_n, pg_edges = build_polarity_graph(5)
        if pg_n == 31:
            is_c4, adj = check_c4_free(pg_edges, 31)
            if is_c4:
                res = has_independent_set(adj, 31, 11, 10.0)
                if res is False:
                    return {'n': 31, 'edges': pg_edges}
    
    if best_result:
        return best_result
    
    return get_fallback()
