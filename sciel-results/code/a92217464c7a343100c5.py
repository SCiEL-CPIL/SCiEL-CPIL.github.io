import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    n = 18
    
    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def has_k4(adj):
        """Check if graph contains a K4 using bitmasks."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                # Only consider vertices > b to avoid duplicates
                common &= ~((1 << (b + 1)) - 1)
                while common:
                    lsb = common & -common
                    c = lsb.bit_length() - 1
                    remaining = common & adj[c]
                    if remaining:
                        return True
                    common ^= lsb
        return False

    def get_triangles(adj):
        """Return list of bitmasks for all triangles in the graph."""
        triangles = []
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                common &= ~((1 << (b + 1)) - 1)
                while common:
                    lsb = common & -common
                    c = lsb.bit_length() - 1
                    triangles.append((1 << a) | (1 << b) | (1 << c))
                    common ^= lsb
        return triangles

    def max_triangle_free_size(adj):
        """
        Exact branch and bound for largest triangle-free induced subgraph.
        Returns the size of the largest subset of vertices with no induced triangle.
        """
        triangles = get_triangles(adj)
        if not triangles:
            return n
        
        # Precompute for each vertex, which triangles it hits
        # tri_idx[v] = list of triangle indices containing v
        tri_idx = [[] for _ in range(n)]
        for idx, t in enumerate(triangles):
            m = t
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                tri_idx[v].append(idx)
                m ^= lsb
        
        # We want to find the maximum size subset S such that no triangle is fully contained in S.
        # Equivalent to: find minimum vertex cover of the hypergraph (vertices=V, edges=triangles).
        # Max triangle-free induced subgraph size = n - min_vertex_cover_size(triangles)
        
        best_cover = [n]
        
        def branch_and_bound(active_tris, chosen_count):
            if not active_tris:
                if chosen_count < best_cover[0]:
                    best_cover[0] = chosen_count
                return
            
            if chosen_count >= best_cover[0]:
                return
            
            # Heuristic: pick a triangle with fewest "options" or just first one
            # To improve pruning, pick the triangle that appears in the most active triangles?
            # Or just pick the first one.
            t = active_tris[0]
            
            # Option 1: Pick one of the vertices in t
            m = t
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                
                # Remove all triangles containing v
                new_active = []
                for tri in active_tris:
                    if not (tri & lsb):
                        new_active.append(tri)
                
                branch_and_bound(new_active, chosen_count + 1)
                m ^= lsb
                
                if time.time() > deadline - 5:
                    return
        
        branch_and_bound(triangles, 0)
        return n - best_cover[0]

    def score(adj):
        return max_triangle_free_size(adj)

    # --- Initial Construction: Paley(17) + Isolated Vertex ---
    # Paley graph of order 17 is a strongly regular graph (17, 8, 3, 4)
    # It is K4-free? Let's check.
    # Paley(17) is self-complementary.
    # It has triangles.
    # Is it K4-free?
    # Max clique size of Paley(17) is 4? No, Paley(17) has clique number 4?
    # Actually, Paley(17) has triangles. Does it have K4s?
    # Let's verify. If it has K4s, we can't use it directly.
    # Paley(17) is the unique SRG(17,8,3,4).
    # The complement is SRG(17,8,3,4).
    # Known result: Paley(17) is K4-free?
    # Let's check a small example.
    # If it has K4s, we need to remove edges.
    # However, the inspiration says "Paley(17)+isolated vertex provides ... known low triangle-free induced subgraph size (measured ~9)".
    # This implies the graph used was K4-free.
    # Let's construct Paley(17) and check for K4s. If any exist, we remove edges to break them.
    # But maybe Paley(17) IS K4-free?
    # Let's assume it is or handle it.
    
    # Construction of Paley(17)
    # Vertices 0..16
    # u ~ v if (u-v)^2 is a quadratic residue mod 17
    # QRs mod 17: 1, 2, 4, 8, 9, 13, 15, 16
    # (1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=8, 6^2=2, 7^2=15, 8^2=13)
    # So QR = {1, 2, 4, 8, 9, 13, 15, 16}
    
    def is_qr(x, p):
        return pow(x, (p-1)//2, p) == 1
    
    p = 17
    qr_set = set()
    for i in range(1, p):
        if is_qr(i, p):
            qr_set.add(i)
    
    adj_paley = [0] * 17
    for u in range(17):
        for v in range(u + 1, 17):
            diff = (u - v) % p
            if diff in qr_set:
                adj_paley[u] |= (1 << v)
                adj_paley[v] |= (1 << u)
    
    # Add isolated vertex 17
    adj = [0] * n
    for i in range(17):
        adj[i] = adj_paley[i]
    adj[17] = 0
    
    # Check K4 in Paley(17)
    if has_k4(adj):
        # If it has K4s, we need to remove edges.
        # This might ruin the properties.
        # But let's assume the inspiration implies it's usable or easily fixable.
        # Actually, Paley(17) is known to be triangle-free? No, it has triangles.
        # Is it K4-free?
        # A quick search suggests Paley(17) has clique number 4.
        # So it contains K4s.
        # We must remove edges to make it K4-free.
        # Strategy: Iterate through K4s and remove one edge.
        
        # Better: Use a greedy approach to find a maximal K4-free subgraph?
        # Or just remove edges from K4s.
        
        # Let's implement a simple K4 remover.
        # While has_k4:
        #   Find a K4 (a,b,c,d)
        #   Remove edge (a,b)
        
        # To find a K4 fast:
        def find_k4(adj):
            for a in range(n):
                for b in range(a + 1, n):
                    if not ((adj[a] >> b) & 1):
                        continue
                    common = adj[a] & adj[b]
                    common &= ~((1 << (b + 1)) - 1)
                    while common:
                        lsb = common & -common
                        c = lsb.bit_length() - 1
                        remaining = common & adj[c]
                        if remaining:
                            lsb2 = remaining & -remaining
                            d = lsb2.bit_length() - 1
                            return (a, b, c, d)
                        common ^= lsb
            return None
        
        # Remove edges until K4-free
        # This might be expensive if many K4s.
        # But Paley(17) is dense.
        # Let's try.
        while has_k4(adj):
            k4 = find_k4(adj)
            if not k4:
                break
            a, b, c, d = k4
            # Remove edge (a,b)
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            
            if time.time() > deadline - 20:
                break
        
        # If still has K4, we are in trouble.
        # But let's proceed.

    current_adj = adj[:]
    current_score = score(current_adj)
    best_adj = current_adj[:]
    best_score = current_score
    
    random.seed(42)
    
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append((i, j))
    
    # Local search
    def local_search(adj_init, max_iterations=5000):
        adj_local = adj_init[:]
        s_local = score(adj_local)
        no_improve = 0
        
        for _ in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            # Pick random edge to flip
            u, v = random.choice(all_edges)
            
            # Flip
            if (adj_local[u] >> v) & 1:
                adj_local[u] ^= (1 << v)
                adj_local[v] ^= (1 << u)
                removed = True
            else:
                adj_local[u] |= (1 << v)
                adj_local[v] |= (1 << u)
                removed = False
            
            # Check K4
            if has_k4(adj_local):
                # Revert
                if removed:
                    adj_local[u] |= (1 << v)
                    adj_local[v] |= (1 << u)
                else:
                    adj_local[u] ^= (1 << v)
                    adj_local[v] ^= (1 << u)
                no_improve += 1
                continue
            
            # Score
            s_new = score(adj_local)
            
            if s_new < s_local:
                s_local = s_new
                no_improve = 0
            else:
                # Revert
                if removed:
                    adj_local[u] |= (1 << v)
                    adj_local[v] |= (1 << u)
                else:
                    adj_local[u] ^= (1 << v)
                    adj_local[v] ^= (1 << u)
                no_improve += 1
            
            if no_improve > 100:
                # Perturb
                for _ in range(5):
                    u2, v2 = random.choice(all_edges)
                    if (adj_local[u2] >> v2) & 1:
                        adj_local[u2] ^= (1 << v2)
                        adj_local[v2] ^= (1 << u2)
                    else:
                        adj_local[u2] |= (1 << v2)
                        adj_local[v2] |= (1 << u2)
                    if has_k4(adj_local):
                        # Revert
                        if (adj_local[u2] >> v2) & 1:
                            adj_local[u2] ^= (1 << v2)
                            adj_local[v2] ^= (1 << u2)
                        else:
                            adj_local[u2] |= (1 << v2)
                            adj_local[v2] |= (1 << u2)
                no_improve = 0
                
        return adj_local, s_local

    # Run local search on initial Paley graph
    if time.time() < deadline - 20:
        best_adj, best_score = local_search(current_adj)
    
    # Try Turan graph as a backup if Paley didn't work well or time permits
    # Turan T(18,3)
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    adj_turan = [0] * n
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj_turan[a] |= (1 << b)
                    adj_turan[b] |= (1 << a)
    
    if time.time() < deadline - 30:
        s_turan = score(adj_turan)
        if s_turan < best_score:
            best_adj, best_score = local_search(adj_turan)
        else:
            # Try local search on Turan if we have time
            if time.time() < deadline - 40:
                adj_res, s_res = local_search(adj_turan, max_iterations=2000)
                if s_res < best_score:
                    best_adj = adj_res
                    best_score = s_res

    return edges_from_adj(best_adj)
