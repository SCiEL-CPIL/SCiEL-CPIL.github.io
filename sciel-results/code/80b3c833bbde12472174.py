import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
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

    # --- Initial Construction: Turan Graph T(18, 3) ---
    # Parts of size 6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    adj = [0] * n
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
    
    current_adj = adj[:]
    current_score = score(current_adj)
    best_adj = current_adj[:]
    best_score = current_score
    
    random.seed(42)
    
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append((i, j))
    
    no_improve_count = 0
    step = 0
    
    # Multi-start strategy:
    # 1. Start with T(18,3)
    # 2. If stuck, try T(18,4), T(18,5) or random K4-free perturbations
    
    start_structures = [
        [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))], # 3 parts
        [list(range(0, 5)), list(range(5, 10)), list(range(10, 14)), list(range(14, 18))], # 4 parts (5,5,4,4)
        [list(range(0, 4)), list(range(4, 8)), list(range(8, 12)), list(range(12, 16)), list(range(16, 18))], # 5 parts (4,4,4,4,2)
    ]
    
    # We already initialized with the first one. Let's run local search.
    
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

    # Run local search on initial Turan graph
    if time.time() < deadline - 20:
        best_adj, best_score = local_search(current_adj)
    
    # Try other structures if time permits
    for struct in start_structures[1:]:
        if time.time() > deadline - 20:
            break
        
        adj_struct = [0] * n
        for i in range(len(struct)):
            for j in range(i + 1, len(struct)):
                for a in struct[i]:
                    for b in struct[j]:
                        adj_struct[a] |= (1 << b)
                        adj_struct[b] |= (1 << a)
        
        if has_k4(adj_struct):
            continue
            
        s_struct = score(adj_struct)
        if s_struct < best_score:
            best_adj, best_score = local_search(adj_struct)
        else:
            # Even if not better initially, maybe local search can improve it?
            # Given time constraints, only pursue if promising or if we have lots of time
            if time.time() < deadline - 30:
                adj_res, s_res = local_search(adj_struct, max_iterations=2000)
                if s_res < best_score:
                    best_adj = adj_res
                    best_score = s_res

    return edges_from_adj(best_adj)
