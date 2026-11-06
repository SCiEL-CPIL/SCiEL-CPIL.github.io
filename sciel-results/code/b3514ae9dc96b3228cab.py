import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin before 120s limit
    
    n = 18
    full_mask = (1 << n) - 1
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) creates a K4."""
        if adj[u] & (1 << v):
            return True
        common = adj[u] & adj[v]
        c = common
        while c:
            lsb = c & -c
            w = lsb.bit_length() - 1
            if adj[w] & (common & ~lsb):
                return False
            c ^= lsb
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_triangles(adj):
        """Return list of triangle bitmasks."""
        triangles = []
        for u in range(n):
            v_mask = adj[u] & ~((1 << (u + 1)) - 1)
            while v_mask:
                lsb = v_mask & -v_mask
                v = lsb.bit_length() - 1
                v_mask ^= lsb
                common = adj[u] & adj[v] & ~((1 << (v + 1)) - 1)
                while common:
                    wlsb = common & -common
                    w = wlsb.bit_length() - 1
                    common ^= wlsb
                    triangles.append((1 << u) | (1 << v) | (1 << w))
        return triangles

    def evaluate(adj):
        """
        Find the size of the largest triangle-free induced subgraph.
        Uses minimum triangle transversal (hitting set) approach.
        """
        triangles = get_triangles(adj)
        if not triangles:
            return n
        
        num_tris = len(triangles)
        
        # Precompute: for each vertex, which triangles it hits (bitmask)
        vertex_hits = [0] * n
        for v in range(n):
            mask = 1 << v
            hit = 0
            for i, tri in enumerate(triangles):
                if tri & mask:
                    hit |= (1 << i)
            vertex_hits[v] = hit
        
        all_tris_mask = (1 << num_tris) - 1
        
        # Sort vertices by number of triangles they hit (descending) for better pruning
        # But we need to track original indices
        vertex_order = sorted(range(n), key=lambda v: bin(vertex_hits[v]).count('1'), reverse=True)
        
        # For m from 1 to n: check if a transversal of size m exists
        for m in range(1, n + 1):
            found = False
            
            def backtrack(idx, chosen_count, covered):
                nonlocal found
                if found:
                    return
                if covered == all_tris_mask:
                    found = True
                    return
                if chosen_count == m:
                    return
                
                # Pruning: if we can't cover all remaining triangles with remaining picks
                remaining_tris = all_tris_mask & ~covered
                # Check if remaining vertices can cover remaining triangles
                if idx >= n:
                    return
                
                # Upper bound on coverage: OR of all remaining vertex hits
                rem_cover = 0
                for j in range(idx, n):
                    rem_cover |= vertex_hits[vertex_order[j]]
                    if (covered | rem_cover) == all_tris_mask and (chosen_count + (n - idx)) < m:
                        break
                if (covered | rem_cover) != all_tris_mask:
                    return
                
                for j in range(idx, n):
                    v = vertex_order[j]
                    new_covered = covered | vertex_hits[v]
                    if new_covered == covered:
                        continue
                    backtrack(j + 1, chosen_count + 1, new_covered)
                    if found:
                        return
            
            backtrack(0, 0, 0)
            if found:
                return n - m
        
        return 1

    # Initialize with K6,6,6 complete tripartite
    def init_k666():
        adj = [0] * n
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        return adj

    # Also try a few random K4-free initializations
    def generate_random_k4_free():
        adj = [0] * n
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        for u, v in edges:
            if can_add_edge(adj, u, v):
                add_edge(adj, u, v)
        return adj

    # Generate multiple initial candidates
    best_adj = None
    best_score = n + 1
    
    # Start with K6,6,6
    candidate = init_k666()
    score = evaluate(candidate)
    best_score = score
    best_adj = candidate[:]
    
    # Try random starts
    for r in range(8):
        if time.time() > deadline - 5:
            break
        candidate = generate_random_k4_free()
        score = evaluate(candidate)
        if score < best_score:
            best_score = score
            best_adj = candidate[:]
    
    current_adj = best_adj[:]
    current_score = best_score
    
    # Local search: try all single-edge modifications systematically
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # Try all possible edge additions
        for u in range(n):
            for v in range(u + 1, n):
                if time.time() > deadline - 2:
                    break
                if current_adj[u] & (1 << v):
                    continue
                if not can_add_edge(current_adj, u, v):
                    continue
                
                old_u = current_adj[u]
                old_v = current_adj[v]
                add_edge(current_adj, u, v)
                new_score = evaluate(current_adj)
                
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = current_adj[:]
                    improved = True
                else:
                    current_adj[u] = old_u
                    current_adj[v] = old_v
            if time.time() > deadline - 2:
                break
        
        # Try all possible edge removals (less likely to help but can escape local minima)
        if time.time() > deadline - 2:
            break
        for u in range(n):
            for v in range(u + 1, n):
                if time.time() > deadline - 2:
                    break
                if not (current_adj[u] & (1 << v)):
                    continue
                
                old_u = current_adj[u]
                old_v = current_adj[v]
                remove_edge(current_adj, u, v)
                new_score = evaluate(current_adj)
                
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = current_adj[:]
                    improved = True
                else:
                    current_adj[u] = old_u
                    current_adj[v] = old_v
            if time.time() > deadline - 2:
                break
    
    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    
    return edges
