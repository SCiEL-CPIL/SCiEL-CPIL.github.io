"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    n = 18
    
    # Helper: check if a set of vertices (bitmask) induces a triangle-free subgraph
    # adj_mask[v] is a bitmask of neighbors of v
    def is_triangle_free(mask, adj_mask):
        # Extract vertices in mask
        verts = []
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        # Check all triples
        for i in range(len(verts)):
            vi = verts[i]
            for j in range(i+1, len(verts)):
                vj = verts[j]
                if adj_mask[vi] & (1 << vj):
                    # Edge exists, check if it forms a triangle with any k
                    # For each k in verts after j, check if both vi-vk and vj-vk exist
                    edge_mask = adj_mask[vi] & adj_mask[vj]
                    # Only consider vertices in mask that are after j to avoid duplicates
                    for k in range(j+1, len(verts)):
                        vk = verts[k]
                        if edge_mask & (1 << vk):
                            return False
        return True
    
    # Helper: find maximum triangle-free induced subgraph size using branch and bound
    def max_triangle_free_size(adj_mask):
        # We want to find the largest subset S such that G[S] is triangle-free
        # Use branch and bound: try to include or exclude each vertex
        # Upper bound: current_size + remaining_vertices
        # Lower bound: best found so far
        
        best = [0]
        
        def dfs(current_mask, current_size, next_vertex):
            if current_size + (n - next_vertex) <= best[0]:
                return
            if next_vertex == n:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Option 1: exclude vertex next_vertex
            dfs(current_mask, current_size, next_vertex + 1)
            
            # Option 2: include vertex next_vertex
            # Check if adding next_vertex to current_mask keeps it triangle-free
            nv = next_vertex
            new_mask = current_mask | (1 << nv)
            
            # Check triangles formed by nv with two existing vertices in current_mask
            ok = True
            # For each pair (u, v) in current_mask where u-v is an edge, check if nv connects to both
            # More efficiently: for each neighbor u of nv in current_mask, check if any neighbor v of nv in current_mask (v > u) has edge u-v
            neighbors_in_current = adj_mask[nv] & current_mask
            # For each pair of neighbors in current that are connected, we have a triangle
            m = neighbors_in_current
            while m:
                lsb = m & (-m)
                u = lsb.bit_length() - 1
                # Check if u has any neighbor v in (neighbors_in_current & (m - lsb)) -- i.e., other neighbors of nv in current
                if adj_mask[u] & (neighbors_in_current ^ lsb):
                    ok = False
                    break
                m ^= lsb
            
            if ok:
                dfs(new_mask, current_size + 1, next_vertex + 1)
        
        dfs(0, 0, 0)
        return best[0]
    
    # Helper: check if graph is K4-free using bitmask
    def is_k4_free(adj_mask):
        # For each edge (u,v), check if there's a common neighbor w, and if w has another common neighbor with u (or v)
        # Actually: K4 exists iff there's a triangle (u,v,w) and all three have a common neighbor in the triangle
        # Simpler: for each triple (u,v,w) that forms a triangle, check if any pair has a 4th common neighbor
        # Even simpler: for each edge (u,v), let C = common neighbors. If |C| >= 2 and any two in C are adjacent, we have K4
        # Wait, that's not quite right. K4 = {u,v,w,x} where all 6 edges exist.
        # For each pair (u,v) that is an edge, let C = N(u) ∩ N(v). If there exist w,x in C with w-x edge, then {u,v,w,x} is K4.
        for u in range(n):
            for v in range(u+1, n):
                if adj_mask[u] & (1 << v):
                    C = adj_mask[u] & adj_mask[v]
                    # Check if any two vertices in C are adjacent
                    m = C
                    while m:
                        lsb = m & (-m)
                        w = lsb.bit_length() - 1
                        if adj_mask[w] & (C ^ lsb):
                            return False
                        m ^= lsb
        return True
    
    # Helper: generate a random sparse graph and check K4-free
    def generate_random_graph():
        """Generate a random graph with controlled density, check K4-free."""
        adj_mask = [0] * n
        # Start with random edges
        # Use a probabilistic approach: for each potential edge, include with probability p
        # Try different densities
        for p in [0.3, 0.35, 0.4, 0.45, 0.5]:
            for _ in range(50):
                adj_mask = [0] * n
                for u in range(n):
                    for v in range(u+1, n):
                        if random.random() < p:
                            adj_mask[u] |= (1 << v)
                            adj_mask[v] |= (1 << u)
                if is_k4_free(adj_mask):
                    return adj_mask
        return None
    
    # Helper: local search to improve (reduce max triangle-free size)
    def local_search(adj_mask, time_limit=10.0):
        """Try to reduce the max triangle-free size by flipping edges."""
        best_score = max_triangle_free_size(adj_mask)
        current_score = best_score
        t_start = time.time()
        
        while time.time() - t_start < time_limit:
            # Try flipping a random edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            # Flip edge
            adj_mask[u] ^= (1 << v)
            adj_mask[v] ^= (1 << u)
            
            if not is_k4_free(adj_mask):
                # Revert
                adj_mask[u] ^= (1 << v)
                adj_mask[v] ^= (1 << u)
                continue
            
            new_score = max_triangle_free_size(adj_mask)
            if new_score <= current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
            else:
                # Revert with some probability (simulated annealing style)
                if random.random() < 0.3:
                    adj_mask[u] ^= (1 << v)
                    adj_mask[v] ^= (1 << u)
                else:
                    # Keep the change even if worse (exploration)
                    pass
        
        return adj_mask, best_score
    
    # Start with the Turán graph as baseline
    def turan_graph_parts():
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        adj_mask = [0] * n
        for i in range(3):
            for j in range(i+1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj_mask[a] |= (1 << b)
                        adj_mask[b] |= (1 << a)
        return adj_mask
    
    best_adj = turan_graph_parts()
    best_score = max_triangle_free_size(best_adj)  # Should be 12
    
    # Try random generation + local search
    random.seed(42)
    
    while time.time() < DEADLINE - 5:
        # Generate a random K4-free graph
        adj = generate_random_graph()
        if adj is None:
            continue
        
        score = max_triangle_free_size(adj)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
        
        # Local search to improve
        if time.time() < DEADLINE - 10:
            adj_ls, score_ls = local_search(adj, time_limit=min(5.0, DEADLINE - time.time() - 10))
            if score_ls < best_score:
                best_score = score_ls
                best_adj = adj_ls[:]
        
        # Also try perturbing the current best
        if time.time() < DEADLINE - 10:
            adj_pert = best_adj[:]
            # Flip a few random edges
            for _ in range(random.randint(1, 5)):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                adj_pert[u] ^= (1 << v)
                adj_pert[v] ^= (1 << u)
            
            if is_k4_free(adj_pert):
                score_pert = max_triangle_free_size(adj_pert)
                if score_pert < best_score:
                    best_score = score_pert
                    best_adj = adj_pert[:]
                
                if time.time() < DEADLINE - 10:
                    adj_ls2, score_ls2 = local_search(adj_pert, time_limit=min(5.0, DEADLINE - time.time() - 10))
                    if score_ls2 < best_score:
                        best_score = score_ls2
                        best_adj = adj_ls2[:]
    
    # Convert best_adj to edge list
    edges = []
    for u in range(n):
        m = adj_mask_best = best_adj[u]
        # Only add edges where v > u to avoid duplicates
        for v in range(u+1, n):
            if m & (1 << v):
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
