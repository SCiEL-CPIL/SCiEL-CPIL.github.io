import random
import time
import sys

sys.setrecursionlimit(10000)

def _time_up(deadline):
    return time.time() > deadline

def _check_triangle_free(adj, n):
    for i in range(n):
        for j in adj[i]:
            if j < i:
                continue
            if adj[i] & adj[j]:
                return False
    return True

def _exact_alpha(adj, n, deadline):
    """
    Returns the size of the maximum independent set.
    Uses branch and bound with bitmasks.
    """
    masks = [0] * n
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        masks[i] = m
        
    best = [0]
    
    def dfs(candidates, current_size):
        if _time_up(deadline):
            return
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
            
        # Upper bound: current_size + popcount(candidates)
        if current_size + bin(candidates).count('1') <= best[0]:
            return
            
        # Choose a vertex to branch on
        # Heuristic: pick vertex with highest degree in the induced subgraph (candidates)
        # This is the "include/exclude" branch.
        # Actually, for Independent Set, we pick a vertex v.
        # Option 1: Include v. Remove v and its neighbors from candidates.
        # Option 2: Exclude v. Remove v from candidates.
        
        # To prune better, pick v that maximizes neighbors in candidates?
        # Or just pick the first one?
        # Let's pick the one with max degree in candidates to prune "Include" branch faster?
        # If v has many neighbors, including it removes many candidates.
        
        best_v = -1
        max_deg = -1
        temp_cands = candidates
        while temp_cands:
            lsb = temp_cands & -temp_cands
            v = lsb.bit_length() - 1
            deg = bin(candidates & masks[v]).count('1')
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp_cands ^= lsb
            
        v = best_v
        if v == -1:
            return
            
        bit_v = 1 << v
        
        # Option 1: Include v
        # New candidates: candidates & ~masks[v] & ~bit_v
        new_cands_1 = candidates & ~masks[v] & ~bit_v
        dfs(new_cands_1, current_size + 1)
        
        # Option 2: Exclude v
        # New candidates: candidates & ~bit_v
        new_cands_2 = candidates & ~bit_v
        dfs(new_cands_2, current_size)
        
    all_mask = (1 << n) - 1
    dfs(all_mask, 0)
    
    return best[0]

def _find_max_independent_set(adj, n, deadline):
    """
    Returns the actual set of vertices in a max independent set.
    """
    masks = [0] * n
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        masks[i] = m
        
    best_set = [0]
    best_size = [0]
    
    def dfs(candidates, current_set, current_size):
        if _time_up(deadline):
            return
        if not candidates:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_set[0] = current_set
            return
            
        if current_size + bin(candidates).count('1') <= best_size[0]:
            return
            
        # Pick vertex with max degree in candidates
        best_v = -1
        max_deg = -1
        temp_cands = candidates
        while temp_cands:
            lsb = temp_cands & -temp_cands
            v = lsb.bit_length() - 1
            deg = bin(candidates & masks[v]).count('1')
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp_cands ^= lsb
            
        v = best_v
        if v == -1:
            return
            
        bit_v = 1 << v
        
        # Option 1: Include v
        new_cands_1 = candidates & ~masks[v] & ~bit_v
        dfs(new_cands_1, current_set | bit_v, current_size + 1)
        
        # Option 2: Exclude v
        new_cands_2 = candidates & ~bit_v
        dfs(new_cands_2, current_set, current_size)
        
    all_mask = (1 << n) - 1
    dfs(all_mask, 0, 0)
    
    return best_set[0]

def construct():
    deadline = time.time() + 115.0
    target_n = 39
    n = target_n
    
    # Fallback if we fail to find a valid graph
    # Use a known small valid graph? 
    # The seed used 26. We can try to return a valid graph on 27 (Andrasfai) if 39 fails.
    # But let's focus on 39.
    
    best_n = 0
    best_edges = []
    
    # Strategy: Local Search
    # 1. Generate random maximal triangle-free graph
    # 2. Check alpha
    # 3. If alpha >= 10, try to reduce alpha by adding edges within max independent set
    # 4. If stuck, kick (remove random edge) and re-maximize
    
    restarts = 0
    while _time_up(deadline) is False and restarts < 500:
        restarts += 1
        
        # Initialize empty graph
        adj = [0] * n  # Bitmasks
        
        # Generate maximal triangle-free graph
        # Random order of edges
        edges_list = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges_list)
        
        for u, v in edges_list:
            if _time_up(deadline):
                break
            # Check if adding (u,v) creates triangle
            # Triangle if common neighbor exists
            if adj[u] & adj[v] == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Verify triangle-free (safety)
        if not _check_triangle_free(adj, n):
            continue
            
        # Check alpha
        alpha = _exact_alpha(adj, n, deadline)
        
        if alpha < 10:
            # Convert to edge list
            edges = []
            for i in range(n):
                m = adj[i]
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    if i < j:
                        edges.append([i, j])
                    m ^= lsb
            if n > best_n:
                best_n = n
                best_edges = edges
            break
        
        # If alpha >= 10, try to improve
        # Find max independent set S
        S_mask = _find_max_independent_set(adj, n, deadline)
        
        # Try to add edges within S
        # Vertices in S are independent, so no edges between them currently.
        # Adding an edge (u,v) in S is safe for triangles ONLY IF u and v have no common neighbors.
        # But wait, if u,v are in S, they have no common neighbors in S.
        # They might have common neighbors outside S.
        # If they have a common neighbor w, then u-w-v is a path.
        # Adding u-v creates triangle u-w-v.
        # So we can only add (u,v) if adj[u] & adj[v] == 0.
        
        improved = False
        vertices_in_S = []
        m = S_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            vertices_in_S.append(v)
            m ^= lsb
            
        # Try to add edges between pairs in S
        for i in range(len(vertices_in_S)):
            for j in range(i+1, len(vertices_in_S)):
                u = vertices_in_S[i]
                v = vertices_in_S[j]
                if adj[u] & adj[v] == 0:
                    # Safe to add
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
                    # Re-check alpha immediately? 
                    # Or just continue adding?
                    # Adding edges reduces alpha.
                    # Let's add all possible safe edges in S.
        
        if improved:
            # Re-evaluate alpha
            alpha = _exact_alpha(adj, n, deadline)
            if alpha < 10:
                edges = []
                for i in range(n):
                    m = adj[i]
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        if i < j:
                            edges.append([i, j])
                        m ^= lsb
                if n > best_n:
                    best_n = n
                    best_edges = edges
                break
        
        # If no improvement, try a kick
        # Remove a random edge
        if restarts % 10 == 0:
            # Find a random edge to remove
            all_edges = []
            for i in range(n):
                m = adj[i]
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    if i < j:
                        all_edges.append((i, j))
                    m ^= lsb
            
            if all_edges:
                u, v = random.choice(all_edges)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                # Re-maximize: add all possible edges
                edges_list = [(i, j) for i in range(n) for j in range(i+1, n)]
                random.shuffle(edges_list)
                for u, v in edges_list:
                    if _time_up(deadline):
                        break
                    if adj[u] & adj[v] == 0:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)

    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Fallback to a known valid graph if possible?
        # Andrasfai graph An(9) has n=27, alpha=9.
        # Construction of An(9):
        # Vertices (i, j) with i in 0..2, j in 0..8?
        # Actually, An(n) is a 3-regular graph on 3n vertices.
        # It is a blow-up of C3? No, C3 has triangles.
        # An(n) is constructed from Z_n x {0,1,2}.
        # Edges: (i,0)-(i+1,1), (i,1)-(i+1,2), (i,2)-(i+1,0)?
        # This is a cycle of length 3n.
        # Wait, a cycle is triangle-free.
        # Alpha of C_{3n} is floor(3n/2).
        # For n=9, 3n=27. Alpha = 13.
        # So An(9) is NOT a simple cycle.
        
        # Let's just return the seed's fallback (n=26) to be safe.
        n = 26
        return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
