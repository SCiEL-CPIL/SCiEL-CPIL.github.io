import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin

    def has_triangle(n, edges):
        # Adjacency matrix for speed
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    for k in range(j + 1, n):
                        if adj[i][k] and adj[j][k]:
                            return True
        return False

    def max_independent_set(n, edges):
        # Bitmask representation for speed if n <= 64
        if n > 64:
            # Fallback to set-based or simple B&B (slow)
            pass
        mask = [0] * n
        for u, v in edges:
            mask[u] |= (1 << v)
            mask[v] |= (1 << u)
        
        # Bron-Kerbosch style or simple B&B for Max Independent Set
        # We want to know if alpha >= 10
        # So we search for a clique of size 10 in the complement graph
        # Complement edges: ~mask[i] & ~mask[j] ...
        # Let's use a simple recursive search with pruning
        
        # Precompute complement adjacency
        comp_adj = [0] * n
        for i in range(n):
            comp_adj[i] = ((1 << n) - 1) ^ mask[i] ^ (1 << i)
            
        # We want to find a clique of size 10 in comp_adj
        # Which is an independent set of size 10 in original
        best = 0
        def dfs(candidates, size):
            nonlocal best
            if time.time() > deadline:
                return
            if size > best:
                best = size
            if not candidates:
                return
            # Pruning: if size + popcount(candidates) <= best, return
            if size + bin(candidates).count('1') <= best:
                return
            
            # Pick a vertex
            # Heuristic: pick vertex with highest degree in candidate subgraph?
            # For clique in comp, we want high degree in comp.
            # Let's just pick the first one
            v_bit = candidates & -candidates
            v = v_bit.bit_length() - 1
            candidates_without_v = candidates ^ v_bit
            
            # Branch 1: Include v
            # New candidates are neighbors of v in comp that are in candidates
            new_cand = candidates_without_v & comp_adj[v]
            dfs(new_cand, size + 1)
            
            # Branch 2: Exclude v
            dfs(candidates_without_v, size)

        dfs((1 << n) - 1, 0)
        return best

    def is_valid(n, edges):
        if has_triangle(n, edges):
            return False
        alpha = max_independent_set(n, edges)
        return alpha < 10

    # Base case: Andrasfai graph on 26 vertices
    def get_andrasfai(k):
        n = 3 * k
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n % 3 == 1:
                    edges.append([i, j])
        return n, edges

    best_result = None
    best_n = 0

    # Try n=36 (k=12)
    n, edges = get_andrasfai(12)
    # Check if base is valid (it's not, alpha=11)
    # Try to add edges
    # Candidate edges: pairs with dist >= 3
    # First, build distance matrix or just check common neighbors
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    def common_neighbors(u, v):
        count = 0
        for k in range(n):
            if adj[u][k] and adj[v][k]:
                count += 1
        return count

    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u][v] and common_neighbors(u, v) == 0:
                candidates.append((u, v))

    random.shuffle(candidates)
    
    current_edges = edges[:]
    
    # Greedy addition
    improved = False
    for u, v in candidates:
        if time.time() > deadline:
            break
        # Check if adding (u,v) keeps it triangle-free
        # We already checked common neighbors, so it's safe
        new_edges = current_edges + [[u, v]]
        # Check alpha
        # Optimization: only check if alpha might drop
        # Just run the checker
        if is_valid(n, new_edges):
            current_edges = new_edges
            improved = True
            # If we reach alpha < 10, we are done
            # is_valid returns True if alpha < 10
            # So if it returns True, we have a valid graph
            best_result = {'n': n, 'edges': current_edges}
            best_n = n
            break
            
    if best_result:
        return best_result

    # Try n=33 (k=11)
    n, edges = get_andrasfai(11)
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u][v] and common_neighbors(u, v) == 0:
                candidates.append((u, v))
                
    random.shuffle(candidates)
    current_edges = edges[:]
    
    for u, v in candidates:
        if time.time() > deadline:
            break
        new_edges = current_edges + [[u, v]]
        if is_valid(n, new_edges):
            current_edges = new_edges
            best_result = {'n': n, 'edges': current_edges}
            best_n = n
            break

    if best_result:
        return best_result

    # Fallback to n=26
    n, edges = get_andrasfai(26 // 3 if 26 % 3 == 0 else 9) # 26 is not 3k
    # Seed graph is specific: n=26, diff % 3 == 1
    n = 26
    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
    return {'n': n, 'edges': edges}
