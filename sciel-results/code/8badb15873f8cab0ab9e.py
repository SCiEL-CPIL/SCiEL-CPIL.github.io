import time
import random

def construct():
    n = 16
    deadline = time.time() + 110  # 110 seconds to be safe
    
    # Helper: check if adding edge (u, v) creates a triangle
    def is_valid_add(u, v, adj):
        if (adj[u] >> v) & 1:
            return False
        if (adj[u] & adj[v]):
            return False
        return True
    
    # Helper: exact 3-colorability check using bitmask backtracking (optimized)
    def is_3_colorable(adj):
        # Precompute neighbor bitmasks
        # Use a faster backtracking with bitmasks for colors
        colors = [-1] * n
        
        def backtrack(uncolored):
            if uncolored == 0:
                return True
            
            # Pick vertex with fewest available colors (MRV heuristic)
            best_v = -1
            best_avail = 4
            best_avail_mask = 0
            for v in range(n):
                if (uncolored >> v) & 1:
                    # Compute forbidden colors from already-colored neighbors
                    forbidden = 0
                    nb = adj[v]
                    # Check each color
                    for c in range(3):
                        # If any colored neighbor has color c, c is forbidden
                        # We can check: for each neighbor u, if colors[u] == c
                        # But we need to check all neighbors
                        conflict = False
                        nb_mask = nb
                        while nb_mask:
                            lsb = nb_mask & (-nb_mask)
                            u = lsb.bit_length() - 1
                            if colors[u] == c:
                                conflict = True
                                break
                            nb_mask ^= lsb
                        if not conflict:
                            forbidden |= (1 << c)
                    avail = 3 - bin(forbidden).count('1')
                    if avail < best_avail:
                        best_avail = avail
                        best_v = v
                        best_avail_mask = forbidden
                        if avail == 0:
                            return False
            
            if best_v == -1:
                return True
            
            for c in range(3):
                if not (best_avail_mask >> c) & 1:
                    colors[best_v] = c
                    if backtrack(uncolored & ~(1 << best_v)):
                        return True
                    colors[best_v] = -1
            
            return False
        
        return backtrack((1 << n) - 1)
    
    # Strategy: Start from complete graph, remove edges to eliminate triangles
    def greedy_triangle_free(adj_start, randomize=False):
        """Start from adj_start, remove edges to make triangle-free, maximizing remaining edges."""
        adj = adj_start[:]
        
        # Build list of all edges with their triangle counts
        edges = {}
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges[(u, v)] = bin(adj[u] & adj[v]).count('1')
        
        # Iteratively remove the edge with the fewest triangles (but > 0)
        while True:
            if time.time() > deadline:
                break
            
            # Find edge with minimum positive triangle count
            min_count = float('inf')
            min_edge = None
            if randomize:
                # Collect all edges with min positive count
                candidates = []
                for e, c in edges.items():
                    if 0 < c < min_count:
                        min_count = c
                        candidates = [e]
                    elif c == min_count:
                        candidates.append(e)
                if not candidates:
                    break
                min_edge = random.choice(candidates)
            else:
                for e, c in edges.items():
                    if 0 < c < min_count:
                        min_count = c
                        min_edge = e
            
            if min_edge is None:
                break  # No triangles left
            
            # Remove min_edge
            u, v = min_edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            del edges[(u, v)]
            
            # Update triangle counts for affected edges
            for w in range(n):
                if w != u and w != v and (adj[u] >> w) & 1:
                    e = (min(u, w), max(u, w))
                    if e in edges:
                        edges[e] = bin(adj[u] & adj[w]).count('1')
            
            for w in range(n):
                if w != u and w != v and (adj[v] >> w) & 1:
                    e = (min(v, w), max(v, w))
                    if e in edges:
                        edges[e] = bin(adj[v] & adj[w]).count('1')
        
        return adj, edges
    
    # Local search: try to add edges to a triangle-free graph while maintaining triangle-freeness
    def local_search_add_edges(adj_start):
        """Try to add more edges to a triangle-free graph."""
        adj = adj_start[:]
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.add((u, v))
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            candidates = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) not in edges:
                        if is_valid_add(u, v, adj):
                            candidates.append((u, v))
            
            if not candidates:
                break
            
            # Try adding each candidate edge and check if it helps
            for u, v in random.sample(candidates, min(len(candidates), 20)):
                if is_valid_add(u, v, adj):
                    # Try adding this edge
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    edges.add((u, v))
                    improved = True
                    break
        
        return adj, edges
    
    # Multiple strategies to try
    
    best_count = 0
    best_edges = []
    
    # Strategy 1: Greedy triangle removal from complete graph (deterministic)
    complete_adj = [(1 << n) - 1] * n
    for i in range(n):
        complete_adj[i] &= ~(1 << i)
    
    adj1, edges1 = greedy_triangle_free(complete_adj)
    count1 = len(edges1)
    if count1 > best_count:
        if not is_3_colorable(adj1):
            best_count = count1
            best_edges = [[u, v] for u, v in edges1.keys()]
    
    # Strategy 2: Randomized greedy with different tie-breaking
    for restart in range(800):
        if time.time() > deadline:
            break
        
        adj = complete_adj[:]
        for i in range(n):
            adj[i] &= ~(1 << i)
        
        adj_r, edges_r = greedy_triangle_free(adj, randomize=True)
        count = len(edges_r)
        if count > best_count:
            if not is_3_colorable(adj_r):
                best_count = count
                best_edges = [[u, v] for u, v in edges_r.keys()]
    
    # Strategy 3: Start from Grötzsch graph seed and add edges via local search
    seed_edges = []
    for i in range(5):
        seed_edges.append((i, (i + 1) % 5))
    for u, w in [(i, (i + 1) % 5) for i in range(5)]:
        seed_edges.append((u, 5 + w))
        seed_edges.append((w, 5 + u))
    for u in range(5):
        seed_edges.append((5 + u, 10))
        
    adj_seed = [0] * n
    for u, v in seed_edges:
        adj_seed[u] |= 1 << v
        adj_seed[v] |= 1 << u
        
    for restart in range(400):
        if time.time() > deadline:
            break
        
        adj_ls, edges_ls = local_search_add_edges(adj_seed)
        count = len(edges_ls)
        if count > best_count:
            if not is_3_colorable(adj_ls):
                best_count = count
                best_edges = [[u, v] for u, v in edges_ls]
    
    # Strategy 4: Randomized construction - build triangle-free graph greedily
    for restart in range(500):
        if time.time() > deadline:
            break
        
        adj = [0] * n
        # Randomly decide edge presence, ensuring no triangles
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if is_valid_add(u, v, adj):
                adj[u] |= 1 << v
                adj[v] |= 1 << u
        
        edges_set = set()
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges_set.add((u, v))
        
        count = len(edges_set)
        if count > best_count:
            if not is_3_colorable(adj):
                best_count = count
                best_edges = [[u, v] for u, v in edges_set]
    
    # If no valid graph found, return empty (shouldn't happen)
    if not best_edges:
        # Fallback: return Grötzsch graph extended with isolated vertices
        return [[u, v] for u, v in seed_edges]
    
    return best_edges
