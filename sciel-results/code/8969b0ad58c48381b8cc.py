import random
import time

def construct():
    # Start with the seed graph
    n_h = 9
    edges_h = set()
    for u in range(n_h):
        i1, j1 = divmod(u, 3)
        for v in range(u + 1, n_h):
            i2, j2 = divmod(v, 3)
            if i1 != i2 and j1 != j2:
                edges_h.add((u, v))
    
    edges_g = set()
    for u in range(18):
        b_u = u // 2
        for v in range(u + 1, 18):
            b_v = v // 2
            if b_u != b_v:
                if (b_u, b_v) in edges_h or (b_v, b_u) in edges_h:
                    edges_g.add((u, v))
    
    # Local search to improve
    # We want to minimize the size of the largest triangle-free induced subgraph.
    # Strategy: Try to add edges to create more triangles, or remove edges to break K4s?
    # Actually, adding edges to a K4-free graph might create K4s.
    # We check K4-freeness after each move.
    
    def is_k4_free(adj):
        # adj is a set of (u,v) with u<v
        # Check all 4-cliques
        # Optimization: iterate over edges, then common neighbors
        for u in range(18):
            for v in range(u+1, 18):
                if (u, v) not in adj:
                    continue
                # Find common neighbors of u and v
                common = [w for w in range(18) if w != u and w != v and (u, w) in adj and (v, w) in adj]
                # Check if any pair in common is connected
                for i in range(len(common)):
                    for j in range(i+1, len(common)):
                        c1, c2 = common[i], common[j]
                        if (min(c1,c2), max(c1,c2)) in adj:
                            return False
        return True

    def max_triangle_free_size(adj):
        # Branch and bound for max triangle-free induced subgraph
        # This is equivalent to finding the max subset with no triangles.
        # We can use a simple recursion.
        # For 18 vertices, this is feasible if pruned well.
        
        # Precompute triangles
        # A set S is triangle-free if no 3 vertices in S form a triangle.
        
        best = 0
        # Order vertices by degree?
        # Use a greedy initial bound
        def dfs(vertices, current_size):
            nonlocal best
            if not vertices:
                if current_size > best:
                    best = current_size
                return
            
            # Bound: current_size + len(vertices) <= best -> prune
            if current_size + len(vertices) <= best:
                return
            
            # Pick a vertex
            # Heuristic: pick vertex with highest degree in induced subgraph?
            # Or just first?
            v = vertices[0]
            rest = vertices[1:]
            
            # Option 1: Exclude v
            dfs(rest, current_size)
            
            # Option 2: Include v
            # If we include v, we must remove any vertex w such that v,w form an edge?
            # No, we just need to ensure no triangle is formed.
            # If we include v, we cannot include any pair (w1, w2) from rest that are both connected to v AND connected to each other.
            # i.e., if w1, w2 are in rest, and (v,w1) in adj, (v,w2) in adj, (w1,w2) in adj, then we can't have {v, w1, w2}.
            # This constraint is complex for simple DFS.
            
            # Alternative: Iterate over all subsets? 2^18 = 262k.
            # 262k is small enough for Python if optimized?
            # Let's just iterate all masks.
            pass
        
        # Simple brute force with bitmasks
        # 18 vertices -> 2^18 = 262,144 masks.
        # Check each mask for triangle-free.
        # Precompute triangle masks?
        
        # List of triangles
        triangles = []
        for i in range(18):
            for j in range(i+1, 18):
                if (i,j) not in adj: continue
                for k in range(j+1, 18):
                    if (i,k) in adj and (j,k) in adj:
                        triangles.append((i,j,k))
        
        # For each mask, check if it contains any triangle.
        # To speed up, we can iterate masks and check triangles.
        # Or iterate triangles and mark invalid masks?
        # Number of triangles in K18 is 816. In our graph, likely less.
        
        valid_masks = [1] * (1 << 18)
        for t in triangles:
            i, j, k = t
            mask_t = (1 << i) | (1 << j) | (1 << k)
            # Any mask containing mask_t is invalid
            # This is tricky to update efficiently.
            
        # Let's just check each mask.
        # 262k masks * (check triangles)
        # Check: for each triangle, if mask & mask_t == mask_t: invalid.
        # If ~100 triangles, 262k * 100 = 26M ops. Feasible.
        
        max_size = 0
        for mask in range(1 << 18):
            # Check if mask is triangle-free
            is_tf = True
            for t in triangles:
                i, j, k = t
                if (mask & (1 << i)) and (mask & (1 << j)) and (mask & (1 << k)):
                    is_tf = False
                    break
            if is_tf:
                size = bin(mask).count('1')
                if size > max_size:
                    max_size = size
        return max_size

    # Local Search
    current_adj = edges_g
    current_score = max_triangle_free_size(current_adj)
    
    # Try to improve by adding edges
    # We can only add edges if it doesn't create K4.
    # And if it reduces the max triangle-free size.
    
    # Generate candidate edges to add
    # Edges not in current_adj
    all_edges = set((i,j) for i in range(18) for j in range(i+1, 18))
    candidates = list(all_edges - current_adj)
    random.shuffle(candidates)
    
    improved = True
    while improved:
        improved = False
        for e in candidates:
            u, v = e
            new_adj = set(current_adj)
            new_adj.add(e)
            
            if not is_k4_free(new_adj):
                continue
            
            new_score = max_triangle_free_size(new_adj)
            if new_score < current_score:
                current_adj = new_adj
                current_score = new_score
                improved = True
                # Update candidates?
                # We can just restart the loop or continue
                break
        
    return [list(e) for e in current_adj]
