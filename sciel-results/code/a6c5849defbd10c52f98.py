def construct():
    import itertools
    
    n = 18
    
    def build_adj(edges):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        return adj
    
    def is_k4_free(adj):
        for a in range(n):
            for b in range(a+1, n):
                if not adj[a][b]:
                    continue
                for c in range(b+1, n):
                    if not adj[a][c] or not adj[b][c]:
                        continue
                    for d in range(c+1, n):
                        if adj[a][d] and adj[b][d] and adj[c][d]:
                            return False
        return True
    
    def max_triangle_free(adj):
        # Use bitmasks for efficiency
        # Precompute for each vertex, which other vertices it's connected to
        # A set is triangle-free if no three vertices form a triangle
        
        # Simple approach: iterate over all subsets, check if triangle-free
        # 2^18 = 262144, for each check O(k^3) where k is subset size
        # Worst case k=18, 18^3 = 5832, total ~1.5e9 - too slow
        
        # Better: use the fact that if a subset contains a triangle, it's invalid
        # We can use DP or branch and bound
        
        best = 0
        vertices = list(range(n))
        
        def has_triangle_in_set(mask):
            # Check if the set of vertices in mask has a triangle
            verts = [i for i in range(n) if (mask >> i) & 1]
            for i in range(len(verts)):
                a = verts[i]
                for j in range(i+1, len(verts)):
                    b = verts[j]
                    if not adj[a][b]:
                        continue
                    for k in range(j+1, len(verts)):
                        c = verts[k]
                        if adj[a][c] and adj[b][c]:
                            return True
            return False
        
        # Branch and bound
        def backtrack(idx, mask, count):
            nonlocal best
            if count > best:
                best = count
            if idx == n:
                return
            # Prune
            if count + (n - idx) <= best:
                return
            
            # Option 1: don't include vertex idx
            backtrack(idx + 1, mask, count)
            
            # Option 2: include vertex idx
            new_mask = mask | (1 << idx)
            # Check if adding idx creates a triangle
            # Need to check if idx forms a triangle with any two vertices already in mask
            ok = True
            v = idx
            # Find vertices in mask that are adjacent to v
            candidates = []
            for u in range(n):
                if (mask >> u) & 1 and adj[v][u]:
                    candidates.append(u)
            for i in range(len(candidates)):
                for j in range(i+1, len(candidates)):
                    if adj[candidates[i]][candidates[j]]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                backtrack(idx + 1, new_mask, count + 1)
        
        backtrack(0, 0, 0)
        return best
    
    # Construction 1: K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges1 = []
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    edges1.append([a, b])
    adj1 = build_adj(edges1)
    assert is_k4_free(adj1)
    score1 = max_triangle_free(adj1)
    
    # Construction 2: Try different edge densities between parts
    # Let's try a random K4-free graph
    import random
    random.seed(42)
    
    best_edges = edges1
    best_score = score1
    
    for trial in range(20):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    edges.append([i, j])
        adj = build_adj(edges)
        if not is_k4_free(adj):
            continue
        score = max_triangle_free(adj)
        if score < best_score:
            best_score = score
            best_edges = edges
    
    return best_edges
