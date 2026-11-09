import random
import time

def construct():
    n = 18
    deadline = time.time() + 145.0
    
    def calc_score(adj):
        # adj: list of sets
        # Find largest triangle-free induced subgraph
        # Precompute triangles
        triangles = []
        for i in range(n):
            for j in range(i + 1, n):
                if j in adj[i]:
                    for k in range(j + 1, n):
                        if k in adj[i] and k in adj[j]:
                            triangles.append((i, j, k))
        
        # If no triangles, score is n
        if not triangles:
            return n
            
        # Check masks from largest to smallest
        # We can prune: if a mask is not triangle-free, its supersets are not.
        # But we are going down.
        # If a mask IS triangle-free, we return its size.
        
        # Optimization: 
        # Instead of checking all masks, we can use a branch and bound on the vertices.
        # But 2^n is small.
        
        for mask in range((1 << n) - 1, 0, -1):
            size = bin(mask).count('1')
            if size <= 1:
                return size
            
            is_tf = True
            for (i, j, k) in triangles:
                if (mask & (1 << i)) and (mask & (1 << j)) and (mask & (1 << k)):
                    is_tf = False
                    break
            
            if is_tf:
                return size
        
        return 1

    def can_add_edge(adj, u, v):
        if u == v or v in adj[u]:
            return False
        common = adj[u] & adj[v]
        # Check if common contains a triangle
        # For each w in common, check if it has a neighbor in common
        for w in common:
            for x in common:
                if w != x and x in adj[w]:
                    return False
        return True

    best_score = 18
    best_adj = None
    
    # Start with K6,6,6
    adj = [set() for _ in range(n)]
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    adj[u].add(v)
                    adj[v].add(u)
    
    score = calc_score(adj)
    best_score = score
    best_adj = [set(s) for s in adj]
    
    # Local Search
    while time.time() < deadline:
        adj = [set(s) for s in best_adj]
        improved = False
        # Try adding edges
        for _ in range(200):
            if time.time() > deadline: break
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if not can_add_edge(adj, u, v):
                continue
            
            adj[u].add(v)
            adj[v].add(u)
            
            new_score = calc_score(adj)
            if new_score < best_score:
                best_score = new_score
                best_adj = [set(s) for s in adj]
                improved = True
            else:
                adj[u].remove(v)
                adj[v].remove(u)
                
        if not improved:
            # Try removing edges? No, removing edges increases triangle-free sets.
            # We want to ADD edges to create triangles.
            # If we are stuck, maybe we need a different start?
            # For now, just keep trying.
            
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if j in best_adj[i]:
                edges.append([i, j])
    return edges
