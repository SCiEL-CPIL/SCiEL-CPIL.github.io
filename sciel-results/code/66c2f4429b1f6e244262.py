import random
import time
from itertools import combinations

def construct():
    n = 18
    # Initialize with seed graph K6,6,6
    # Parts: 0-5, 6-11, 12-17
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            part_u = u // 6
            part_v = v // 6
            if part_u != part_v:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    def get_triangles(adj):
        triangles = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        if w > v:
                            triangles.append((u, v, w))
                        c -= lsb
        return triangles

    def min_hitting_set_size(triangles):
        if not triangles:
            return 0
        
        # Remove duplicates
        triangles = list(set(triangles))
        
        # Sort triangles to improve B&B
        # Heuristic: triangles sharing more vertices might be harder?
        # Or just sort by first vertex?
        triangles.sort()
        
        best = n
        
        # Precompute for each vertex, which triangles it covers
        # Not strictly necessary for simple B&B, but helps with bounds
        
        def dfs(idx, current_size, mask):
            nonlocal best
            if current_size >= best:
                return
            
            if idx == len(triangles):
                if current_size < best:
                    best = current_size
                return
            
            # Lower bound: 
            # If we have R triangles left, and max cover is C, need at least ceil(R/C)
            # Simple bound: if current_size + 1 >= best, return? No.
            
            u, v, w = triangles[idx]
            
            # Try adding u
            if not (mask & (1 << u)):
                dfs(idx + 1, current_size + 1, mask | (1 << u))
            
            # Try adding v
            if not (mask & (1 << v)):
                dfs(idx + 1, current_size + 1, mask | (1 << v))
            
            # Try adding w
            if not (mask & (1 << w)):
                dfs(idx + 1, current_size + 1, mask | (1 << w))

        dfs(0, 0, 0)
        return best

    def score(adj):
        triangles = get_triangles(adj)
        hss = min_hitting_set_size(triangles)
        return n - hss

    # Simulated Annealing
    best_adj = adj[:]
    best_score = score(best_adj)
    current_adj = adj[:]
    current_score = best_score
    
    start_time = time.time()
    deadline = 110.0 # Leave margin
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    while time.time() < deadline:
        # Pick a random edge
        u, v = random.choice(all_edges)
        
        # Try flipping the edge
        if current_adj[u] & (1 << v):
            # Remove edge
            current_adj[u] ^= (1 << v)
            current_adj[v] ^= (1 << u)
            new_adj = current_adj[:]
            new_score = score(new_adj)
            
            if new_score < current_score or random.random() < pow(2.718, -(new_score - current_score) / T):
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_adj = current_adj[:]
                    best_score = current_score
            else:
                # Revert
                current_adj[u] ^= (1 << v)
                current_adj[v] ^= (1 << u)
        else:
            # Add edge, check K4
            # Check if N(u) & N(v) has an edge
            common = current_adj[u] & current_adj[v]
            has_k4 = False
            c = common
            while c:
                lsb = c & -c
                w = lsb.bit_length() - 1
                if current_adj[w] & (1 << (w + 1)): # Check if w connected to any other in common?
                    # Actually, we just need to check if there is ANY edge in common
                    # If there is an edge (x,y) in common, then u,v,x,y form K4
                    # So we just need to check if the induced subgraph on 'common' has any edge.
                    # Since 'common' is a bitmask, we can check if any pair in common is connected.
                    # Faster: if common != 0, check if any vertex in common is connected to another in common.
                    # Actually, if x, y in common, they are both neighbors of u and v.
                    # If x-y is an edge, then {u,v,x,y} is K4.
                    # So we need to check if the subgraph induced by 'common' has any edges.
                    pass
                c -= lsb
            
            # Efficient K4 check:
            # Iterate through bits in common
            c = common
            while c:
                lsb = c & -c
                w = lsb.bit_length() - 1
                # Check if w is connected to any other vertex in common
                # We can check if (common & current_adj[w]) != 0
                # But we must exclude w itself? No, adj[w] doesn't have w.
                # So if (common & current_adj[w]) != 0, there is an edge.
                if common & current_adj[w]:
                    has_k4 = True
                    break
                c -= lsb
            
            if has_k4:
                # Cannot add edge
                continue
            
            # Add edge
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            new_adj = current_adj[:]
            new_score = score(new_adj)
            
            if new_score < current_score or random.random() < pow(2.718, -(new_score - current_score) / T):
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_adj = current_adj[:]
                    best_score = current_score
            else:
                # Revert
                current_adj[u] ^= (1 << v)
                current_adj[v] ^= (1 << u)
        
        T *= cooling
        if T < T_min:
            T = T_min

    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    return edges
