import random
import time

def construct():
    n = 20
    num_colors = 20
    
    # Fixed seed for reproducibility
    rng = random.Random(42)
    
    # adj[u][v] stores color, -1 if no edge
    adj = [[-1 for _ in range(n)] for _ in range(n)]
    
    # Generate all possible edges
    edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Shuffle to explore different graph structures
    rng.shuffle(edges)
    
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    for u, v in edges:
        if time.time() - start_time > deadline:
            break
            
        forbidden = set()
        can_add = True
        
        # Check all potential 4-cycles involving (u, v)
        # A 4-cycle is u-a-b-v-u
        # a must be neighbor of u, b must be neighbor of v, and (a,b) must be an edge
        # Also a != b to ensure it's a simple 4-cycle (distinct vertices)
        
        for a in range(n):
            if a == u or a == v:
                continue
            if adj[u][a] == -1:
                continue
                
            for b in range(n):
                if b == u or b == v or b == a:
                    continue
                if adj[v][b] == -1:
                    continue
                if adj[a][b] == -1:
                    continue
                
                # Path u-a-b-v exists
                c_ua = adj[u][a]
                c_ab = adj[a][b]
                c_bv = adj[v][b]
                
                # If the existing 3 edges have duplicate colors, 
                # no color for (u,v) can make the 4-cycle rainbow.
                if c_ua == c_ab or c_ab == c_bv or c_ua == c_bv:
                    can_add = False
                    break
                
                # Otherwise, these colors are forbidden for (u,v)
                forbidden.add(c_ua)
                forbidden.add(c_ab)
                forbidden.add(c_bv)
                
            if not can_add:
                break
        
        if can_add:
            # Find a valid color
            chosen_color = -1
            for c in range(num_colors):
                if c not in forbidden:
                    chosen_color = c
                    break
            
            if chosen_color != -1:
                adj[u][v] = chosen_color
                adj[v][u] = chosen_color

    # Build output
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v] != -1:
                result.append([u, v, adj[u][v]])
                
    return result
