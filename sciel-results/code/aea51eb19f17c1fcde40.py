import random
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    n = 20
    colors = 20
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_edges = []
    best_count = 0
    
    # Run multiple random restarts to find a dense valid graph
    while time.time() < deadline:
        random.shuffle(all_edges)
        # adj[u] is a dictionary {v: color}
        adj = {i: {} for i in range(n)}
        edges = []
        
        for u, v in all_edges:
            forbidden = set()
            valid = True
            
            # Check all 3-paths v-w-x-u
            # If any such path has non-distinct colours, we cannot add (u,v)
            # because the resulting 4-cycle u-v-w-x-u would not be rainbow.
            
            # Iterate w neighbors of v
            for w in list(adj[v].keys()):
                if w == u:
                    continue
                # Iterate x neighbors of w
                for x in list(adj[w].keys()):
                    if x == u or x == v:
                        continue
                    # Check if (x, u) exists
                    if u in adj[x]:
                        c1 = adj[v][w]
                        c2 = adj[w][x]
                        c3 = adj[x][u]
                        
                        # The colours of the existing 3 edges must be distinct
                        if c1 == c2 or c1 == c3 or c2 == c3:
                            valid = False
                            break
                            
                        forbidden.add(c1)
                        forbidden.add(c2)
                        forbidden.add(c3)
            
            if not valid:
                continue
                
            if len(forbidden) < colors:
                # Pick a colour not in forbidden
                available = [c for c in range(colors) if c not in forbidden]
                c = random.choice(available)
                adj[u][v] = c
                adj[v][u] = c
                edges.append([u, v, c])
                
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            
        # Optional: Early stop if we reach a very high density?
        # Max possible is 190. If we get close, we might stop.
        # But let's let it run to find the best.
        
    return best_edges
