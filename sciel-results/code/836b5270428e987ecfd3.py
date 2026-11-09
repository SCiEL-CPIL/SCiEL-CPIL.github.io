"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    n = 20
    # State: adj[u][v] is color (0-19) or -1 if no edge
    adj = [[-1 for _ in range(n)] for _ in range(n)]
    
    def get_c4s_involving(u, v):
        """
        Yield (w, x) such that u-v-w-x-u is a 4-cycle in the current graph.
        Assumes (u,v) might or might not be in graph, but the other 3 edges must be.
        """
        for w in range(n):
            if w == u or w == v: continue
            if adj[v][w] == -1: continue
            for x in range(n):
                if x == u or x == v or x == w: continue
                if adj[u][x] == -1: continue
                if adj[w][x] != -1:
                    yield (w, x)

    def is_rainbow(c1, c2, c3, c4):
        return len({c1, c2, c3, c4}) == 4

    def check_add(u, v, c):
        """Check if adding (u,v) with color c creates any non-rainbow C4."""
        for w, x in get_c4s_involving(u, v):
            if not is_rainbow(c, adj[v][w], adj[w][x], adj[x][u]):
                return False
        return True

    def check_change(u, v, new_c):
        """Check if changing color of (u,v) to new_c results in 0 violations for C4s involving (u,v)."""
        for w, x in get_c4s_involving(u, v):
            if not is_rainbow(new_c, adj[v][w], adj[w][x], adj[x][u]):
                return False
        return True

    start_time = time.time()
    time_limit = 115.0
    
    best_edges = 0
    best_adj = [row[:] for row in adj]
    
    # Random Restart Hill Climbing
    num_restarts = 200
    steps_per_restart = 1000
    
    for _ in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Reset
        adj = [[-1 for _ in range(n)] for _ in range(n)]
        current_edges = 0
        
        for _ in range(steps_per_restart):
            if time.time() - start_time > time_limit:
                break
            
            # Try to add an edge
            # Pick random pair
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if adj[u][v] != -1: continue
            
            # Try random colors
            found = False
            for _ in range(10):
                c = random.randint(0, 19)
                if check_add(u, v, c):
                    adj[u][v] = c
                    adj[v][u] = c
                    current_edges += 1
                    found = True
                    break
            
            if not found:
                # Try to change color of an existing edge to allow more additions?
                if current_edges > 0:
                    for _ in range(5):
                        ru = random.randint(0, n-1)
                        rv = random.randint(0, n-1)
                        if ru == rv or adj[ru][rv] == -1: continue
                        
                        old_c = adj[ru][rv]
                        new_c = random.randint(0, 19)
                        if new_c == old_c: continue
                        
                        if check_change(ru, rv, new_c):
                            adj[ru][rv] = new_c
                            adj[rv][ru] = new_c
                            break

            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = [row[:] for row in adj]
                
        # End restart

    coloured_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v] != -1:
                coloured_edges.append([u, v, best_adj[u][v]])
    
    return coloured_edges
# EVOLVE-BLOCK-END
