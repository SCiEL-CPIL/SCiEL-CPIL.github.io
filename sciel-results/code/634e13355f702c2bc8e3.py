"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def construct():
    n = 20
    num_colors = 20
    # State: adj[u][v] is color (0-19) or -1 if no edge
    # Using NumPy for faster indexing and operations
    adj = np.full((n, n), -1, dtype=np.int8)
    
    def is_rainbow(c1, c2, c3, c4):
        return c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4

    def check_add(u, v, c):
        """Check if adding (u,v) with color c creates any non-rainbow C4."""
        # Find w such that edge (v,w) exists
        # We need to check all 4-cycles u-v-w-x-u
        # Edges: (u,v) new, (v,w), (w,x), (x,u)
        
        # Get neighbors of v (excluding u)
        # adj[v] contains colors, -1 for no edge
        v_neighbors = np.where(adj[v] != -1)[0]
        # Filter out u
        v_neighbors = [w for w in v_neighbors if w != u]
        
        for w in v_neighbors:
            c_vw = adj[v, w]
            # Get neighbors of w (excluding u, v)
            w_neighbors = np.where(adj[w] != -1)[0]
            w_neighbors = [x for x in w_neighbors if x != u and x != v]
            
            # Get neighbors of u (excluding v, w)
            u_neighbors = np.where(adj[u] != -1)[0]
            u_neighbors = [x for x in u_neighbors if x != v and x != w]
            
            # Intersection of w_neighbors and u_neighbors gives potential x
            # For efficiency with small n, we can just iterate w_neighbors and check if in u_neighbors
            # Or use set intersection
            u_neigh_set = set(u_neighbors)
            
            for x in w_neighbors:
                if x in u_neigh_set:
                    c_wx = adj[w, x]
                    c_xu = adj[u, x]
                    
                    # Check if colors are distinct
                    if not is_rainbow(c, c_vw, c_wx, c_xu):
                        return False
        return True

    def check_change(u, v, new_c):
        """Check if changing color of (u,v) to new_c results in 0 violations for C4s involving (u,v)."""
        # Same logic as check_add, but (u,v) is already in graph
        # We just replace the color in the check
        return check_add(u, v, new_c)

    start_time = time.time()
    time_limit = 115.0
    
    best_edges = 0
    best_adj = adj.copy()
    
    # Strategy: Greedy addition with random restarts and local search
    num_restarts = 50
    steps_per_restart = 2000
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Reset
        adj = np.full((n, n), -1, dtype=np.int8)
        current_edges = 0
        
        # List of all possible edges
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_edges)
        
        # Phase 1: Greedy Addition
        # Try to add each edge if possible
        for u, v in all_edges:
            if time.time() - start_time > time_limit:
                break
            if adj[u, v] != -1:
                continue
            
            # Try colors in random order
            colors = list(range(num_colors))
            random.shuffle(colors)
            
            added = False
            for c in colors:
                if check_add(u, v, c):
                    adj[u, v] = c
                    adj[v, u] = c
                    current_edges += 1
                    added = True
                    break
        
        # Phase 2: Local Search / Color Swapping
        # If we are stuck, try to change colors to allow more edges
        # This is tricky. Instead, let's try a "shake" if we hit a local max
        
        # Update best
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj.copy()
            
        # If we didn't fill the graph, try a second pass with different randomization?
        # Or just restart.
        
    coloured_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u, v] != -1:
                coloured_edges.append([u, v, int(best_adj[u, v])])
    
    return coloured_edges
# EVOLVE-BLOCK-END
