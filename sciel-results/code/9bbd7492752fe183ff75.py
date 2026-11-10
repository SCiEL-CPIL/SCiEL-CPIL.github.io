import random
import time

def construct():
    n = 20
    best_edges = []
    best_count = 0
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    for seed in range(1000):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        # Initialize with a random graph
        # Using a list of bitmasks
        adj = [0] * n
        
        # Random initialization
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Function to check if edge (u, v) can be added
        def can_add(u, v):
            # u, v are not connected
            # Check triples containing u
            # For all x, y != u, v
            # If v in N(x) and v in N(y), then v becomes common neighbor of {u, x, y}
            # We need |N(u) & N(x) & N(y)| <= 1
            # Note: N(u) does NOT contain v yet.
            
            # Optimization: Precompute neighbors?
            # Just iterate.
            
            for x in range(n):
                if x == u or x == v: continue
                nx = adj[x]
                for y in range(x+1, n):
                    if y == u or y == v: continue
                    ny = adj[y]
                    
                    # Check triple {u, x, y}
                    # v is in N(x) and N(y)?
                    if (nx & (1 << v)) and (ny & (1 << v)):
                        # v will be a common neighbor
                        # Count current common neighbors of u, x, y
                        common = adj[u] & nx & ny
                        # If count >= 2, adding v makes it >= 3
                        if common.bit_count() >= 2:
                            return False
                    
                    # Check triple {v, x, y}
                    # u is in N(x) and N(y)?
                    if (nx & (1 << u)) and (ny & (1 << u)):
                        common = adj[v] & nx & ny
                        if common.bit_count() >= 2:
                            return False
            return True

        # Function to count edges
        def count_edges():
            c = 0
            for i in range(n):
                c += adj[i].bit_count()
            return c // 2

        # Function to get edges list
        def get_edges():
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            return edges

        # Clean phase: Remove edges that cause violations
        # We need to find a violating triple and remove an edge.
        # A violation is a triple {i, j, k} with >= 3 common neighbors.
        
        changed = True
        while changed:
            changed = False
            # Find a violation
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        common = adj[i] & adj[j] & adj[k]
                        if common.bit_count() >= 3:
                            # Remove an edge
                            # Pick a common neighbor v
                            # and a vertex from {i, j, k}, say i
                            # Remove (i, v)
                            
                            # Find first set bit in common
                            v = (common & -common).bit_length() - 1
                            
                            # Remove edge (i, v)
                            adj[i] &= ~(1 << v)
                            adj[v] &= ~(1 << i)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
        
        # Grow phase: Add all valid edges
        added = True
        while added:
            added = False
            for i in range(n):
                for j in range(i+1, n):
                    if not (adj[i] & (1 << j)):
                        if can_add(i, j):
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                            added = True
        
        current_count = count_edges()
        if current_count > best_count:
            best_count = current_count
            best_edges = get_edges()
            
    return best_edges
