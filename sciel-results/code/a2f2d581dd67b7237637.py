import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n = 40
    
    best_edges = 0
    best_adj = [0] * n
    
    def creates_c6(u, v, adj):
        mask_u = adj[u]
        mask_v = adj[v]
        
        x_mask = mask_u
        while x_mask:
            x_bit = x_mask & -x_mask
            x = x_bit.bit_length() - 1
            x_mask ^= x_bit
            
            y_mask = mask_v
            while y_mask:
                y_bit = y_mask & -y_bit
                y = y_bit.bit_length() - 1
                y_mask ^= y_bit
                
                if x == y:
                    continue
                
                z_mask = adj[x] & adj[y]
                forbidden = (1 << u) | (1 << v) | (1 << x) | (1 << y)
                z_mask &= ~forbidden
                
                if z_mask:
                    return True
        return False

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # Pre-generate list of all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Main loop
    while time.time() < deadline:
        adj = [0] * n
        current_edges = 0
        
        # Randomize the order of edges to try
        random.shuffle(all_edges)
        
        # Greedy addition
        for u, v in all_edges:
            if not (adj[u] & (1 << v)):
                if not creates_c6(u, v, adj):
                    add_edge(adj, u, v)
                    current_edges += 1
        
        # If we reached a local optimum, perturb
        if time.time() < deadline:
            # Get a list of edges to pick one to remove
            edges_list = []
            for u in range(n):
                mask = adj[u]
                while mask:
                    v_bit = mask & -mask
                    v = v_bit.bit_length() - 1
                    if u < v:
                        edges_list.append((u, v))
                    mask ^= v_bit
            
            if edges_list:
                # Remove 1-3 random edges to escape local optimum
                num_remove = random.randint(1, 3)
                for _ in range(num_remove):
                    if not edges_list:
                        break
                    u, v = random.choice(edges_list)
                    remove_edge(adj, u, v)
                    edges_list.remove((u, v))
                    current_edges -= 1
        
        # Update best
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]

    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        mask = best_adj[u]
        while mask:
            v_bit = mask & -mask
            v = v_bit.bit_length() - 1
            if u < v:
                edges.append([u, v])
            mask ^= v_bit
            
    return edges
