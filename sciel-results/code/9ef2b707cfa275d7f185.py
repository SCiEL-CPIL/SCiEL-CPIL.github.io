import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115
    n = 20
    rng = random.Random(42)
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def bit_count(x):
        return x.bit_count()
        
    # State
    adj = [0] * n
    # common_neighbors[x][y] = adj[x] & adj[y]
    # Use a 2D list or flat list. 20x20=400.
    cn = [[0]*n for _ in range(n)]
    
    def update_cn(x, y):
        cn[x][y] = adj[x] & adj[y]
        cn[y][x] = cn[x][y]
        
    def init_cn():
        for i in range(n):
            for j in range(n):
                update_cn(i, j)
                
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        # Update cn for all pairs involving u or v
        for i in range(n):
            if i != u and i != v:
                update_cn(u, i)
                update_cn(v, i)
            elif i == u:
                update_cn(u, v) # Just in case, though symmetric
            elif i == v:
                pass
        # Actually, simpler:
        for i in range(n):
            update_cn(u, i)
            update_cn(v, i)
            
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        for i in range(n):
            update_cn(u, i)
            update_cn(v, i)
            
    def is_valid_add(u, v):
        # Check triples containing u: {u, x, y}
        # x, y in N(u)
        # Condition: if v in C_xy, then |N(u) & C_xy| <= 1
        # N(u) & C_xy is adj[u] & cn[x][y]
        
        # We iterate over x in N(u)
        nu = adj[u]
        nv = adj[v]
        
        # Get neighbors of u
        # We can just iterate all i, j
        # Optimization: Precompute list of neighbors?
        # n=20, so simple loops are fine.
        
        for x in range(n):
            if x == u or x == v: continue
            if not ((nu >> x) & 1): continue
            
            cx = cn[x] # List of cn[x][y]
            for y in range(x+1, n):
                if y == u or y == v: continue
                if not ((nu >> y) & 1): continue
                
                # Check if v is in C_xy
                if ((cx[y] >> v) & 1):
                    # Check count
                    if bit_count(nu & cx[y]) > 1:
                        return False
        
        # Check triples containing v: {v, x, y}
        for x in range(n):
            if x == u or x == v: continue
            if not ((nv >> x) & 1): continue
            
            cx = cn[x]
            for y in range(x+1, n):
                if y == u or y == v: continue
                if not ((nv >> y) & 1): continue
                
                # Check if u is in C_xy
                if ((cx[y] >> u) & 1):
                    if bit_count(nv & cx[y]) > 1:
                        return False
                        
        return True

    def count_edges():
        return sum(bit_count(adj[i]) for i in range(n)) // 2
        
    best_edges = 0
    best_adj = None
    
    num_restarts = 40
    per_restart_time = 2.5
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Reset
        for i in range(n):
            adj[i] = 0
        init_cn()
        
        # Greedy Init
        edges_list = all_pairs[:]
        rng.shuffle(edges_list)
        for u, v in edges_list:
            if is_valid_add(u, v):
                add_edge(u, v)
                
        cur_edges = count_edges()
        
        # Local Search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try to add edges
            for u, v in edges_list:
                if not ((adj[u] >> v) & 1):
                    if is_valid_add(u, v):
                        add_edge(u, v)
                        cur_edges += 1
                        improved = True
            
            # If no edges added, try swaps
            if not improved:
                # Try random swaps
                swap_found = False
                for _ in range(500):
                    if time.time() > deadline - 5:
                        break
                    
                    # Pick random edge to remove
                    # Find a list of current edges
                    # This is O(n^2)
                    current_edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if (adj[i] >> j) & 1:
                                current_edges.append((i,j))
                                
                    if not current_edges:
                        break
                        
                    e1 = rng.choice(current_edges)
                    e2 = rng.choice(edges_list)
                    if e1 == e2 or e1 == (e2[1], e2[0]):
                        continue
                    if (adj[e2[0]] >> e2[1]) & 1:
                        continue
                        
                    # Try swap
                    remove_edge(e1[0], e1[1])
                    if is_valid_add(e2[0], e2[1]):
                        add_edge(e2[0], e2[1])
                        # Check if we can add more
                        add_ok = False
                        for u, v in edges_list:
                            if not ((adj[u] >> v) & 1):
                                if is_valid_add(u, v):
                                    add_edge(u, v)
                                    cur_edges += 1
                                    add_ok = True
                        if add_ok:
                            improved = True
                            swap_found = True
                            break
                        else:
                            # Revert
                            remove_edge(e2[0], e2[1])
                            add_edge(e1[0], e1[1])
                            
                if not swap_found:
                    break
                    
        if cur_edges > best_edges:
            best_edges = cur_edges
            best_adj = [a for a in adj]
            
    if best_adj is None:
        return []
        
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
