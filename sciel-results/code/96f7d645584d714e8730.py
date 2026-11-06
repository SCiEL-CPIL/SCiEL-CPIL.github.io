import time
import random

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    n = 20
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    
    # common[u][v] stores the number of common neighbors for pair (u, v)
    common = [[0] * n for _ in range(n)]
    
    def popcount(x):
        return bin(x).count('1')

    def add_edge(u, v):
        nonlocal adj, common
        if u == v:
            return False
        if adj[u] & (1 << v):
            return False
        
        # Check validity
        # For x in N(v), common[u][x] must be < 2
        # For y in N(u), common[v][y] must be < 2
        
        # We iterate through bits of adj[v] to find neighbors of v
        mask_v = adj[v]
        mask_u = adj[u]
        
        # Check neighbors of v
        m = mask_v
        while m:
            lsb = m & -m
            x = lsb.bit_length() - 1
            if x != u:
                if common[u][x] >= 2:
                    return False
            m ^= lsb
            
        # Check neighbors of u
        m = mask_u
        while m:
            lsb = m & -m
            y = lsb.bit_length() - 1
            if y != v:
                if common[v][y] >= 2:
                    return False
            m ^= lsb
            
        # If valid, apply
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Update common counts
        # For x in N(v) (old), common[u][x] += 1
        m = mask_v
        while m:
            lsb = m & -m
            x = lsb.bit_length() - 1
            if x != u:
                common[u][x] += 1
                common[x][u] += 1
            m ^= lsb
            
        # For y in N(u) (old), common[v][y] += 1
        m = mask_u
        while m:
            lsb = m & -m
            y = lsb.bit_length() - 1
            if y != v:
                common[v][y] += 1
                common[y][v] += 1
            m ^= lsb
            
        return True

    def remove_edge(u, v):
        nonlocal adj, common
        if not (adj[u] & (1 << v)):
            return False
        
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        # Update common counts (reverse)
        # For x in N(v) (new, which is old N(v) minus u? No, u was not in N(v) before removal? 
        # Wait. Before removal, u was in N(v). 
        # The common counts were updated when u was added to N(v).
        # So we reverse that.
        # When u was added to N(v), for all x in N(v) (excluding u), common[u][x] was incremented.
        # So now, for all x in N(v) (current, which excludes u), common[u][x] should be decremented.
        
        m = adj[v]
        while m:
            lsb = m & -m
            x = lsb.bit_length() - 1
            # x is in N(v) currently. Since u was removed, x != u is guaranteed if graph is simple?
            # Yes, u is no longer neighbor of v.
            common[u][x] -= 1
            common[x][u] -= 1
            m ^= lsb
            
        m = adj[u]
        while m:
            lsb = m & -m
            y = lsb.bit_length() - 1
            common[v][y] -= 1
            common[y][v] -= 1
            m ^= lsb
            
        return True

    def get_edge_count():
        return sum(popcount(x) for x in adj) // 2

    def polish(adj, common):
        """Greedily add edges until no more can be added."""
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        if add_edge(u, v):
                            improved = True
        return get_edge_count()

    # Initialize best
    best_edges = 0
    best_adj = [0] * n
    best_common = [[0] * n for _ in range(n)]
    
    random.seed(42)
    
    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Main loop
    while time.time() - start_time < deadline:
        # 1. Initialize a random valid graph
        adj = [0] * n
        common = [[0] * n for _ in range(n)]
        
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            add_edge(u, v)
            
        # 2. Polish
        current_edges = polish(adj, common)
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]
            best_common = [row[:] for row in common]
            
        # 3. Local Search with Swaps
        # Try to improve further via 2-swaps
        # If we get stuck, restart with a different random seed or perturbation
        
        # To save time, we only do extensive local search if we are close to the best
        # or if we have time.
        
        # Let's do a limited number of swap attempts
        # We will try to find a swap that allows adding more edges after polishing.
        
        # To make this efficient, we'll pick random edges to remove.
        # Current edges list
        current_edge_list = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    current_edge_list.append((u, v))
                    
        # We need to be careful with time. 
        # If we do too many swaps, we might timeout.
        
        # Simple strategy: Try 200 random 2-swaps.
        # If a swap leads to a state that, after polishing, has more edges, accept.
        
        for _ in range(200):
            if time.time() - start_time > deadline:
                break
                
            if len(current_edge_list) < 2:
                break
                
            e1, e2 = random.sample(current_edge_list, 2)
            u1, v1 = e1
            u2, v2 = e2
            
            # Check if disjoint
            if len({u1, v1, u2, v2}) < 4:
                continue
                
            # Try swap: remove (u1,v1), (u2,v2), add (u1,u2), (v1,v2)
            # Or (u1,v2), (v1,u2)
            # We'll try one combination
            
            new_u1, new_v1 = u1, u2
            new_u2, new_v2 = v1, v2
            
            # Check if new edges exist
            if (adj[new_u1] & (1 << new_v1)) or (adj[new_u2] & (1 << new_v2)):
                continue
                
            # Backup
            old_adj = adj[:]
            old_common = [row[:] for row in common]
            
            # Remove
            remove_edge(u1, v1)
            remove_edge(u2, v2)
            
            # Add
            ok1 = add_edge(new_u1, new_v1)
            ok2 = add_edge(new_u2, new_v2) if ok1 else False
            
            if ok1 and ok2:
                # Valid swap
                # Polish
                new_edges = polish(adj, common)
                if new_edges > current_edges:
                    current_edges = new_edges
                    # Update current edge list
                    current_edge_list = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if adj[u] & (1 << v):
                                current_edge_list.append((u, v))
                    
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                        best_common = [row[:] for row in common]
                else:
                    # Restore if not improved (or even if not, to keep search consistent?)
                    # Actually, if we don't improve, we might still want to explore.
                    # But for simplicity and speed, only accept if it leads to immediate improvement after polish.
                    adj = old_adj
                    common = old_common
            else:
                adj = old_adj
                common = old_common

    # Return best valid graph
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
