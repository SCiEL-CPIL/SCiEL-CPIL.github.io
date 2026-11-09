import random

def construct():
    n = 20
    
    def is_valid_add(u, v, adj):
        # Check if adding (u, v) creates a K3,3
        # Condition: No triple has 3 common neighbors.
        # Adding (u, v) increases common neighbors for triples containing u or v.
        
        # Check triples {u, x, y}
        # Adding v to N(u).
        # If v is in N(x) & N(y), then common count for {u, x, y} increases by 1.
        # So we must ensure that for all x, y where v in N(x) & N(y),
        # current common count of {u, x, y} is < 2.
        
        for x in range(n):
            if x == u or x == v: continue
            for y in range(x+1, n):
                if y == u or y == v: continue
                # Check if v is common neighbor of x and y
                if (adj[x] & adj[y] & (1 << v)):
                    current_common = adj[u] & adj[x] & adj[y]
                    if current_common.bit_count() >= 2:
                        return False
        
        # Check triples {v, x, y}
        # Adding u to N(v).
        # If u is in N(x) & N(y), then common count for {v, x, y} increases by 1.
        for x in range(n):
            if x == u or x == v: continue
            for y in range(x+1, n):
                if y == u or y == v: continue
                if (adj[x] & adj[y] & (1 << u)):
                    current_common = adj[v] & adj[x] & adj[y]
                    if current_common.bit_count() >= 2:
                        return False
                        
        return True

    def find_violation(adj):
        # Returns (u, v, w, c) if violation found, else None
        # Violation: triple u, v, w with >= 3 common neighbors
        for u in range(n):
            for v in range(u+1, n):
                for w in range(v+1, n):
                    common = adj[u] & adj[v] & adj[w]
                    if common.bit_count() >= 3:
                        # Pick a random common neighbor to form the K3,3
                        # We need to return one edge to remove
                        # Let's pick a random c from common
                        c_list = [i for i in range(n) if (common >> i) & 1]
                        c = random.choice(c_list)
                        return (u, v, w, c)
        return None

    best_edges = 0
    best_adj = [0]*n
    
    # Try multiple random starts
    for _ in range(20): # 20 restarts
        adj = [0] * n
        
        # Initialize with a random dense graph (p=0.6)
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.6:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Peeling Phase: Remove edges to eliminate all K3,3
        # To avoid infinite loops, we limit iterations
        for _iter in range(1000):
            violation = find_violation(adj)
            if not violation:
                break
            u, v, w, c = violation
            # Remove edge (u, c)
            adj[u] &= ~(1 << c)
            adj[c] &= ~(1 << u)
        
        # Optimization Phase: Try to add missing edges
        for _pass in range(5):
            improved = False
            missing = []
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj[u] >> v) & 1:
                        missing.append((u, v))
            
            random.shuffle(missing)
            
            for u, v in missing:
                if is_valid_add(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
            
            if not improved:
                break
        
        # Count edges
        edge_count = sum(adj[i].bit_count() for i in range(n)) // 2
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = adj[:]
            
    # Convert best_adj to list of edges
    res = []
    for u in range(n):
        mask = best_adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            mask ^= lsb
            if u < v:
                res.append([u, v])
                
    return res
