import random
import time

def construct():
    n = 20
    # Bitmask representation of adjacency
    # adj[i] is a bitmask where bit j is 1 if (i,j) is an edge
    
    def get_common_neighbors_mask(u, v, w, adj):
        return adj[u] & adj[v] & adj[w]
    
    def is_valid_addition(u, v, adj):
        # Check if adding edge (u,v) creates a K3,3
        # A K3,3 is formed if there exists a triple {a,b,c} such that 
        # the number of common neighbors becomes >= 3.
        # Adding (u,v) only affects triples containing u or v.
        
        # Check triples containing u: {u, x, y}
        # The common neighbors are N(u) & N(x) & N(y).
        # Before adding (u,v), v is not in N(u).
        # After adding, if v is in N(x) and N(y), then v becomes a common neighbor.
        # So we need to ensure that for all x, y, if v in N(x) and v in N(y),
        # then the current number of common neighbors of {u,x,y} is <= 1.
        
        # Actually, simpler:
        # For any x, y != u, v:
        # Let C_old = |N(u) & N(x) & N(y)|
        # If v is connected to x and y, then v will be added to the common neighbors.
        # So new count = C_old + 1. We need C_old + 1 <= 2 => C_old <= 1.
        # If v is not connected to x or y, count remains C_old. We need C_old <= 2.
        
        # We can just iterate x, y.
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(n):
                if y == u or y == v or y == x:
                    continue
                
                # Check triple {u, x, y}
                # Current common neighbors mask
                mask = adj[u] & adj[x] & adj[y]
                count = bin(mask).count('1')
                
                if (adj[x] >> v) & 1 and (adj[y] >> v) & 1:
                    # v will become a common neighbor
                    if count > 1:
                        return False
                else:
                    if count > 2:
                        return False
                        
                # Check triple {v, x, y}
                mask_v = adj[v] & adj[x] & adj[y]
                count_v = bin(mask_v).count('1')
                
                if (adj[x] >> u) & 1 and (adj[y] >> u) & 1:
                    # u will become a common neighbor
                    if count_v > 1:
                        return False
                else:
                    if count_v > 2:
                        return False
        return True

    def find_violation(adj):
        # Find a triple with >= 3 common neighbors
        for u in range(n):
            for v in range(u+1, n):
                for w in range(v+1, n):
                    mask = adj[u] & adj[v] & adj[w]
                    if bin(mask).count('1') >= 3:
                        return (u, v, w, mask)
        return None

    def repair(adj):
        # Remove edges to fix violations
        while True:
            viol = find_violation(adj)
            if viol is None:
                break
            u, v, w, mask = viol
            # Get list of common neighbors
            neighbors = []
            for i in range(n):
                if (mask >> i) & 1:
                    neighbors.append(i)
            
            # Pick one neighbor to remove an edge from
            # We can remove (u, neighbor), (v, neighbor) or (w, neighbor)
            # Let's pick the first one
            target = neighbors[0]
            
            # Remove edge (u, target)
            adj[u] &= ~(1 << target)
            adj[target] &= ~(1 << u)

    def optimize(adj):
        # Try to add missing edges
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if not ((adj[u] >> v) & 1):
                        if is_valid_addition(u, v, adj):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            improved = True

    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_edges = 0
    best_adj = None
    
    # Try multiple random starts
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Initialize random graph
        # p=0.6 is a good starting point for n=20
        p = 0.6 + random.random() * 0.2
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Repair
        repair(adj)
        
        # Optimize
        optimize(adj)
        
        # Count edges
        edges = 0
        for i in range(n):
            edges += bin(adj[i]).count('1')
        edges //= 2
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
    # If no best_adj found (shouldn't happen), return seed
    if best_adj is None:
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    # Convert bitmask to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])
                
    return result
