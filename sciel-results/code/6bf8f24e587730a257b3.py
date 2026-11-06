import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = 118  # Safety margin for 120s limit

    def time_check():
        return time.time() - start_time > deadline

    best_edge_count = 0
    best_edges = None

    # Pre-compute all possible edges
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))

    # We will try multiple restarts
    for seed in range(500): # Try many restarts
        if time_check():
            break
            
        random.seed(seed)
        
        # Initialize with 8 disjoint K5s
        adj = [0] * n
        edges = []
        
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    u = base + a
                    v = base + b
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
        
        # Get non-edges
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    non_edges.append((u, v))
        
        random.shuffle(non_edges)
        
        def would_create_c6(u, v):
            """
            Check if adding edge (u,v) creates a C6.
            C6: u - w - z - y - x - v
            w in N(u), x in N(v).
            Path w-z-y-x of length 3 in old graph.
            """
            nu = adj[u]
            nv = adj[v]
            
            # Optimize by iterating smaller neighborhood first?
            # But we need w in N(u) and x in N(v) for the path w-z-y-x.
            # The structure is symmetric in w and x?
            # Path w-z-y-x.
            # If we swap u and v, we get x-z-y-w. Same path.
            # So we can iterate w in N(u) and x in N(v).
            
            # To speed up, we can check if there is any common neighbor structure.
            
            # Iterate w in N(u)
            w_mask = nu
            while w_mask:
                wb = w_mask & -w_mask
                wi = wb.bit_length() - 1
                w_mask ^= wb
                
                # Iterate x in N(v)
                x_mask = nv
                while x_mask:
                    xb = x_mask & -x_mask
                    xi = xb.bit_length() - 1
                    x_mask ^= xb
                    
                    if wi == xi:
                        continue
                    
                    # Check path w-z-y-x
                    # z in N(wi) \ {xi}
                    nw = adj[wi]
                    # z cannot be xi
                    nw_minus_x = nw & ~(1 << xi)
                    
                    z_mask = nw_minus_x
                    while z_mask:
                        zb = z_mask & -z_mask
                        zi = zb.bit_length() - 1
                        z_mask ^= zb
                        
                        # y in N(zi) & N(xi) \ {wi}
                        # y must be common neighbor of zi and xi
                        common = adj[zi] & adj[xi]
                        # y cannot be wi
                        common_minus_w = common & ~(1 << wi)
                        
                        if common_minus_w:
                            return True
            return False

        added = 0
        for u, v in non_edges:
            if time_check():
                break
            if not would_create_c6(u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
                added += 1
        
        if len(edges) > best_edge_count:
            best_edge_count = len(edges)
            best_edges = [list(e) for e in edges]
            
            # If we hit 105, we can stop early
            if best_edge_count >= 105:
                break

    return best_edges
