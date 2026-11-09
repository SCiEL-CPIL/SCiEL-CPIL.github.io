import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # We want to maximize n.
    # Contract says 1 <= n <= 47.
    
    for n in range(47, 38, -1):
        min_deg_needed = n - 39
        
        # Try multiple random restarts for this n
        # If n=47, we are happy to return as soon as we find one.
        max_attempts = 2000 if n == 47 else 500
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Initialize empty graph with bitmasks
            adj = [0] * n
            
            # Create list of all possible edges
            edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(edges)
            
            # Greedy addition
            for u, v in edges:
                # Check if adding (u,v) creates a C4
                # Condition: exists x in adj[u] such that (adj[x] & adj[v]) != 0
                
                u_mask = adj[u]
                v_mask = adj[v]
                
                # Iterate over set bits in u_mask
                temp = u_mask
                c4_found = False
                while temp:
                    lsb = temp & -temp
                    x = lsb.bit_length() - 1
                    if adj[x] & v_mask:
                        c4_found = True
                        break
                    temp -= lsb
                
                if c4_found:
                    continue
                
                # Safe to add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            # Check min degree
            # Bit count of each mask
            min_deg = min(bin(mask).count('1') for mask in adj)
            
            if min_deg >= min_deg_needed:
                # Valid graph found
                # Convert adj to edge list
                result_edges = []
                for i in range(n):
                    mask = adj[i]
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        if i < j:
                            result_edges.append([i, j])
                        mask -= lsb
                
                return {'n': n, 'edges': result_edges}
            
        # If we exhausted attempts for this n, try smaller n
        if n == 39:
            return {'n': 39, 'edges': []}

    # Fallback
    return {'n': 39, 'edges': []}
