import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Try n from 47 down to 39
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
            
        min_deg = n - 39
        if min_deg < 0:
            min_deg = 0
        
        # For n=39, min_deg=0, empty graph works
        if min_deg == 0:
            return {'n': n, 'edges': []}
        
        # Try multiple restarts for this n
        # With optimized check, we can afford more restarts or longer runs per restart
        max_restarts = 1000
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Bitmask adjacency
            adj = [0] * n
            # reach2[u] is the set of vertices at distance exactly 2 from u? 
            # No, it's the union of neighbors of neighbors.
            # If x in reach2[u], there is a path u-w-x.
            reach2 = [0] * n
            degrees = [0] * n
            edges = []
            
            # Helper to get list of neighbors from bitmask
            def get_neighbors(mask):
                neighbors = []
                m = mask
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    neighbors.append(w)
                    m ^= lsb
                return neighbors

            # Greedy construction
            # We will try to add edges until we can't or time runs out
            # Prioritize vertices with lowest degree
            
            added_any = True
            while added_any and time.time() < deadline:
                added_any = False
                
                # Find vertices below target degree
                low_verts = [i for i in range(n) if degrees[i] < min_deg]
                
                if not low_verts:
                    break
                
                # Randomly pick a low-degree vertex
                u = random.choice(low_verts)
                
                # Try random partners
                # We need to find v such that reach2[u] & adj[v] == 0
                # And v not in adj[u]
                
                # To speed up, we can iterate through all v and check
                # But shuffling helps avoid bias
                candidates = [v for v in range(n) if v != u and not ((adj[u] >> v) & 1)]
                random.shuffle(candidates)
                
                for v in candidates:
                    # Check C4 condition
                    # Path u-w-x-v exists if there is w in N(u) such that x in N(v) and x in N(w)
                    # This is equivalent to: exists w in N(u) such that N(w) intersects N(v)
                    # Which is equivalent to: (Union of N(w) for w in N(u)) intersects N(v)
                    # Union of N(w) for w in N(u) is reach2[u]
                    if (reach2[u] & adj[v]) == 0:
                        # Safe to add
                        # Update adj
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        degrees[u] += 1
                        degrees[v] += 1
                        edges.append([u, v])
                        added_any = True
                        
                        # Update reach2
                        # 1. reach2[u] gains N(v)
                        reach2[u] |= adj[v]
                        # 2. reach2[v] gains N(u)
                        reach2[v] |= adj[u]
                        
                        # 3. For all w in OLD N(u), reach2[w] gains v
                        # We need the neighbors of u BEFORE adding v
                        # Let's retrieve them
                        # Note: adj[u] now includes v.
                        # Old neighbors are adj[u] without v.
                        old_nu = adj[u] & ~(1 << v)
                        # But we also need to update reach2 for w in old_nu
                        m = old_nu
                        while m:
                            lsb = m & -m
                            w = lsb.bit_length() - 1
                            reach2[w] |= (1 << v)
                            m ^= lsb
                            
                        # 4. For all x in OLD N(v), reach2[x] gains u
                        old_nv = adj[v] & ~(1 << u)
                        m = old_nv
                        while m:
                            lsb = m & -m
                            x = lsb.bit_length() - 1
                            reach2[x] |= (1 << u)
                            m ^= lsb
                            
                        break
            
            # Check if we met the requirement
            if min(degrees) >= min_deg:
                return {'n': n, 'edges': edges}
        
        # If we exhausted restarts for this n, try smaller n
    
    # Fallback: return the smallest valid graph
    # For n=39, min_deg=0, empty graph works
    return {'n': 39, 'edges': []}
