import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    def try_construct(n, min_deg):
        # Returns edges list if successful, else None
        # Use bitmask representation
        # adj[i] is int bitmask of neighbors
        # dist2[i] is int bitmask of vertices at distance 2
        
        max_iter = 50000
        for _ in range(max_iter):
            if time.time() > deadline:
                return None
            
            adj = [0] * n
            dist2 = [0] * n
            
            # Greedy/Randomized construction
            # We want to add edges.
            # Candidate edges are pairs (u, v) such that:
            # 1. (u, v) not in adj
            # 2. adj[u] & dist2[v] == 0 (no path of length 3)
            
            # To speed up, we can maintain a list of candidates?
            # Or just scan all pairs. n=46, pairs ~1000.
            # Scanning 1000 pairs * 50000 iters = 50M ops. Might be slow in Python.
            # We need to be smarter or limit iterations.
            
            # Let's try a simpler heuristic:
            # Start with random edges, then fix C4s?
            # Or start empty and add valid edges.
            
            # Optimized scan:
            # We can precompute valid edges? No, graph changes.
            
            # Let's limit iterations to 10000 and use random restarts.
            pass

        # Implementation of a single attempt
        def single_attempt():
            adj = [0] * n
            dist2 = [0] * n
            deg = [0] * n
            
            # Randomly initialize with some edges?
            # Or start empty.
            
            # We will try to add edges until no more valid edges can be added.
            # Then check if min(deg) >= min_deg.
            
            # To avoid getting stuck with low degree, we can prioritize adding edges
            # to vertices with low degree.
            
            # Simple loop:
            # While True:
            #   Find all valid (u, v)
            #   If none, break
            #   Pick one (randomly from valid, or weighted by degree deficit)
            #   Add it
            
            # To make it faster, we can just pick a random valid edge.
            
            while True:
                if time.time() > deadline:
                    return None
                
                valid_edges = []
                # Scan all pairs
                for u in range(n):
                    # Only consider v > u to avoid duplicates? 
                    # Bitmask check is symmetric.
                    # We need to check if adj[u] & dist2[v] == 0.
                    # Note: dist2[v] includes vertices at distance 2 from v.
                    # If u is at distance 2 from v, then u is in dist2[v].
                    # So adj[u] & dist2[v] != 0 if they share a neighbor?
                    # No. dist2[v] = {x | dist(v,x)=2}.
                    # If u in dist2[v], then there is w s.t. v-w-u.
                    # Then adj[u] contains w? No, adj[u] contains neighbors of u.
                    # The condition adj[u] & dist2[v] == 0 checks if N(u) intersects N(N(v)).
                    # If they intersect at x, then u-x-w-v is a path of length 3.
                    
                    # We also need to ensure (u,v) is not already an edge.
                    # if (adj[u] >> v) & 1: continue
                    
                    # Optimization: 
                    # If deg[u] is already high, maybe skip? No, we need to fill gaps.
                    
                    for v in range(u + 1, n):
                        if (adj[u] >> v) & 1:
                            continue
                        # Check C4 condition
                        if (adj[u] & dist2[v]) == 0:
                            valid_edges.append((u, v))
                
                if not valid_edges:
                    break
                
                # Pick an edge
                # Heuristic: Pick edge connecting lowest degree vertices?
                # Or random?
                # Random is simpler and often works well with restarts.
                u, v = random.choice(valid_edges)
                
                # Add edge (u, v)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                
                # Update dist2
                # dist2[u] |= adj[v]
                # dist2[v] |= adj[u]
                # For x in N(u) \ {v}: dist2[x] |= 1 << v
                # For x in N(v) \ {u}: dist2[x] |= 1 << u
                
                # Update dist2[u] and dist2[v] first
                dist2[u] |= adj[v]
                dist2[v] |= adj[u]
                
                # Update neighbors of u
                # Iterate through bits in adj[u] (excluding v)
                mask_u = adj[u] & ~(1 << v)
                while mask_u:
                    lsb = mask_u & -mask_u
                    x = lsb.bit_length() - 1
                    dist2[x] |= (1 << v)
                    mask_u ^= lsb
                    
                # Update neighbors of v
                mask_v = adj[v] & ~(1 << u)
                while mask_v:
                    lsb = mask_v & -mask_v
                    x = lsb.bit_length() - 1
                    dist2[x] |= (1 << u)
                    mask_v ^= lsb
            
            # Check min degree
            if min(deg) >= min_deg:
                # Convert to edge list
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj[u] >> v) & 1:
                            edges.append([u, v])
                return edges
            else:
                return None
        
        # Try multiple times
        attempts = 0
        while time.time() < deadline - 10: # Leave 10s margin
            res = single_attempt()
            if res is not None:
                return res
            attempts += 1
            if attempts % 100 == 0:
                # Maybe randomize initialization? 
                # Current init is empty. 
                # Could start with random edges that don't form C4?
                pass
                
        return None

    # Try n=46, min_deg=7
    edges = try_construct(46, 7)
    if edges is not None:
        return {'n': 46, 'edges': edges}
        
    # Try n=45, min_deg=6
    edges = try_construct(45, 6)
    if edges is not None:
        return {'n': 45, 'edges': edges}
        
    # Fallback: n=43, min_deg=7 (Polarity graph of PG(2,6) is 43 vertices, deg 7)
    # We can construct it explicitly or just return a known valid smaller one.
    # For safety, return n=39 empty graph if all fails?
    # But 43 is better.
    # Let's just return n=39 if search fails, as it's guaranteed.
    # Or try n=44, min_deg=5?
    edges = try_construct(44, 5)
    if edges is not None:
        return {'n': 44, 'edges': edges}
        
    return {'n': 39, 'edges': []}
