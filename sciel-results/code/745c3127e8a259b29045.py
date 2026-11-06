import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # We want to maximize n.
    # Try n from 47 down to 40.
    # If we find a valid graph for n, return it.
    
    for n in range(47, 39, -1):
        if time.time() > deadline - 10:
            break
            
        min_deg_req = n - 39
        
        # Multiple attempts for each n
        # Increase attempts for smaller n where success is more likely
        if n >= 46:
            max_attempts = 20
        else:
            max_attempts = 50
            
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            # 1. Initialize
            adj = [0] * n
            degrees = [0] * n
            
            # Generate all possible edges
            edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(edges)
            
            # 2. Greedy construction
            for u, v in edges:
                if time.time() > deadline - 2:
                    break
                # Check if adding (u, v) creates a C4
                # Condition: No a in N(u) such that N(a) intersects N(v)
                # Note: u, v are not adjacent yet.
                
                c4_found = False
                # Iterate over neighbors of u
                temp = adj[u]
                while temp:
                    lsb = temp & (-temp)
                    a = lsb.bit_length() - 1
                    # Check if any neighbor of a is in N(v)
                    if adj[a] & adj[v]:
                        c4_found = True
                        break
                    temp ^= lsb
                
                if not c4_found:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
            
            # 3. Local Search
            # Goal: Increase min degree to at least min_deg_req
            # Strategy: 
            #   - Try to add edges incident to low-degree vertices.
            #   - If stuck, try edge swaps (remove one, add another) to rebalance degrees.
            
            improved = True
            polish_deadline = deadline - 2
            while improved and time.time() < polish_deadline:
                improved = False
                min_deg = min(degrees)
                
                if min_deg >= min_deg_req:
                    break
                
                # Find vertices with min degree
                min_deg_val = min(degrees)
                low_verts = [i for i in range(n) if degrees[i] == min_deg_val]
                random.shuffle(low_verts)
                
                # Phase 1: Try to add edges
                for u in low_verts:
                    if time.time() > polish_deadline:
                        break
                    
                    # Candidates for v: non-neighbors of u
                    # We want to pick v that doesn't create C4.
                    # To speed up, we can shuffle all non-neighbors and try them.
                    non_neighbors = [v for v in range(n) if v != u and not (adj[u] >> v) & 1]
                    random.shuffle(non_neighbors)
                    
                    added = False
                    for v in non_neighbors:
                        # Check C4 for adding (u, v)
                        c4_found = False
                        temp = adj[u]
                        while temp:
                            lsb = temp & (-temp)
                            a = lsb.bit_length() - 1
                            if adj[a] & adj[v]:
                                c4_found = True
                                break
                            temp ^= lsb
                        
                        if not c4_found:
                            # Add edge
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            degrees[u] += 1
                            degrees[v] += 1
                            improved = True
                            added = True
                            break
                    
                    if added:
                        break # Restart loop as degrees changed
                
                # Phase 2: If no edge could be added, try a random edge swap
                if not improved:
                    # Pick a random edge to remove
                    # To ensure we don't remove too many, we only swap if it helps min degree
                    # Find an edge (a, b) where degrees[a] > min_deg_req and degrees[b] > min_deg_req
                    # Or just any edge, and check if swap improves min degree.
                    
                    # Collect edges
                    current_edges = []
                    for i in range(n):
                        temp = adj[i]
                        while temp:
                            lsb = temp & (-temp)
                            j = lsb.bit_length() - 1
                            if i < j:
                                current_edges.append((i, j))
                            temp ^= lsb
                    
                    if current_edges:
                        a, b = random.choice(current_edges)
                        # Remove edge (a, b)
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                        degrees[a] -= 1
                        degrees[b] -= 1
                        
                        # Find a non-edge (c, d) to add
                        # Prefer c, d with low degrees
                        non_edges = []
                        # This is expensive to generate every time.
                        # Instead, pick random c, d
                        c = random.randint(0, n-1)
                        d = random.randint(0, n-1)
                        
                        # We need to find a valid non-edge to add.
                        # Let's try a few random pairs
                        swap_success = False
                        for _ in range(20):
                            c = random.randint(0, n-1)
                            d = random.randint(0, n-1)
                            if c == d: continue
                            if (adj[c] >> d) & 1: continue
                            
                            # Check if adding (c, d) to current graph (without a,b) is C4-free
                            c4_found = False
                            temp = adj[c]
                            while temp:
                                lsb = temp & (-temp)
                                x = lsb.bit_length() - 1
                                if adj[x] & adj[d]:
                                    c4_found = True
                                    break
                                temp ^= lsb
                            
                            if not c4_found:
                                # Check if this swap improves min degree
                                # Degrees change:
                                # a, b lose 1. c, d gain 1.
                                # New min degree?
                                # If min_deg was determined by c or d, it increases.
                                # If min_deg was determined by a or b, it decreases (bad).
                                # So we should only swap if c, d have low degree and a, b have high degree.
                                
                                if degrees[c] <= min_deg_val and degrees[d] <= min_deg_val and degrees[a] > min_deg_val and degrees[b] > min_deg_val:
                                    # Accept swap
                                    adj[c] |= (1 << d)
                                    adj[d] |= (1 << c)
                                    degrees[c] += 1
                                    degrees[d] += 1
                                    swap_success = True
                                    improved = True
                                    break
                            
                        if not swap_success:
                            # Revert removal of (a, b)
                            adj[a] |= (1 << b)
                            adj[b] |= (1 << a)
                            degrees[a] += 1
                            degrees[b] += 1

            # Final check
            min_deg = min(degrees)
            if min_deg >= min_deg_req:
                # Valid graph found
                edges_list = []
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj[i] >> j) & 1:
                            edges_list.append([i, j])
                return {'n': n, 'edges': edges_list}
        
    # Fallback: return a trivial valid graph (e.g., n=39, empty graph? No, min deg 0. Req 0. Valid.)
    # Or n=40, min deg 1.
    # Let's return n=39, edges=[]
    return {'n': 39, 'edges': []}
