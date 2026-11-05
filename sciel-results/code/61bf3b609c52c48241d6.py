import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_n = 0
    best_edges = []
    
    # Try n=47 first, then 46
    for target_n in [47, 46]:
        if time.time() > deadline:
            break
            
        min_deg_req = target_n - 39
        
        # Multiple random restarts
        found = False
        for attempt in range(200):
            if time.time() > deadline:
                break
                
            # Initialize
            masks = [0] * target_n
            reach2 = [0] * target_n
            degrees = [0] * target_n
            
            # Greedy construction
            # We want to ensure min degree >= min_deg_req
            # Strategy: Always pick the vertex with the lowest degree
            # and try to connect it to a valid vertex
            
            # To avoid getting stuck in a specific local minimum, 
            # we can randomize the tie-breaking or the initial state.
            # Here we start empty.
            
            # List of vertices to process, sorted by degree dynamically?
            # Simpler: Loop until all degrees are sufficient or no moves possible.
            
            max_iter = 10000
            stuck = 0
            
            for _ in range(max_iter):
                # Find vertex with lowest degree
                min_deg = min(degrees)
                if min_deg >= min_deg_req:
                    # Check if ALL vertices are >= min_deg_req
                    if all(d >= min_deg_req for d in degrees):
                        found = True
                        break
                    
                # Find a vertex u with degree < min_deg_req
                # If all are >=, we break above.
                # So there must be at least one.
                candidates_u = [i for i in range(target_n) if degrees[i] < min_deg_req]
                
                if not candidates_u:
                    # All satisfied, but we didn't break?
                    # Actually if all satisfied, we break.
                    # If not all satisfied, candidates_u is not empty.
                    pass
                
                # Randomly pick one to add randomness
                u = random.choice(candidates_u)
                
                # Find valid v
                valid_vs = []
                for v in range(target_n):
                    if v == u:
                        continue
                    if masks[u] & (1 << v):
                        continue
                    # Check safety
                    if (masks[v] & reach2[u]) == 0:
                        valid_vs.append(v)
                
                if not valid_vs:
                    stuck += 1
                    if stuck > 50:
                        break
                    continue
                else:
                    stuck = 0
                    # Pick v with highest degree to balance? Or random?
                    # Highest degree might be safer (more connected, but reach2 is larger?)
                    # Actually, higher degree v has larger masks[v], so harder to be safe.
                    # So pick v with LOWEST degree?
                    # Or just random.
                    v = random.choice(valid_vs)
                    
                    # Add edge
                    # Update masks
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
                    
                    # Update degrees
                    degrees[u] += 1
                    degrees[v] += 1
                    
                    # Update reach2
                    # reach2[u] gets neighbors of v
                    reach2[u] |= masks[v]
                    # reach2[v] gets neighbors of u
                    reach2[v] |= masks[u]
                    
                    # Update neighbors of u
                    # Iterate through bits in masks[u] (old, before adding v? 
                    # Wait, masks[u] now includes v. 
                    # We need to update reach2 for all w in N(u) \ {v}
                    # And reach2 for all w in N(v) \ {u}
                    
                    # To iterate bits, we can use a helper or just loop 0..n-1
                    # Since n <= 47, loop is fast.
                    for w in range(target_n):
                        if w == u or w == v:
                            continue
                        # If w is neighbor of u
                        if masks[u] & (1 << w):
                            reach2[w] |= (1 << v)
                        # If w is neighbor of v
                        if masks[v] & (1 << w):
                            reach2[w] |= (1 << u)
                        
            if found:
                # Extract edges
                edges = []
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        if masks[i] & (1 << j):
                            edges.append([i, j])
                
                if target_n > best_n:
                    best_n = target_n
                    best_edges = edges
                break # Found for this n, try next n (smaller) or return?
                # We want max n. If we found 47, we can return.
                # If we found 46, we should return as 47 failed.
                # So break the n loop if found 47.
                if target_n == 47:
                    return {'n': best_n, 'edges': best_edges}

    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback: Return a valid small graph
    # n=39, min deg 0. Empty graph is valid.
    return {'n': 39, 'edges': []}
