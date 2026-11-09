import numpy as np
import random
import time

def construct():
    # We want to maximize n. The known max is 35.
    # We will attempt to construct a valid graph for n=35 down to 1.
    # If we find one, we return it.
    
    max_n = 35
    min_n = 17 # Fallback to seed if search fails
    
    for n in range(max_n, min_n - 1, -1):
        # Try to find a valid graph of size n
        # Use a repair-based local search
        found = _find_valid_graph(n)
        if found is not None:
            return found
            
    # Fallback to Paley 17
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

def _find_valid_graph(n):
    # Time limit per n is roughly 120 / (35-17) ~ 6 seconds?
    # Let's give it 5 seconds per n to be safe.
    start_time = time.time()
    timeout = 5.0
    
    # Initialize random graph
    # Use a density that balances the constraints.
    # For R(4,6), p=0.5 is a good start.
    p = 0.5
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                adj[i, j] = True
                adj[j, i] = True
                
    # Helper to get edges list
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    edges.append([i, j])
        return edges

    # Helper to check validity and return violations
    def get_violations(adj):
        k4s = []
        indep6s = []
        
        # Find K4
        # Iterate over all 4-tuples? n=35, 35C4 = 52365. Fast enough.
        for a in range(n):
            for b in range(a+1, n):
                if not adj[a, b]: continue
                for c in range(b+1, n):
                    if not adj[a, c] or not adj[b, c]: continue
                    for d in range(c+1, n):
                        if adj[a, d] and adj[b, d] and adj[c, d]:
                            k4s.append((a, b, c, d))
                            
        # Find Independent 6-sets
        # Iterate over all 6-tuples? 35C6 = 1,623,160. Might be slow in Python loops.
        # We can optimize by checking if a set is independent.
        # Or we can just check if the complement graph has a K6.
        # Complement adjacency: ~adj (excluding diagonal)
        # Let's stick to checking 6-tuples but optimize with bitsets or numpy?
        # For n=35, 1.6M iterations in Python is ~0.1-0.5s. Acceptable for a few iterations.
        
        for a in range(n):
            for b in range(a+1, n):
                if adj[a, b]: continue
                for c in range(b+1, n):
                    if adj[a, c] or adj[b, c]: continue
                    for d in range(c+1, n):
                        if adj[a, d] or adj[b, d] or adj[c, d]: continue
                        for e in range(d+1, n):
                            if adj[a, e] or adj[b, e] or adj[c, e] or adj[d, e]: continue
                            for f in range(e+1, n):
                                if adj[a, f] or adj[b, f] or adj[c, f] or adj[d, f] or adj[e, f]:
                                    continue
                                indep6s.append((a, b, c, d, e, f))
                                
        return k4s, indep6s

    # Main loop
    for _ in range(1000):
        if time.time() - start_time > timeout:
            break
            
        k4s, indep6s = get_violations(adj)
        
        if not k4s and not indep6s:
            return {'n': n, 'edges': get_edges(adj)}
            
        if k4s:
            # Pick a random K4 and remove a random edge
            k4 = random.choice(k4s)
            # Edges in K4: (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
            edges_in_k4 = [(k4[0], k4[1]), (k4[0], k4[2]), (k4[0], k4[3]), 
                           (k4[1], k4[2]), (k4[1], k4[3]), (k4[2], k4[3])]
            u, v = random.choice(edges_in_k4)
            adj[u, v] = False
            adj[v, u] = False
        elif indep6s:
            # Pick a random independent set and add a random edge
            ind = random.choice(indep6s)
            # Edges NOT in graph within this set
            # All pairs in ind are non-edges. We pick one and make it an edge.
            i, j = random.sample(range(6), 2)
            u, v = ind[i], ind[j]
            adj[u, v] = True
            adj[v, u] = True
            
    return None
