"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base = {tuple(e) for e in P['edges']}
    base_undirected = set()
    for u, v in base:
        base_undirected.add((u, v))
        base_undirected.add((v, u))
        
    # Strategy: Single Hub at 11 (Odd)
    # We determined that connecting 11 to a minimum vertex cover of the "even neighbor" graph
    # minimizes the edges.
    # The set of even vertices to connect to 11 is S = {2, 6, 10, 14, 18, 22}.
    # 10 is already a neighbor of 11 in the base path, so we don't add it.
    # Added edges: (11, 2), (11, 6), (11, 14), (11, 18), (11, 22).
    
    hub = 11
    target_evens = [2, 6, 14, 18, 22]
    
    added = []
    for e in target_evens:
        if (hub, e) not in base_undirected and (e, hub) not in base_undirected:
            added.append([hub, e])
            
    return added
# EVOLVE-BLOCK-END
