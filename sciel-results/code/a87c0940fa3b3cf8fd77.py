"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base = {tuple(e) for e in P['edges']}
    # Ensure we have a set of undirected edges
    base_undirected = set()
    for u, v in base:
        base_undirected.add((u, v))
        base_undirected.add((v, u))
        
    # Strategy: Single Hub at 11 (Odd)
    # Connect 11 to all Even vertices except its immediate neighbors (10, 12)
    hub = 11
    evens = [i for i in range(n) if i % 2 == 0]
    
    added = []
    for e in evens:
        if e == hub - 1 or e == hub + 1:
            continue
        # Check if edge exists in base
        if (hub, e) not in base_undirected and (e, hub) not in base_undirected:
            added.append([hub, e])
            
    return added
# EVOLVE-BLOCK-END
