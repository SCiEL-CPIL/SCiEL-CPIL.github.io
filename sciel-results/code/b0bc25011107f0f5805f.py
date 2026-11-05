import time

def construct():
    N = 40
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    # Build a ladder graph with vertices 0..19 (top) and 20..39 (bottom)
    # Top path: (i, i+1) for i=0..18
    # Bottom path: (i+20, i+21) for i=0..18
    # Rungs: (i, i+20) for i=0..19
    # One extra chord: (0, 21) to create the triangle and cover odd cycles
    
    edges = []
    edges_set = set()
    
    def add_edge(a, b):
        if a == b:
            return
        if a > b:
            a, b = b, a
        if (a, b) not in edges_set:
            edges_set.add((a, b))
            edges.append((a, b))
            
    # Top path
    for i in range(19):
        add_edge(i, i+1)
        
    # Bottom path
    for i in range(19):
        add_edge(i+20, i+21)
        
    # Rungs
    for i in range(20):
        add_edge(i, i+20)
        
    # Extra chord to create triangle: (0, 21)
    add_edge(0, 21)
    
    # Check if we are within time limit (should be instantaneous)
    if time.time() - start_time > deadline:
        # Fallback: return current best (which is complete)
        pass

    cycles = []
    
    for k in range(3, 41):
        if k % 2 == 1:
            # Odd cycle: use the chord (0, 21)
            # Pattern: 0-1-...-m-(m+20)-(m+19)-...-21-0
            # Length = 2m + 1
            m = (k - 1) // 2
            # Cycle: 0, 1, ..., m, m+20, m+19, ..., 21, 0
            cyc = list(range(0, m+1))
            cyc.extend(range(m+20, 20, -1))
            cycles.append(cyc)
        else:
            # Even cycle: use the ladder structure without the chord
            # Pattern: 0-1-...-m-(m+20)-(m+19)-...-20-0
            # Length = 2m + 2
            m = k // 2 - 1
            # Cycle: 0, 1, ..., m, m+20, m+19, ..., 20, 0
            cyc = list(range(0, m+1))
            cyc.extend(range(m+20, 19, -1))
            cycles.append(cyc)
            
    return {'edges': edges, 'cycles': cycles}
