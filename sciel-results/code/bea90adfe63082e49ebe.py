import time

_START_TIME = None

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Safety deadline: 85 seconds
    DEADLINE = _START_TIME + 85
    
    # Construct two disjoint K_{3,3} graphs.
    # This ensures:
    # 1. Min degree 3.
    # 2. Triangle-free (bipartite).
    # 3. Cycle lengths are exactly {4, 6}.
    # 4. Number of distinct cycle lengths is 2.
    
    edges = []
    
    # First K_{3,3} on vertices 0..5
    part_a1 = [0, 1, 2]
    part_b1 = [3, 4, 5]
    for u in part_a1:
        for v in part_b1:
            edges.append([u, v])
            
    # Second K_{3,3} on vertices 6..11
    part_a2 = [6, 7, 8]
    part_b2 = [9, 10, 11]
    for u in part_a2:
        for v in part_b2:
            edges.append([u, v])
            
    # Final cleanup and type checking
    final_edges = []
    seen = set()
    for e in edges:
        u, v = e
        u_int = int(u)
        v_int = int(v)
        if u_int == v_int:
            continue
        key = (min(u_int, v_int), max(u_int, v_int))
        if key not in seen:
            seen.add(key)
            final_edges.append([u_int, v_int])
            
    final_edges.sort()
    
    return final_edges
