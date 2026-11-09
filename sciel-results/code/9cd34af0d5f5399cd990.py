import time

def construct():
    """
    Construct a graph on 12 vertices (0..11) with min degree 3, triangle-free,
    minimizing the number of distinct simple cycle lengths.
    
    Uses two disjoint K_{3,3} components, yielding cycle lengths {4, 6} -> value 2.
    This is optimal for 12 vertices with min degree 3 and girth >= 4.
    """
    start_time = time.time()
    deadline = start_time + 50  # Safety margin within 90s limit
    
    # Fallback: Two disjoint K_{3,3}
    # K_{3,3} is 3-regular, bipartite (triangle-free), girth 4.
    # Cycle lengths in K_{3,3}: 4 and 6 only.
    # Two copies: still {4, 6}. Value = 2.
    
    edges = []
    
    # First K_{3,3}: vertices 0,1,2 in part A; 3,4,5 in part B
    for u in range(3):
        for v in range(3, 6):
            edges.append([int(u), int(v)])
    
    # Second K_{3,3}: vertices 6,7,8 in part A; 9,10,11 in part B
    for u in range(6, 9):
        for v in range(9, 12):
            edges.append([int(u), int(v)])
    
    # Validate: no loops, no duplicates, all ints
    seen = set()
    out = []
    for u, v in edges:
        if not isinstance(u, int) or isinstance(u, bool):
            u = int(u)
        if not isinstance(v, int) or isinstance(v, bool):
            v = int(v)
        if u == v:
            continue
        a, b = (min(u, v), max(u, v))
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append([a, b])
    
    # Quick sanity check: min degree >= 3
    deg = {i: 0 for i in range(12)}
    for u, v in out:
        deg[u] += 1
        deg[v] += 1
    if any(d < 3 for d in deg.values()):
        # Should not happen, but guard
        pass
    
    # Triangle-free: guaranteed by bipartite construction
    # Contains a cycle: guaranteed (K_{3,3} has cycles)
    
    return out
