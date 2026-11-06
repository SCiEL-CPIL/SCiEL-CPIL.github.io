import time

def construct():
    """
    Constructs a graph on 32 vertices with no 5-cycle and no independent set of size 9.
    The graph is a disjoint union of 8 complete graphs K4.
    
    - Each K4 has 4 vertices and 6 edges.
    - Total vertices n = 8 * 4 = 32.
    - Total edges = 8 * 6 = 48.
    - A K4 does not contain a 5-cycle. Since the components are disconnected, no 5-cycle can span multiple components.
    - The independence number of a K4 is 1.
    - The independence number of the disjoint union is the sum of the independence numbers of the components: 8 * 1 = 8.
    - 8 < 9, so the condition is satisfied.
    - It is known that R(C5, K9) = 33, meaning no graph on 33 vertices satisfies these conditions.
    - Thus, n=32 is the maximum possible value.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 32
    edges = []
    
    # Construct 8 disjoint K4s
    for b in range(8):
        base = 4 * b
        # Add all edges for K4 on vertices base, base+1, base+2, base+3
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
                
    # Verify time limit (though this is instant)
    if time.time() > deadline:
        # Fallback to a smaller valid graph if somehow we timed out (unlikely)
        # But for n=32, this is just a safety net.
        pass

    return {'n': n, 'edges': edges}
