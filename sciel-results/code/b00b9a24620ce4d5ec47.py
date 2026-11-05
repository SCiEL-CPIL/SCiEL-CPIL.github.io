import time

N = 40

def construct():
    t0 = time.time()
    
    # Strategy 1: Try to find a very sparse solution in the Ladder graph
    # But to ensure we don't timeout, we limit this to a very short time.
    # If it fails, we fall back to Wheel Graph.
    
    # Wheel Graph Construction
    edges = set()
    # Hub 0 to 1..39
    for i in range(1, N):
        edges.add((0, i))
    # Rim 1..39
    for i in range(1, N):
        edges.add((i, (i % N) + 1 if i < N-1 else 1)) 
        # Wait, rim is 1-2-...-39-1
        # (1,2), (2,3), ..., (38,39), (39,1)
        
    # Let's correct the rim edge generation
    edges = set()
    for i in range(1, N):
        edges.add((0, i))
    for i in range(1, N):
        if i < N - 1:
            edges.add((i, i + 1))
        else:
            edges.add((i, 1))
            
    # Generate Cycles
    cycles = []
    # Lengths 3 to 39
    for L in range(3, N):
        # Cycle: 0, 1, 2, ..., L-1
        # Vertices: 0, 1, ..., L-1
        # Edges: (0,1), (1,2), ..., (L-2, L-1), (L-1, 0)
        # All these edges exist in Wheel Graph
        cyc = [0] + list(range(1, L))
        cycles.append(cyc)
        
    # Length 40
    # Cycle: 1, 2, ..., 39, 1
    cyc = list(range(1, N))
    cycles.append(cyc)
    
    edge_list = sorted(edges)
    return {'edges': edge_list, 'cycles': cycles}
