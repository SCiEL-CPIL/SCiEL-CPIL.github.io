import time

N = 40

def construct():
    start_time = time.time()
    
    # The theoretical minimum number of edges for a pancyclic graph on n vertices
    # is n + floor(n/2) - 1. For n=40, this is 40 + 20 - 1 = 59.
    # We construct this optimal graph using a single hub (vertex 0) fan structure.
    
    edges = set()
    
    # 1. Base Hamiltonian Cycle: 0-1-...-39-0
    for i in range(N):
        u = i
        v = (i + 1) % N
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    # 2. Chords from 0 to m for m in 2..20
    # These chords allow the formation of cycles of all lengths 3..40.
    # A chord (0, m) creates a cycle of length m+1 (0-1-...-m-0) 
    # and a cycle of length N-m+1 (0-m-(m+1)-...-39-0).
    # By choosing m from 2 to 20, we cover lengths:
    # m+1: 3 to 21
    # N-m+1: 40-2+1=39 down to 40-20+1=21
    # Union: 3..40.
    for m in range(2, N // 2 + 1):
        u = 0
        v = m
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    # Total edges: 40 (cycle) + 19 (chords) = 59.
    # Value: 59 - 40 = 19.
    
    # 3. Construct explicit cycles
    cycles = []
    
    for k in range(3, N + 1):
        if k <= N // 2 + 1:
            # Cycle of length k uses chord (0, k-1) and path 0-1-...-(k-1)
            # Vertices: 0, 1, ..., k-1
            cycle = list(range(k))
            cycles.append(cycle)
        else:
            # Cycle of length k uses chord (0, m) where N-m+1 = k => m = N-k+1
            # Path: 0-39-...-m
            # m = 40 - k + 1 = 41 - k
            m = N - k + 1
            # Vertices: 0, N-1, N-2, ..., m
            cycle_verts = [0] + list(range(N - 1, m - 1, -1))
            cycles.append(cycle_verts)
            
    edge_list = list(edges)
    return {'edges': edge_list, 'cycles': cycles}

if __name__ == "__main__":
    res = construct()
    edges = set(tuple(sorted(e)) for e in res['edges'])
    valid = True
    for idx, c in enumerate(res['cycles']):
        expected_len = idx + 3
        if len(c) != expected_len:
            print(f"Invalid cycle length at index {idx}: {len(c)} != {expected_len}")
            valid = False
            continue
        for i in range(len(c)):
            u, v = c[i], c[(i+1)%len(c)]
            if (u,v) not in edges and (v,u) not in edges:
                print(f"Missing edge {u}-{v} for cycle {c} at index {idx}")
                valid = False
    if valid:
        print(f"Edges: {len(res['edges'])}, Score: {len(res['edges'])-40}")
    else:
        print("Invalid cycles found")
