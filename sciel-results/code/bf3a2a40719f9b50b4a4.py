N = 40

def construct():
    # Base cycle: vertices 0..39 in order
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    
    # Add chords from vertex 0 to vertices 2..20
    # Chord (0, k) creates cycles of length k+1 and 41-k
    for k in range(2, 21):  # k = 2, 3, ..., 20
        edges.add((0, k))
    
    edge_list = list(edges)
    
    # Construct cycles of each length 3..40
    cycles = []
    for length in range(3, N + 1):
        if length == N:
            # Hamiltonian cycle: 0, 1, 2, ..., 39, 0
            cycle = list(range(N))
        else:
            # length in 3..39
            # Use chord (0, k) where k+1 = length, so k = length - 1
            # Cycle: 0, 1, 2, ..., k, 0 where k = length - 1
            # This has length+1 vertices? No: 0,1,...,k has k+1 vertices = length vertices
            k = length - 1
            # The cycle is 0, 1, 2, ..., k
            # Vertices: 0, 1, ..., k (that's k+1 = length vertices)
            # Edges: (0,1), (1,2), ..., (k-1,k), (k,0) - all exist
            cycle = list(range(length))  # 0, 1, ..., length-1
            # Wait, k = length - 1, so vertices are 0, 1, ..., length-1
            # That's length vertices, correct
            # But we need to verify edge (length-1, 0) exists, which is chord (0, length-1)
            # This exists for length-1 in 2..20, i.e., length in 3..21
            # For length > 21, we need a different approach
            
            if length <= 21:
                cycle = list(range(length))
            else:
                # length in 22..39
                # Use the other cycle from chord (0, k) where 41-k = length, so k = 41-length
                # Cycle: 0, k, k+1, ..., 39, 0
                # Vertices: 0, k, k+1, ..., 39
                # Count: 1 + (39-k+1) = 41-k = length vertices
                k = 41 - length
                # Vertices: 0, k, k+1, ..., 39
                cycle = [0] + list(range(k, N))
                # Verify: this has 1 + (N - k) = 1 + 40 - k = 41 - k = length vertices
                # Edges: (0,k) [chord], (k,k+1), ..., (38,39), (39,0) [base cycle edge]
                # All exist
        
        cycles.append(cycle)
    
    return {'edges': edge_list, 'cycles': cycles}
