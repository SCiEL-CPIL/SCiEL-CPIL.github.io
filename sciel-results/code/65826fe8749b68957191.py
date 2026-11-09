N = 40

def construct():
    # Base Hamiltonian cycle: 0-1-...-39-0
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    
    # Add chords to cover all cycle lengths 3..39
    # Chords from 0 to k
    # We determined k in {2, 3, ..., 19, 38}
    chords = list(range(2, 20)) + [38]
    for k in chords:
        edges.add((0, k))
    
    # Convert to list of tuples
    edge_list = [tuple(sorted(e)) for e in edges]
    
    # Generate cycles
    cycles = []
    # Length 40 is the Hamiltonian cycle
    # Lengths 3..39 are formed by chords
    # For a chord (0, k), the cycles are:
    # 1. 0-1-...-k-0 (length k+1)
    # 2. 0-(N-1)-...-k-0 (length N-k)
    
    # We need one cycle for each length 3..40.
    # We can just construct them explicitly.
    
    for length in range(3, N + 1):
        if length == N:
            # Hamiltonian cycle
            cyc = list(range(N))
            cycles.append(cyc)
        else:
            # Find a chord that generates a cycle of this length
            # We know which chords we added.
            # For length L, we need a chord k such that k+1 == L or N-k == L.
            # k = L-1 or k = N-L.
            k1 = length - 1
            k2 = N - length
            
            # Check if chord (0, k1) exists
            if (0, k1) in edges or (k1, 0) in edges:
                # Cycle 0-1-...-k1-0
                cyc = list(range(k1 + 1))
                cycles.append(cyc)
            else:
                # Check if chord (0, k2) exists
                if (0, k2) in edges or (k2, 0) in edges:
                    # Cycle 0-(N-1)-...-k2-0
                    # Vertices: 0, N-1, N-2, ..., k2
                    cyc = [0] + list(range(N - 1, k2 - 1, -1))
                    cycles.append(cyc)
                else:
                    # Fallback: should not happen if chords are correct
                    # But for safety, if not found, we have a bug in chord selection
                    # For now, raise error or use a default
                    raise ValueError(f"Could not find cycle for length {length}")
                    
    return {'edges': edge_list, 'cycles': cycles}
