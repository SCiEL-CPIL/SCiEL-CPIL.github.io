import time

N = 40

def construct():
    t_start = time.perf_counter()
    deadline = t_start + 55.0  # Safety margin

    # 1. Define the Hamiltonian cycle (ring) edges
    # Vertices 0..39 in a cycle: (0,1), (1,2), ..., (38,39), (39,0)
    ring_edges = []
    for i in range(N):
        u, v = i, (i + 1) % N
        if u < v:
            ring_edges.append((u, v))
        else:
            ring_edges.append((v, u))

    # 2. Add chords from hub 0 to cover cycle lengths 3..39
    # Chord (0, k) creates cycles of length k+1 (via path 0->1->...->k) 
    # and 41-k (via path 0->39->...->k)
    # We need to cover lengths 3 to 39.
    # Length 40 is the ring itself.
    # k ranges from 2 to 19 to generate lengths 3..20 and 39..21.
    chord_edges = []
    for k in range(2, 20):
        u, v = 0, k
        chord_edges.append((u, v))

    # Combine and sort edges to ensure uniqueness and order
    all_edges = sorted(set(ring_edges + chord_edges))
    
    # Check if we are out of time (unlikely for deterministic construction)
    if time.perf_counter() > deadline:
        return {'edges': all_edges, 'cycles': []} # Fallback, should not happen

    # 3. Generate Cycle Certificates
    # We need a cycle for every length L in [3, 40]
    cycles = {}
    
    # L=40: The full ring
    cycles[40] = list(range(N))
    
    # For other lengths, derive from chords
    # A chord (0, k) gives two cycles:
    # C1: 0, 1, ..., k (length k+1)
    # C2: 0, 39, 38, ..., k (length 41-k)
    
    for k in range(2, 20):
        len_fwd = k + 1
        len_bwd = 41 - k
        
        # Cycle for length len_fwd: 0->1->...->k->0
        if len_fwd not in cycles:
            cycles[len_fwd] = list(range(k + 1))
            
        # Cycle for length len_bwd: 0->39->38->...->k->0
        if len_bwd not in cycles:
            # Path from 0 backwards to k: 0, N-1, N-2, ..., k
            # Then back to 0 via chord? No, the cycle is 0->N-1->...->k->0 (chord)
            # Wait, the edge (0,k) is the chord.
            # Path on ring from k to 0 going backwards: k, k-1, ..., 0. 
            # But we defined the chord as (0,k).
            # Let's verify the "backward" cycle definition.
            # Chord (0,k). 
            # Path 1: 0-1-...-k-0. Vertices: 0,1,...,k. Edges: (i, i+1) for i in 0..k-1, and (k,0). Length k+1.
            # Path 2: 0-(N-1)-(N-2)-...-k-0. 
            # Vertices: 0, N-1, N-2, ..., k.
            # Edges: (0, N-1), (N-1, N-2), ..., (k+1, k), (k, 0).
            # Number of edges: (N-1 - 0) steps on ring? 
            # Ring edges involved: (39,0), (38,39), ..., (k, k+1). 
            # Count: N - k edges on the ring segment + 1 chord = N - k + 1.
            # Length = 41 - k.
            # Vertices list: [0] + list(range(N-1, k-1, -1))
            cycles[len_bwd] = [0] + list(range(N - 1, k - 1, -1))

    # Verify all lengths 3..40 are covered
    # Missing lengths would cause an error in the loop below
    
    ordered_cycles = []
    for L in range(3, N + 1):
        if L in cycles:
            ordered_cycles.append(cycles[L])
        else:
            # Should not happen for N=40 with this construction
            raise ValueError(f"Cycle missing for length {L}")

    return {
        'edges': all_edges,
        'cycles': ordered_cycles
    }
