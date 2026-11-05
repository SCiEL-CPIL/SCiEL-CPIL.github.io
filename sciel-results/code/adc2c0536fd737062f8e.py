N = 40

# EVOLVE-BLOCK-START
def construct():
    # Backbone cycle 0-1-...-39-0
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    
    # Chords from 0 to k for k in 2..20
    for k in range(2, 21):
        edges.add((0, k))
    
    # Convert to list of tuples for output
    edge_list = sorted([(min(u,v), max(u,v)) for u,v in edges])
    
    # Generate cycles
    cycles = []
    for L in range(3, N+1):
        if L == N:
            # The backbone cycle
            cycles.append(list(range(N)))
        else:
            # Find chord that covers L
            # L = k + 1  => k = L - 1
            # L = N - k + 1 => k = N - L + 1
            found = False
            # Try k from L-1
            k1 = L - 1
            if 2 <= k1 <= 20:
                # Cycle: 0-1-...-k-0
                cyc = list(range(k1 + 1))
                cycles.append(cyc)
                found = True
            
            if not found:
                # Try k from N - L + 1
                k2 = N - L + 1
                if 2 <= k2 <= 20:
                    # Cycle: 0-k-(k+1)-...-(N-1)-0
                    # Vertices: k, k+1, ..., N-1, 0
                    # Order in cycle: 0, k, k+1, ..., N-1
                    # Check edges: (0,k) is chord. (k, k+1) ... (N-1, 0) are backbone.
                    cyc = [0] + list(range(k2, N))
                    cycles.append(cyc)
                    found = True
            
            if not found:
                raise ValueError(f"Cycle {L} not found")
                
    return {'edges': edge_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
