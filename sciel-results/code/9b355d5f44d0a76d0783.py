import time

N = 40

def construct():
        # Deadline guard (optional but good practice per directive)
        # Not strictly needed for this O(N) construction, but safe.
        
        edges = set()
        
        # Hamiltonian Cycle
        for i in range(N):
            edges.add((i, (i+1)%N))
        
        # Chords from 0 to k for k in 2..20
        for k in range(2, 21):
            edges.add((0, k))
        
        # Convert to list of tuples for output
        # Ensure consistent ordering (min, max)
        edge_list = [tuple(sorted(e)) for e in edges]
        
        # Generate Cycles
        cycles = []
        
        # Length 3 to 40
        for L in range(3, N+1):
            if L == N:
                # Hamiltonian Cycle
                # 0-1-...-39-0
                cyc = list(range(N))
            elif L <= 21:
                # Use chord (0, L-1)
                # Path 0-1-...-(L-1)-0
                # Vertices: 0, 1, ..., L-1
                k = L - 1
                # Cycle: 0, 1, 2, ..., k, 0
                # The list of vertices in order
                cyc = list(range(0, k+1))
            else:
                # L > 21
                # Use chord (0, 41-L)
                k = 41 - L
                # Cycle on long side: 0, k, k+1, ..., 39, 0
                # Vertices: 0, k, k+1, ..., 39
                cyc = [0] + list(range(k, N))
            
            # Verify cycle validity (distinct vertices, edges exist)
            # For robustness, we can add a check, but the construction is deterministic.
            cycles.append(cyc)
            
        return {'edges': edge_list, 'cycles': cycles}
