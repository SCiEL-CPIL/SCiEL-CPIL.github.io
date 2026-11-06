import time
import sys

N = 40

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin from 90s limit

    # Strategy: Hamiltonian Cycle + 19 Chords from Vertex 0
    # 
    # Theoretical Lower Bound Analysis:
    # 1. The graph must contain a cycle of length 40, which implies a Hamiltonian cycle.
    #    This requires 40 edges.
    # 2. We need to cover cycle lengths 3 through 39 (37 lengths).
    # 3. Any cycle of length L < 40 must use at least one chord (edge not in the Hamiltonian cycle).
    # 4. A single chord (u, v) in a Hamiltonian cycle creates exactly two simple cycles 
    #    involving that chord: one using the "short" path and one using the "long" path.
    #    Let the distance along the cycle be d. The cycle lengths are d+1 and N-d+1.
    # 5. Thus, each chord can "certify" at most 2 distinct cycle lengths (unless d=N/2, 
    #    in which case it certifies only 1 unique length N/2+1).
    # 6. To cover 37 lengths, we need at least ceil(37/2) = 19 chords.
    # 7. Total edges = 40 (base) + 19 (chords) = 59.
    # 8. Score = 59 - 40 = 19.
    #
    # This construction achieves this bound by using chords (0, k) for k in 2..20.
    # - k=2 covers lengths 3 and 39.
    # - k=3 covers lengths 4 and 38.
    # - ...
    # - k=19 covers lengths 20 and 22.
    # - k=20 covers length 21.
    # All lengths 3..39 are covered.

    edges = set()
    
    # 1. Base Hamiltonian Cycle: (i, (i+1)%N)
    for i in range(N):
        u, v = i, (i + 1) % N
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    # 2. Chords from 0 to k for k in 2..20
    # These cover all cycle lengths 3..39.
    for k in range(2, 21):
        u, v = 0, k
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    # 3. Generate Cycles
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Hamiltonian cycle: 0-1-2-...-39-0
            cycle_verts = list(range(N))
        else:
            if L <= 21:
                # Use short path from 0 to L-1
                # Cycle: 0, 1, ..., L-1, 0
                # Requires chord (0, L-1)
                k = L - 1
                # Vertices: 0, 1, ..., k
                cycle_verts = list(range(k + 1))
            else:
                # Use long path from 0 to k via N-1
                # Cycle: 0, N-1, ..., k, 0
                # Requires chord (0, k)
                # Length L = (N - k) + 1 => k = N - L + 1
                k = N - L + 1
                # Vertices: 0, N-1, N-2, ..., k
                cycle_verts = [0] + list(range(N - 1, k - 1, -1))
        
        cycles.append(cycle_verts)
        
    # Validation
    edge_set = set(tuple(sorted(e)) for e in edges)
    valid = True
    
    # Check edges distinct and valid
    if len(edge_set) != len(edges):
        valid = False
    for u, v in edge_set:
        if not (0 <= u < N and 0 <= v < N and u != v):
            valid = False
            break
            
    # Check cycles
    if len(cycles) != N - 2:
        valid = False
    else:
        for idx, L in enumerate(range(3, N + 1)):
            cycle = cycles[idx]
            if len(cycle) != L:
                valid = False
                break
            if len(set(cycle)) != L:
                valid = False
                break
            for i in range(L):
                u = cycle[i]
                v = cycle[(i + 1) % L]
                if (min(u, v), max(u, v)) not in edge_set:
                    valid = False
                    break
            if not valid:
                break
                
    if valid:
        return {'edges': [tuple(sorted(e)) for e in edges], 'cycles': cycles}
    else:
        # This should never happen, but if it does, we raise an error 
        # to indicate a bug in the construction logic rather than returning 
        # a suboptimal fallback which might also be invalid or just worse.
        # However, the task says "invalid result... scores 0".
        # The fallback fan graph is valid but has score 37.
        # It is better to return the fan graph than to raise an exception (score 0).
        # So we keep the fallback.
        fan_edges = set()
        for i in range(N - 1):
            fan_edges.add((i, i + 1))
        for i in range(2, N):
            fan_edges.add((0, i))
            
        fan_cycles = []
        for k in range(3, N + 1):
            fan_cycles.append(list(range(k)))
            
        return {'edges': [tuple(sorted(e)) for e in fan_edges], 'cycles': fan_cycles}
