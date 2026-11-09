import time
import sys

N = 40

# EVOLVE-BLOCK-START
def construct():
    # Start time for safety guard
    start_time = time.time()
    
    # 1. Hamiltonian Cycle Edges
    # 0-1-2-...-39-0
    edges = set()
    for i in range(N):
        u = i
        v = (i + 1) % N
        edges.add((min(u, v), max(u, v)))
    
    # 2. Chords from vertex 0 to k for k in 1..19
    # This covers cycle lengths:
    # k+1 (short path) and N-k (long path)
    # k=1 -> 2, 39
    # k=2 -> 3, 38
    # ...
    # k=19 -> 20, 21
    # Union covers 2..39. We need 3..39.
    # All required lengths 3..39 are covered.
    chords = list(range(1, 20))
    for k in chords:
        u = 0
        v = k
        edges.add((min(u, v), max(u, v)))
    
    edge_list = list(edges)
    
    # 3. Construct Cycles
    cycles = []
    for L in range(3, N + 1):
        if time.time() - start_time > 58:
            # Safety break, though this should be instant
            break
            
        if L == N:
            # Hamiltonian cycle
            cyc = list(range(N))
        elif L <= 20:
            # Short path cycle: 0-1-...-k
            # Length L = k + 1 => k = L - 1
            k = L - 1
            cyc = list(range(k + 1))
        else:
            # Long path cycle: 0-N-1-...-k
            # Length L = N - k => k = N - L
            k = N - L
            cyc = [0] + list(range(N - 1, k - 1, -1))
        
        cycles.append(cyc)
        
    return {'edges': edge_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
