import time

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s kill

    edges = set()
    
    # Base Hamiltonian cycle: 0-1-2-...-39-0
    for i in range(N):
        u, v = i, (i + 1) % N
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    # Chords from 0 to k for k in 2..20
    # Cycle k+1: 0->1->2->...->k->0 (forward path)
    # Cycle 41-k: 0->39->38->...->k->0 (backward path)
    for k in range(2, 21):
        u, v = 0, k
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    edge_list = sorted(list(edges))
    
    # Generate cycles for lengths 3..40
    cycle_by_length = {40: list(range(N))}
    
    for k in range(2, 21):
        # Forward cycle: 0, 1, 2, ..., k (length k+1)
        len1 = k + 1
        cycle1 = list(range(k + 1))
        cycle_by_length[len1] = cycle1
        
        # Backward cycle: 0, 39, 38, ..., k (length 41-k)
        len2 = N - k + 1
        cycle2 = [0] + list(range(N - 1, k - 1, -1))
        cycle_by_length[len2] = cycle2

    # Build ordered list for lengths 3..40
    final_cycles = [cycle_by_length[L] for L in range(3, N + 1)]
    
    return {'edges': edge_list, 'cycles': final_cycles}
