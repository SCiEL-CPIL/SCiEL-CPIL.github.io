import time

N = 40

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 60.0
    
    # Base Hamiltonian cycle edges
    edges = set()
    for i in range(N):
        edges.add((min(i, (i+1)%N), max(i, (i+1)%N)))
        
    # Chords from 0 to k for k in 2..20
    for k in range(2, 21):
        edges.add((0, k))
        
    # Convert to list for return
    edge_list = sorted(list(edges))
    
    # Generate cycles
    # We need cycles of length 3 to 40 in that order
    cycle_by_length = {40: list(range(N))}
    
    for k in range(2, 21):
        # Cycle 1: 0 -> 1 -> ... -> k -> 0
        len1 = k + 1
        cycle1 = list(range(k+1))
        cycle_by_length[len1] = cycle1
        
        # Cycle 2: 0 -> 39 -> 38 -> ... -> k -> 0
        len2 = N - k + 1
        cycle2 = [0] + list(range(N-1, k-1, -1))
        cycle_by_length[len2] = cycle2

    # Build the ordered list for lengths 3..40
    final_cycles = [cycle_by_length[L] for L in range(3, N+1)]
    
    return {'edges': edge_list, 'cycles': final_cycles}
