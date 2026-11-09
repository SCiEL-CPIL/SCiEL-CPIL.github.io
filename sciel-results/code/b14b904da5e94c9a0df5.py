import time

N = 40

def construct():
    # Deadline guard
    t_start = time.time()
    limit = t_start + 58.0

    # Construct the edge set
    # Hamiltonian cycle: i -> (i+1)%N
    # Chords: 0 -> k for k in 2..20
    
    edge_set = set()
    
    # Add Hamiltonian cycle edges
    for i in range(N):
        u, v = i, (i + 1) % N
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
        
    # Add chords
    for k in range(2, 21):
        u, v = 0, k
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
        
    # Convert to sorted list
    final_edges = sorted(edge_set)
    
    # Construct cycles
    # We need cycles of length L for L in 3..40
    # L=40 is the Hamiltonian cycle
    # L=k+1 is 0..k
    # L=41-k is 0, 39..k
    
    cycles_map = {}
    
    # L=40
    cycles_map[40] = list(range(N))
    
    for k in range(2, 21):
        # Cycle 1: Length k+1
        # Vertices: 0, 1, ..., k
        c1 = list(range(k + 1))
        cycles_map[k + 1] = c1
        
        # Cycle 2: Length 41 - k
        # Vertices: 0, 39, 38, ..., k
        c2 = [0] + list(range(N - 1, k - 1, -1))
        cycles_map[41 - k] = c2
        
        # Check time
        if time.time() > limit:
            break
            
    # Assemble final cycles list
    result_cycles = []
    for L in range(3, N + 1):
        if L in cycles_map:
            result_cycles.append(cycles_map[L])
        else:
            # Fallback, should not happen
            raise RuntimeError(f"Missing cycle {L}")
            
    return {
        'edges': final_edges,
        'cycles': result_cycles
    }
