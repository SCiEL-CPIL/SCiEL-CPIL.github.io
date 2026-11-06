"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 40

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # Safety margin before 90s kill
    
    # Hamiltonian cycle: 0-1-2-...-39-0
    edges = set()
    for i in range(N):
        edges.add((min(i, (i+1) % N), max(i, (i+1) % N)))
    
    # Chords from vertex 0 to vertices 2..20
    for k in range(2, 21):
        edges.add((0, k))
    
    edge_list = sorted(edges)
    
    # Construct cycles of every length 3..40
    cycles = []
    for length in range(3, N + 1):
        if length == N:
            # Full Hamiltonian cycle
            cycles.append(list(range(N)))
        else:
            # Use chord (0, k) where k = length - 1
            # Short arc: 0, 1, 2, ..., k, 0 → cycle 0-1-...-k
            k = length - 1
            if k <= 20:
                # Cycle: 0, 1, 2, ..., k
                cycle = list(range(k + 1))
            else:
                # k > 20, use the long arc: 0, 39, 38, ..., k
                # This gives length 41 - k
                # Actually if length > 21, we need 41-k = length, so k = 41 - length
                k = 41 - length
                cycle = [0] + list(range(39, k - 1, -1))
            cycles.append(cycle)
    
    return {'edges': edge_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
