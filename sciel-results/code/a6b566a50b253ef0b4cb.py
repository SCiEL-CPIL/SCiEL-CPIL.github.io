"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 40

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a sparse pancyclic graph on 40 vertices.
    Strategy: Hamiltonian cycle (40 edges) + 19 chords incident to vertex 0.
    Total edges: 59.
    Minimizes (edge_count - 40) = 19.
    """
    edges = []
    
    # 1. Base Hamiltonian Cycle: 0-1-...-39-0
    for i in range(N):
        edges.append((i, (i + 1) % N))
    
    # 2. Chords from 0 to k for k in 2..20
    # These chords create cycles of lengths k+1 and N-k+1.
    # Union of lengths covered: 3..39.
    for k in range(2, 21):
        edges.append((0, k))
        
    # Remove duplicates (none expected, but safe)
    edges = list(set(edges))
    
    # 3. Construct Cycles
    # We need one cycle of each length L from 3 to 40.
    cycles = []
    
    for L in range(3, N + 1):
        if L == N:
            # Length 40: The base cycle
            cycles.append(list(range(N)))
        else:
            # Length L (3..39)
            # Check if L = k + 1 for some k in 2..20
            # => k = L - 1
            k1 = L - 1
            if 2 <= k1 <= 20:
                # Cycle: 0 -> 1 -> ... -> k1 -> 0
                # Vertices: 0, 1, ..., k1
                cycle = list(range(0, k1 + 1))
                cycles.append(cycle)
            else:
                # Check if L = N - k + 1 for some k in 2..20
                # => N - k + 1 = L => k = N - L + 1
                k2 = N - L + 1
                if 2 <= k2 <= 20:
                    # Cycle: 0 -> k2 -> k2+1 -> ... -> N-1 -> 0
                    # Vertices: 0, k2, k2+1, ..., 39
                    cycle = [0] + list(range(k2, N))
                    cycles.append(cycle)
                else:
                    # Should not happen for this construction
                    raise ValueError(f"Failed to find cycle for length {L}")
    
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
