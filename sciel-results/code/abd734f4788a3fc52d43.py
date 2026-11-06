"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit

    # Construction: Hamiltonian cycle + 19 chords from vertex 0
    # This yields 59 edges total.
    # Value = 59 - 40 = 19.
    
    edges = set()
    
    # 1. Add Hamiltonian cycle edges
    for i in range(N):
        edges.add((i, (i + 1) % N))
        
    # 2. Add chords (0, j) for j in 2..20
    # These chords create cycles of length j+1 and 41-j
    for j in range(2, 21):
        edges.add((0, j))
        
    edge_list = list(edges)
    
    # 3. Construct cycles for lengths 3..40
    cycles = []
    
    for L in range(3, N + 1):
        # Check deadline (though this loop is fast)
        if time.time() > deadline:
            # If we somehow hit deadline, return what we have (should not happen)
            # But we must return a valid structure. 
            # Since we are in the middle of construction, we might have incomplete cycles.
            # Better to ensure construction is fast enough.
            pass

        if L == N:
            # Hamiltonian cycle
            cyc = list(range(N))
        elif L <= 21:
            # Use chord j = L-1
            # Cycle: 0 -> 1 -> ... -> L-1 -> 0
            # Vertices: 0, 1, ..., L-1
            cyc = list(range(L))
        else:
            # L in 22..39
            # Use chord j = 41 - L
            j = 41 - L
            # Cycle: 0 -> 39 -> ... -> j -> 0
            # Vertices: 0, 39, 38, ..., j
            cyc = [0] + list(range(39, j - 1, -1))
            
        cycles.append(cyc)
        
    return {'edges': edge_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
