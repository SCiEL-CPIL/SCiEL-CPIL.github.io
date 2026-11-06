import time

N = 40
START_TIME = time.time()
DEADLINE = 90  # seconds

def elapsed():
    return time.time() - START_TIME

def construct():
    """
    Constructs a sparse pancyclic graph on 40 vertices.
    
    Strategy:
    1. Start with a Hamiltonian cycle (40 edges).
    2. Add 19 chords from vertex 0 to vertices 2..20.
       - Chord (0, k) creates a cycle of length k+1 and 41-k.
       - This covers all lengths 3..39.
    3. Total edges: 40 + 19 = 59.
    4. Value: 59 - 40 = 19.
    """
    
    # Base Hamiltonian Cycle
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    
    # Add Chords
    # We need chords (0, k) for k in 2..20
    # This covers lengths:
    # k=2 -> 3, 39
    # k=3 -> 4, 38
    # ...
    # k=19 -> 20, 22
    # k=20 -> 21, 21
    for k in range(2, 21):
        edges.add((0, k))
    
    edge_list = list(edges)
    
    # Generate Cycles
    cycles = []
    
    # Length 40: The Hamiltonian Cycle
    cycles.append(list(range(N)))
    
    # Lengths 3..39
    for L in range(3, N):
        if L <= 21:
            # Use chord (0, L-1)
            # Cycle: 0 -> 1 -> ... -> L-1 -> 0
            # Vertices: 0, 1, ..., L-1
            cycle = list(range(L))
        else:
            # Use chord (0, 41-L)
            # Let k = 41-L
            # Cycle: 0 -> 39 -> ... -> k -> 0
            # Vertices: 0, 39, 38, ..., k
            k = 41 - L
            # The path goes 39 down to k
            # List: 0, 39, 38, ..., k
            # Note: The cycle is 0-39-38-...-k-0
            # Vertices in order: 0, 39, 38, ..., k
            cycle = [0] + list(range(39, k - 1, -1))
        
        cycles.append(cycle)
        
        # Safety check (optional, but good for debugging in dev)
        # In production, we trust the construction.
        
    # Ensure we return the correct format
    # 'edges': undirected edge pairs
    # 'cycles': one vertex-list cycle of every length 3..40 in that order
    # The order in cycles list:
    # Index 0: Length 40
    # Index 1: Length 3
    # Index 2: Length 4
    # ...
    # Index 37: Length 39
    
    # The prompt says: "cycles': one vertex-list cycle of every length 3..40 in that order"
    # This implies the list should be [Cycle_3, Cycle_4, ..., Cycle_40]
    # My current list is [Cycle_40, Cycle_3, Cycle_4, ...]
    # I need to reorder.
    
    final_cycles = [None] * (N - 2) # Indices 0..37 correspond to lengths 3..40
    # Wait, lengths are 3..40. Count = 38.
    # List indices 0..37.
    # Index 0 -> Length 3
    # Index 37 -> Length 40
    
    # My 'cycles' list currently:
    # cycles[0] is Length 40
    # cycles[1] is Length 3
    # cycles[2] is Length 4
    # ...
    # cycles[37] is Length 39
    
    # Reorder:
    result_cycles = [None] * 38
    for i, c in enumerate(cycles):
        if i == 0:
            # Length 40 goes to index 37
            result_cycles[37] = c
        else:
            # Length 3..39 go to indices 0..36
            # cycles[i] has length i+2
            # We want length L at index L-3
            # So cycles[i] (len i+2) goes to index (i+2)-3 = i-1
            result_cycles[i - 1] = c
            
    return {'edges': edge_list, 'cycles': result_cycles}
