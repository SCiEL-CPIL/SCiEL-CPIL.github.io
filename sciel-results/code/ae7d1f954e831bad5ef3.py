import time

N = 40

def construct():
    """
    Constructs a sparse pancyclic graph on 40 vertices.
    Returns a dictionary with 'edges' and 'cycles'.
    
    Strategy:
    1. Create a Hamiltonian cycle on vertices 0..39. (40 edges)
    2. Add chords from vertex 0 to vertices 2..20. (19 edges)
    Total edges: 59.
    Score: 59 - 40 = 19.
    
    Cycle Coverage:
    - The Hamiltonian cycle covers length 40.
    - Each chord (0, k) creates two cycles with the Hamiltonian paths:
      - Short path: 0 -> 1 -> ... -> k (length k) + chord (1) = k+1
      - Long path: 0 -> 39 -> ... -> k (length 40-k) + chord (1) = 41-k
    - For k in 2..20:
      - k=2: lengths 3, 39
      - k=3: lengths 4, 38
      - ...
      - k=19: lengths 20, 22
      - k=20: length 21 (both paths length 20, so 21)
    - Union of lengths: {3, ..., 20} U {21, 22, ..., 39} = {3, ..., 39}.
    - All lengths 3..40 are covered.
    """
    
    edges = set()
    
    # 1. Hamiltonian Cycle
    for i in range(N):
        u = i
        v = (i + 1) % N
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    # 2. Chords from 0 to k for k in 2..20
    for k in range(2, 21):
        u = 0
        v = k
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    edges_list = list(edges)
    
    cycles = []
    # Generate cycles for lengths 3 to 40
    for L in range(3, N + 1):
        if L == N:
            # Hamiltonian cycle
            cycle = list(range(N))
        elif L <= 21:
            # Use chord (0, L-1)
            # Cycle: 0 -> 1 -> ... -> L-1 -> 0
            # Vertices: 0, 1, ..., L-1
            # Note: L-1 is the chord endpoint.
            # Path 0 to L-1 is 0, 1, ..., L-1.
            # Edge (L-1, 0) closes it.
            k = L - 1
            # Ensure k is in 2..20 for L in 3..21
            # If L=3, k=2. If L=21, k=20.
            cycle = list(range(L))
        else:
            # L in 22..39
            # Use chord (0, k) where 41 - k = L => k = 41 - L
            k = 41 - L
            # Cycle: 0 -> 39 -> 38 -> ... -> k -> 0
            # Vertices: 0, 39, 38, ..., k
            # The path is 0 to k via 39.
            # Sequence: 0, 39, 38, ..., k
            cycle = [0] + list(range(N - 1, k - 1, -1))
            
        cycles.append(cycle)
        
    return {'edges': edges_list, 'cycles': cycles}

if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res['edges'])}, Cycles: {len(res['cycles'])}")
