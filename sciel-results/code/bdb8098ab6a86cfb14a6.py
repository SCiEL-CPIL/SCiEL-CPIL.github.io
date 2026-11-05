import time

N = 40

# EVOLVE-BLOCK-START
def construct():
    # Strategy: 
    # Backbone: Hamiltonian cycle 0-1-...-39-0 (40 edges)
    # Chords: S = {2, 3, ..., 18, 21} (18 chords)
    # Total edges: 58.
    # 
    # Coverage:
    # L in [3, 19]: k = L-1 in [2, 18] subset S. Cycle 0-1-...-k-0.
    # L = 20: k = 19 not in S. 41-20=21 in S. Cycle 0-21-22-...-39-0.
    # L = 21: k = 20 not in S. 41-21=20 not in S. 
    #          Composite: Chords 2 and 21. Cycle 0-2-3-...-21-0. Length 1+19+1=21.
    # L = 22: k = 21 in S. Cycle 0-1-...-21-0.
    # L in [23, 39]: 41-L in [2, 18] subset S. Cycle 0-(41-L)-...-39-0.
    # L = 40: Backbone.

    edges = set()
    
    # Add backbone
    for i in range(N):
        u, v = i, (i + 1) % N
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    # Add chords
    # S = {2, ..., 18, 21}
    for k in range(2, 19):
        edges.add((0, k))
    edges.add((0, 21))
    
    edge_list = sorted([(min(u,v), max(u,v)) for u,v in edges])
    
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cycles.append(list(range(N)))
        elif L == 21:
            # Composite cycle using chords 2 and 21
            # 0 - 2 - 3 - ... - 21 - 0
            cyc = [0] + list(range(2, 22))
            cycles.append(cyc)
        else:
            k1 = L - 1
            k2 = N - L + 1
            
            found = False
            # Check if k1 is in our chord set
            # Chord set is {2..18} U {21}
            if (2 <= k1 <= 18) or k1 == 21:
                # Cycle: 0-1-...-k1-0
                cyc = list(range(k1 + 1))
                cycles.append(cyc)
                found = True
            
            if not found:
                # Check if k2 is in our chord set
                if (2 <= k2 <= 18) or k2 == 21:
                    # Cycle: 0-k2-(k2+1)-...-(N-1)-0
                    cyc = [0] + list(range(k2, N))
                    cycles.append(cyc)
                    found = True
            
            if not found:
                # Fallback: This should not happen with the specific set chosen
                # But for safety, if it does, we might need to search or raise
                # For L=20, k1=19 (no), k2=21 (yes).
                # For L=22, k1=21 (yes).
                # For L=23, k1=22 (no), k2=18 (yes).
                raise ValueError(f"Cycle {L} not found in optimized set")
                
    return {'edges': edge_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
