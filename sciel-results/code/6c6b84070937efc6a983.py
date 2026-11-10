import numpy as np

N = 40

def construct():
    # Vertices 0 to N-1
    # Base: Cycle C_N
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    
    # Chords from 0 to j
    # We need to cover all cycle lengths L in 3..N-1
    # Chord (0, j) creates cycles of length j+1 and N-j+1
    # We need to select S subset of {2..N-2} such that for all L in 3..N-1,
    # L-1 in S or N-L+1 in S
    # This is equivalent to picking one from each pair {j, N-j} for j in 2..N-2
    
    S = set()
    # Pairs are {j, N-j}
    # We pick j for j in 2..N//2
    # If N is even, N//2 pairs with itself.
    # If N is odd, (N-1)//2 pairs with (N+1)//2.
    
    # For N=40:
    # j=2 -> 38
    # j=3 -> 37
    # ...
    # j=19 -> 21
    # j=20 -> 20
    
    for j in range(2, N // 2 + 1):
        S.add(j)
        
    for j in S:
        edges.add((0, j))
        
    edge_list = list(edges)
    
    # Generate cycles
    # cycles[k] should be a cycle of length k+3
    # Lengths needed: 3, 4, ..., 40
    
    cycles = [None] * (N - 2) # Index 0 is len 3, Index N-3 is len N
    
    # Length N is the base cycle
    cycles[N - 3] = list(range(N))
    
    # For each chord j in S, we have two cycles
    # Cycle 1: 0, 1, ..., j, 0 (Length j+1)
    # Cycle 2: 0, j, j+1, ..., N-1, 0 (Length N-j+1)
    
    for j in S:
        len1 = j + 1
        len2 = N - j + 1
        
        # Cycle 1
        c1 = list(range(0, j + 1))
        # c1 is 0..j. We need to close it. The edge (j,0) exists.
        # The list of vertices is 0, 1, ..., j.
        # This is a valid simple cycle.
        
        # Cycle 2
        c2 = [0] + list(range(j, N))
        # c2 is 0, j, j+1, ..., N-1.
        # Edge (N-1, 0) exists in base cycle.
        # Edge (0, j) is the chord.
        
        # Assign to correct slots
        # We need to fill cycles array for lengths 3 to N-1
        # len1 and len2 are in 3..N-1
        
        if len1 <= N - 1:
            cycles[len1 - 3] = c1
        if len2 <= N - 1:
            cycles[len2 - 3] = c2
            
    # Check if all are filled
    # For N=40, S={2..20}.
    # j=2: len1=3, len2=39.
    # j=3: len1=4, len2=38.
    # ...
    # j=19: len1=20, len2=22.
    # j=20: len1=21, len2=21.
    # All lengths 3..39 covered.
    
    # Ensure distinct vertices in cycles (they are by construction)
    # Ensure edges exist (they do by construction)
    
    return {'edges': edge_list, 'cycles': cycles}
