"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    """Construct a sparse pancyclic graph on N=40 vertices.
    
    Design: A 40-cycle (rim) plus 18 chords (0,i) for i=3..20.
    - Length 40: the full rim cycle.
    - Length L (3<=L<=20): chord (0,L) creates cycle 0,1,...,L-1,0.
    - Length L (21<=L<=39): chord (0,40-L) creates cycle 
      0,1,...,39-L, 40-L, 41-L, ..., 38, 0 (length L).
    
    Total edges: 40 (rim) + 18 (chords) = 58.
    Score: 58 - 40 = 18 (objective) -> seed/objective = 38/18... 
    Actually the score is seed_value / child_value where value = edges - 40.
    Seed has 77 edges -> value 37. Child has 58 edges -> value 18.
    Score = 37/18 ≈ 2.056? No wait, let me re-read.
    
    "Minimize edge count minus 40" - so the metric is edges - 40.
    "The search score is seed/value" where value is the metric.
    Seed: 77 edges, value = 77-40 = 37.
    Child: 58 edges, value = 58-40 = 18.
    Score = 37/18 ≈ 2.056.
    
    Wait, but the observation shows 1.947 for a child. Let me recheck.
    37/19 = 1.947. So the child in observation had value 19, i.e., 59 edges.
    
    Let me recount my construction:
    - Rim: 40 edges (0-1, 1-2, ..., 38-39, 39-0)
    - Chords: (0,3), (0,4), ..., (0,20) = 18 chords
    - Total: 58 edges
    - Value: 58 - 40 = 18
    - Score: 37/18 ≈ 2.056
    
    This should be better than 1.947!
    """
    edges = []
    
    # Rim of the 40-cycle
    for i in range(N):
        edges.append((i, (i + 1) % N))
    
    # Chords: (0, k) for k = 3, 4, ..., 20
    for k in range(3, 21):
        edges.append((0, k))
    
    # Verify no duplicate edges
    edge_set = set()
    unique_edges = []
    for e in edges:
        key = (min(e[0], e[1]), max(e[0], e[1]))
        if key not in edge_set:
            edge_set.add(key)
            unique_edges.append(e)
    
    # Build cycles for lengths 3 through 40
    cycles = []
    
    for length in range(3, N + 1):
        if length == N:
            # Full Hamiltonian cycle: 0, 1, 2, ..., 39
            cyc = list(range(N))
        elif length <= 20:
            # Use chord (0, length): cycle is 0, 1, 2, ..., length-1, 0
            # Vertices: 0, 1, ..., length-1 (that's 'length' vertices)
            cyc = list(range(length))
        else:
            # Use chord (0, N-length): 
            # Cycle: 0, 1, ..., N-length-1, N-length, N-length+1, ..., 38, 0
            # That's N-length rim edges from 0 to N-length, then 
            # (N-length) to 38 is N-length-1 rim edges, plus chord (0, N-length)
            # Wait, let me think more carefully.
            # 
            # Chord is (0, N-length). The two paths from 0 to N-length on the rim:
            # Path 1: 0 -> 1 -> ... -> N-length (length = N-length edges)
            # Path 2: 0 -> N-1 -> N-2 -> ... -> N-length (length = length edges)
            # The cycle using chord + Path 2 has length (length) + 1 = length + 1.
            # Wait no. The cycle is: chord (0, N-length) + Path 2.
            # Path 2 has N-length vertices from N-length to 0 going the long way:
            # N-length, N-length+1, ..., N-1, 0. That's N-length+1 vertices.
            # So the cycle has N-length+1 vertices, which equals N - length + 1.
            # That's not what we want.
            #
            # Let me reconsider. For L > 20, I want a cycle of length L.
            # Chord (0, N-L) where N-L < 20.
            # Path from 0 to N-L going forward on rim: 0,1,...,N-L (N-L+1 vertices, N-L edges)
            # Cycle = chord + this path: length = N-L + 1.
            # For this to equal L: N-L+1 = L => 2L = N+1 => L = 20.5. Not integer.
            #
            # Path from 0 to N-L going backward on rim: 0, N-1, N-2, ..., N-L
            # That's N-L+1 vertices, N-L edges. Same thing.
            #
            # Hmm, so chord (0,k) always gives cycles of length k+1 and N-k+1.
            # For k=3: cycles of length 4 and 38.
            # For k=4: cycles of length 5 and 37.
            # ...
            # For k=20: cycles of length 21 and 21.
            #
            # So with chords (0,k) for k=3..20, I get cycles of lengths:
            # 4,38, 5,37, 6,36, 7,35, 8,34, 9,33, 10,32, 11,31, 12,30, 13,29, 14,28, 15,27, 16,26, 17,25, 18,24, 19,23, 20,22, 21,21
            # Plus the rim gives length 40.
            # 
            # Missing: 3! I need a triangle.
            # 
            # I need a chord that gives a triangle. Chord (0,k) gives triangle if k=2 or k=N-2=38.
            # So I need chord (0,2) or (0,38) to get a triangle.
            #
            # If I add chord (0,2): gives cycle of length 3 (0,1,2,0) and length 39 (0,1,2,3,...,39,0... wait no)
            # Chord (0,2): Path 0->1->2 (2 edges), cycle length 3. 
            # Other path: 0->39->38->...->2 (38 edges), cycle length 39.
            # 
            # So I need chords for k=2,3,4,...,20. That's 19 chords.
            # Total edges: 40 + 19 = 59. Value = 19. Score = 37/19 ≈ 1.947.
            
            # So the correct construction needs chords k=2..20 (19 chords).
            # Let me redo this properly.
            pass
        
        cycles.append(cyc)
    
    # I need to redo this. Let me restart the cycle construction properly.
    # The issue is that chord (0,k) gives cycles of length k+1 and N-k+1.
    # To get all lengths 3..40, I need:
    # - Length 40: rim
    # - Length 3: chord (0,2) -> cycle 0,1,2,0
    # - Length 4: chord (0,3) -> cycle 0,1,2,3,0
    # - Length 5: chord (0,4) -> cycle 0,1,2,3,4,0
    # - ...
    # - Length 21: chord (0,20) -> cycle 0,1,...,20,0 (21 vertices)
    # - Length 22: chord (0,19) -> cycle 0,1,...,19,0 is length 20. No.
    #   Chord (0,19): cycle length 20 (forward) or 41-19=22 (backward).
    #   Backward: 0,39,38,...,19,0. That's 22 vertices. Yes!
    # - Length 23: chord (0,18) -> backward: 0,39,...,18,0. 23 vertices.
    # - ...
    # - Length 39: chord (0,2) -> backward: 0,39,38,...,2,0. 39 vertices.
    
    # So chords k=2..20 (19 chords) cover lengths 3..40.
    # For length L:
    #   If L <= 21: use chord (0, L-1), cycle = 0,1,...,L-1,0 (forward path)
    #   If L >= 22: use chord (0, N-L), cycle = 0, N-1, N-2, ..., N-L, 0 (backward path)
    #     Check: N-L = 40-L. For L=22, N-L=18. Cycle: 0,39,38,...,18,0. 
    #     Vertices: 0,39,38,...,18. Count = 40-18 = 22. Yes!
    
    edges = []
    
    # Rim of the 40-cycle
    for i in range(N):
        edges.append((i, (i + 1) % N))
    
    # Chords: (0, k) for k = 2, 3, ..., 20
    for k in range(2, 21):
        edges.append((0, k))
    
    # Verify no duplicate edges
    edge_set = set()
    unique_edges = []
    for e in edges:
        key = (min(e[0], e[1]), max(e[0], e[1]))
        if key not in edge_set:
            edge_set.add(key)
            unique_edges.append(e)
    
    edges = unique_edges
    
    # Build cycles for lengths 3 through 40
    cycles = []
    
    for length in range(3, N + 1):
        if length == N:
            # Full Hamiltonian cycle
            cyc = list(range(N))
        elif length <= 21:
            # Use chord (0, length-1): forward cycle 0,1,...,length-1,0
            # Vertices: 0, 1, 2, ..., length-1
            cyc = list(range(length))
        else:
            # Use chord (0, N-length): backward cycle 0, N-1, N-2, ..., N-length, 0
            # Vertices: 0, N-1, N-2, ..., N-length
            k = N - length
            # Vertices from N-1 down to k, plus 0 at start
            # 0, 39, 38, ..., k
            cyc = [0] + list(range(N - 1, k - 1, -1))
        
        cycles.append(cyc)
    
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END

