"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 40

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sparse pancyclic graph on N=40 vertices.
    Strategy: Use a path backbone 0-1-...-39 (39 edges) plus back-edges from vertex 0.
    A back-edge (0, d) creates a cycle of length d+1: 0-1-2-...-d-0.
    We need cycles of lengths 3..40.
    - Length 40: need a Hamiltonian cycle. Add edge (0,39) to close the path into a cycle.
    - Lengths 3..39: back-edges (0,2) through (0,38) give lengths 3..39.
    But that's 37 back-edges + 39 path edges = 76. Too many.
    
    Better: Use path 0-1-...-39 (39 edges) + edge (0,39) for Hamiltonian cycle (40 edges total).
    Then back-edges create shorter cycles. Back-edge (0,d) with d<39 creates cycle
    0-1-...-d-0 of length d+1. But we also get cycle 0-d-(d+1)-...-39-0 of length N-d+1.
    Wait, that uses the Hamiltonian cycle edge (39,0) and path edges.
    
    Actually with Hamiltonian cycle 0-1-...-39-0:
    Chord (0,d): cycle 0-1-...-d-0 (length d+1) and cycle 0-39-38-...-d-0 (length N-d+1).
    Same as before. Need 19 chords for d=2..20. Total 59 edges.
    
    New structural idea: Use TWO hubs. Split vertices into two groups.
    Group A: 0..19, Group B: 20..39.
    Path in A: 0-1-...-19 (19 edges)
    Path in B: 20-21-...-39 (19 edges)
    Connect: (19,20) (1 edge)
    Total path: 0-1-...-39 (39 edges)
    
    Now add back-edges strategically from both hubs.
    From vertex 0: (0,2),(0,3),...,(0,19) -> 18 edges, gives cycles of length 3..20
    From vertex 39: (39,37),(39,36),...,(39,20) -> 18 edges, gives cycles of length 3..20 (in B)
    But we need lengths 21..39 too.
    
    Let me try: path 0-1-...-39 (39 edges) + (0,39) for Hamiltonian (40 edges).
    Back-edges: (0,2)..(0,19) = 18 edges for lengths 3..20.
    For lengths 21..39: use back-edges from other vertices.
    (19,0) is already there? No, (0,19) is a back-edge.
    
    For length 21: need a 21-cycle. With Hamiltonian cycle, chord (0,20) gives length 21.
    So add (0,20). Now lengths 3..21 covered with 19 back-edges. Total: 40+19=59.
    
    For lengths 22..39: chord (0,19) gives long cycle length 40-19+1=22.
    chord (0,18) gives long cycle length 23.
    ...
    chord (0,2) gives long cycle length 39.
    These are ALREADY in our set! (0,2)..(0,19) are all added.
    
    So with Hamiltonian cycle + chords (0,2)..(0,20), we have:
    - Short cycles: lengths 3..21 from chords (0,2)..(0,20)
    - Long cycles: lengths 22..39 from chords (0,19)..(0,2)
    - Length 40: Hamiltonian cycle
    Total: 40 + 19 = 59 edges. Score: 19.
    
    Can we reduce? What if we DON'T use a Hamiltonian cycle as base?
    
    Try: path 0-1-...-39 (39 edges) + back-edges.
    Back-edge (0,d) creates cycle 0-1-...-d-0 of length d+1.
    For length 40, we need a Hamiltonian cycle, so we need edge (0,39) or equivalent.
    Adding (0,39) makes it a Hamiltonian cycle (40 edges).
    
    Alternative for length 40 without (0,39): use a different Hamiltonian cycle.
    But any Hamiltonian cycle on 40 vertices needs 40 edges.
    
    So minimum is at least 40 edges for the Hamiltonian cycle, plus chords for other lengths.
    
    Can we share? If the Hamiltonian cycle is not 0-1-...-39-0 but something else,
    maybe fewer chords are needed.
    
    Key insight: each chord on a Hamiltonian cycle covers at most 2 cycle lengths.
    We need 37 other lengths (3..39). So we need at least ceil(37/2) = 19 chords.
    Total: 40 + 19 = 59 edges. This is a lower bound IF we use a Hamiltonian cycle base.
    
    But what if we don't use a Hamiltonian cycle? What if the length-40 cycle uses
    edges that are also part of other cycles?
    
    Actually, the length-40 cycle IS a Hamiltonian cycle, which uses 40 edges.
    Those 40 edges form the base. Additional chords create shorter cycles.
    Each chord can create at most 2 new cycle lengths (on a Hamiltonian cycle).
    So 19 chords minimum. Total 59.
    
    UNLESS: a chord creates a cycle that's not just "chord + arc". E.g., two chords
    together can create a cycle. This could reduce the count.
    
    Let me try: can 2 chords create 3 cycle lengths?
    Chord (0,a) and chord (0,b) with a<b.
    Cycle 1: 0-1-...-a-0 (length a+1)
    Cycle 2: 0-a-(a+1)-...-b-0 (length b-a+1) -- uses chord (0,a), path a to b, chord (0,b)
    Cycle 3: 0-b-(b+1)-...-39-0 (length 40-b+1)
    Cycle 4: 0-39-...-a-0 (length 40-a+1)
    So 2 chords can create up to 4 cycle lengths!
    
    With chords (0,2) and (0,3):
    Length 3: 0-1-2-0
    Length 2: 0-2-3-0? No, length 2 is not a simple cycle.
    Actually cycle 0-2-3-0 has vertices {0,2,3}, length 3.
    Length 4: 0-1-2-3-0
    Length 38: 0-39-38-...-3-0
    Length 39: 0-39-38-...-2-0
    
    Hmm, I'm getting confused. Let me just implement the 59-edge solution cleanly
    and verify it works, since that's what the current code does.
    
    Actually, let me try to get 58 edges. If I can find a way to cover all 37 lengths
    with only 18 chords (some chords covering more than 2 lengths via combinations),
    I'd get 58 edges.
    
    Let me try a different chord placement. Instead of all from vertex 0, use
    chords that form a "ladder" pattern.
    
    Chords: (0,2), (1,3), (2,4), ..., (18,20) -- 19 chords forming a "shifted" pattern.
    Each chord (i, i+2) creates a triangle: i, i+1, i+2.
    That gives us length-3 cycles. But we need all lengths.
    
    This is getting complicated. Let me just go with the clean 59-edge solution
    and make sure it's correct and fast.
    """
    deadline = time.time() + 50  # safety margin
    
    # Base: Hamiltonian cycle 0-1-2-...-39-0
    base_edges = set()
    for i in range(N):
        j = (i + 1) % N
        base_edges.add((min(i, j), max(i, j)))
    
    # Chords from vertex 0 to vertices 2..20 (19 chords)
    chords = [(0, d) for d in range(2, 21)]
    
    all_edges = list(base_edges) + chords
    # Normalize
    all_edges = [(min(u, v), max(u, v)) for u, v in all_edges]
    edge_set = set(all_edges)
    
    # Construct cycles for each length 3..40
    cycles = []
    for k in range(3, N + 1):
        if k == N:
            # Full Hamiltonian cycle: 0,1,2,...,39
            cyc = list(range(N))
        elif k <= 21:
            # Short cycle: 0,1,2,...,k-1 (length k)
            # Uses path edges (0,1),(1,2),...,(k-2,k-1) and chord (0,k-1)
            d = k - 1
            cyc = list(range(0, d + 1))
        else:
            # Long cycle: 0,39,38,...,d where d = N-k+1
            # Length = N - d + 1 = k
            # Uses edges (0,39),(39,38),...,(d+1,d) and chord (d,0)
            d = N - k + 1
            cyc = [0] + list(range(N - 1, d - 1, -1))
        cycles.append(cyc)
    
    # Verify
    valid = True
    for cyc in cycles:
        for i in range(len(cyc)):
            u, v = cyc[i], cyc[(i + 1) % len(cyc)]
            if (min(u, v), max(u, v)) not in edge_set:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: complete graph on first k vertices for each cycle
        fallback_edges = set()
        fallback_cycles = []
        for k in range(3, N + 1):
            cyc = list(range(k))
            fallback_cycles.append(cyc)
            for i in range(k):
                u, v = cyc[i], cyc[(i + 1) % k]
                fallback_edges.add((min(u, v), max(u, v)))
        return {'edges': list(fallback_edges), 'cycles': fallback_cycles}
    
    return {'edges': all_edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
