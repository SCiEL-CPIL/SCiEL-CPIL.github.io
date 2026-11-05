"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    # Strategy: 
    # 1. Start with a path graph 0-1-...-39 (39 edges).
    # 2. We need to add chords (u, v) with u < v such that the graph is pancyclic.
    # 3. A chord (u, v) creates a cycle of length L = v - u + 1 using the path segment.
    # 4. If we restrict chords to be "non-crossing" (i.e., if (u1, v1) and (u2, v2) are chords, 
    #    they don't interleave as u1 < u2 < v1 < v2), the structure is a tree of intervals.
    # 5. However, even with non-crossing chords, we can get more cycles.
    #    But the simplest guarantee is: each chord (u, v) gives cycle length v-u+1.
    #    If we want to minimize edges, we want to cover all lengths 3..40 with the fewest chords.
    #    
    #    Wait, the previous program used a Hamiltonian Cycle (40 edges) + 19 chords = 59 edges.
    #    The path has 39 edges.
    #    If we use a path, we need chords to create ALL cycles 3..40.
    #    The path itself has no cycles.
    #    So we need chords to create cycles of length 3, 4, ..., 40.
    #    
    #    Can we do it with fewer than 19 chords?
    #    If a chord (u, v) creates a cycle of length L, it uses the path segment u..v.
    #    If we have another chord (u', v') inside u..v, it creates a smaller cycle.
    #    But we need ALL lengths.
    #    
    #    Let's try a DP approach to find the minimum number of chords needed for a path 
    #    to be pancyclic, assuming chords only interact via nesting.
    #    Actually, if chords are nested, a chord (u, v) creates a cycle of length v-u+1.
    #    The subgraph inside (u, v) is independent in terms of creating cycles that stay within u..v.
    #    But we need cycles of all lengths 3..N.
    #    The "outer" cycle of the whole path is not a cycle unless we add a chord (0, N-1) or similar.
    #    If we add chord (0, 39), we get a cycle of length 40.
    #    Then inside (0, 39), we need cycles of length 3..39.
    #    This is a recursive structure.
    #    
    #    Let f(L) be the minimum number of chords needed to make a path of length L (L-1 edges, L vertices) 
    #    pancyclic (cycles 3..L).
    #    To get a cycle of length L, we must add a chord (0, L-1) (or similar spanning chord).
    #    This adds 1 chord and creates a cycle of length L.
    #    The interior of this cycle is a path of length L-2 (vertices 1..L-2).
    #    We need cycles of length 3..L-1 in the whole graph.
    #    Any cycle entirely within the interior path of length L-2 has length <= L-2.
    #    Any cycle using the chord (0, L-1) has length L.
    #    What about cycles of length L-1? 
    #    A cycle of length L-1 cannot use the chord (0, L-1) because that would require a path of length L-2 between 0 and L-1, 
    #    but the only path is the full path of length L-1. So a cycle using (0, L-1) must use the full path, giving length L.
    #    Therefore, cycles of length < L must be formed within the subgraphs.
    #    
    #    This implies that simply adding a spanning chord doesn't help generate intermediate lengths efficiently 
    #    unless the interior is also pancyclic.
    #    
    #    If we make the interior (path 1..L-2) pancyclic for lengths 3..L-2, we still miss L-1.
    #    To get L-1, we might need a chord like (0, L-2) or (1, L-1).
    #    
    #    Let's check the known result: 
    #    A path of N vertices needs at least N-2 chords to be pancyclic? 
    #    If so, 39 + 38 = 77 edges. This is worse than 59.
    #    
    #    The Hamiltonian Cycle approach (59 edges) is much better because the base cycle provides length N 
    #    and chords split it into two cycles, covering two lengths per chord.
    #    
    #    Is there a way to get below 59 edges?
    #    Lower bound: A pancyclic graph on N vertices must have at least N edges (Hamiltonian cycle).
    #    Actually, a graph with N edges is a cycle. A cycle is only pancyclic if N=3 (triangle). 
    #    For N>3, a simple cycle only has one cycle length. 
    #    So we need more edges.
    #    
    #    Let's look at the "Double Ring" inspiration. 
    #    Two 20-cycles connected by 20 rungs. 
    #    Edges: 20 + 20 + 20 = 60. 
    #    This is 60, which is worse than 59.
    #    
    #    What about a "Ladder" graph? 
    #    A ladder with 20 rungs (40 vertices) has 2*20 + 19 = 59 edges? 
    #    Ladder graph P_2 x P_20. 
    #    Vertices: (0,0)..(0,19) and (1,0)..(1,19).
    #    Edges: 
    #    - Rungs: (0,i)-(1,i) for i=0..19 (20 edges)
    #    - Rails: (0,i)-(0,i+1) and (1,i)-(1,i+1) for i=0..18 (38 edges)
    #    Total: 58 edges.
    #    
    #    Is the Ladder graph P_2 x P_20 pancyclic?
    #    It contains cycles of length 4 (squares).
    #    Does it contain cycles of all lengths 3..40?
    #    Ladder graphs are bipartite, so they only contain even-length cycles.
    #    So P_2 x P_20 is NOT pancyclic (no triangles, no 5-cycles, etc.).
    #    
    #    So we need an odd cycle. 
    #    If we add a chord to break bipartiteness, we might get more cycles.
    #    
    #    Let's stick to the current best known: 59 edges.
    #    Can we improve it?
    #    
    #    Consider a graph with a Hamiltonian cycle and chords.
    #    Current: 40 + 19 = 59.
    #    
    #    What if we use a different backbone?
    #    A "Theta" graph?
    #    
    #    Let's try a local search to see if we can find a 58-edge pancyclic graph.
    #    Start with the 59-edge graph.
    #    Try to remove an edge and see if it's still pancyclic.
    #    If yes, try to remove another.
    #    
    #    However, removing an edge from a pancyclic graph often breaks pancyclicity.
    #    
    #    Let's try to construct a 58-edge graph directly.
    #    If we have 58 edges, we have 18 "extra" edges beyond the spanning tree (39 edges).
    #    Or 18 extra edges beyond the Hamiltonian cycle (40 edges) -> 40 + 18 = 58.
    #    So we need a Hamiltonian cycle + 18 chords to be pancyclic.
    #    
    #    With 18 chords, we can cover at most 18*2 = 36 cycle lengths (plus the base 40).
    #    Total lengths covered: 1 (base) + 36 = 37.
    #    We need 38 lengths (3..40).
    #    So we are short by 1 length.
    #    Unless some chord creates 3 cycles, or two chords interact to create a new length.
    #    
    #    If we can find a configuration where 18 chords cover 37 lengths (including 40), we win.
    #    Or if 17 chords cover 37 lengths.
    #    
    #    Let's try to optimize the chord placement on the Hamiltonian cycle.
    #    We want to cover {3, ..., 39} using chords (0, k) or similar.
    #    The current strategy uses chords (0, 2)..(0, 20).
    #    This covers 3..39 perfectly.
    #    Can we do it with 18 chords?
    #    
    #    If we drop one chord, say (0, 20), we lose lengths 21 and 20 (from that chord).
    #    But 20 might be covered by another chord?
    #    Chord (0, 19) covers 20 and 22.
    #    So if we have (0, 19), we have 20.
    #    If we drop (0, 20), we lose 21?
    #    Chord (0, 19) covers 22. Chord (0, 21) covers 22 and 20.
    #    Wait, the set of chords was (0, 2)..(0, 20).
    #    Lengths covered:
    #    k=2: 3, 39
    #    k=3: 4, 38
    #    ...
    #    k=19: 20, 22
    #    k=20: 21, 21
    #    
    #    If we remove k=20, we lose 21.
    #    Is 21 covered by any other chord?
    #    k=19 gives 20, 22.
    #    k=21 (if we added it) gives 22, 20.
    #    No other chord gives 21.
    #    So we need k=20 or k=21 or some other combination to get 21.
    #    
    #    What if we change the chords?
    #    Instead of all starting at 0, can we use different starts?
    #    If we use chords (0, 2), (1, 4), (2, 6)... 
    #    This might allow better coverage.
    #    
    #    Let's try a greedy/random search for a set of 18 chords that covers all lengths 3..39.
    #    
    edges = []
    # Base Hamiltonian cycle
    for i in range(N):
        edges.append((i, (i + 1) % N))
    
    # We need to find 18 chords such that the set of cycle lengths generated covers 3..39.
    # A chord (u, v) generates lengths d+1 and N-d+1 where d is the distance along the cycle.
    
    # Let's try a simple heuristic:
    # Keep the current 19 chords and try to remove one if possible.
    # Or try to find a 18-chord solution.
    
    # For now, let's stick to the known 59-edge solution to ensure validity, 
    # but I will add a quick check to see if we can find a 58-edge solution within the time limit.
    # Since the search space is large, a simple local search might not find it quickly.
    # I will return the 59-edge solution as it is robust.
    
    chords = [(0, k) for k in range(2, 21)]
    for c in chords:
        edges.append(c)
        
    edges = list(set(edges))
    
    # Construct cycles
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cycles.append(list(range(N)))
        else:
            found = False
            k1 = L - 1
            if 2 <= k1 <= 20:
                cycle = list(range(0, k1 + 1))
                cycles.append(cycle)
                found = True
            
            if not found:
                k2 = N - L + 1
                if 2 <= k2 <= 20:
                    cycle = [0] + list(range(k2, N))
                    cycles.append(cycle)
                    found = True
            
            if not found:
                raise ValueError(f"Could not find cycle for length {L}")
        
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
