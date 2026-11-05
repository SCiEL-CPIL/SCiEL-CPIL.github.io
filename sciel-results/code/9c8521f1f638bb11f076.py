N = 40

# EVOLVE-BLOCK-START
def construct():
    # Ladder-based pancyclic construction.
    # Vertices: top path 0..19, bottom path 20..39.
    # Edges:
    #   Top path:   (i, i+1) for i = 0..18       -> 19 edges
    #   Bottom path:(i+20, i+21) for i = 0..18   -> 19 edges
    #   Rungs:      (i, i+20) for i = 0..19      -> 20 edges
    #   Chord:      (0, 21)                       -> 1 edge
    # Total: 59 edges.
    #
    # Cycle construction:
    #   For even length k = 2m+2 (m = 1..19):
    #     0 -> 1 -> ... -> m -> (m+20) -> (m+19) -> ... -> 20 -> 0
    #     Uses top path 0..m, rung m, bottom path (m+20)..20, rung 0.
    #     Length: (m+1) + 1 + (m+1) = 2m+2.  k ranges 4..40.
    #
    #   For odd length k = 2m+1 (m = 1..19):
    #     0 -> 1 -> ... -> m -> (m+20) -> (m+19) -> ... -> 21 -> 0
    #     Uses top path 0..m, rung m, bottom path (m+20)..21, chord (21,0).
    #     Length: (m+1) + 1 + (m) + 1 = 2m+2? Let me recount.
    #     Vertices: 0,1,...,m (that's m+1 vertices), then m+20, m+19, ..., 21 (that's m vertices).
    #     Total vertices: m+1 + m = 2m+1. Correct.
    #     k ranges 3..39.
    #
    # Together: k = 3..40 all covered.
    # Score: 77 / 59 ≈ 1.305... wait, seed is 77 edges? 
    # Actually the score formula is seed_edges / current_edges.
    # Seed had 59 edges (40 cycle + 19 chords). So seed score = 59/59 = 1.0? 
    # But the prompt says current score is 1.9474. Let me re-read.
    # "Minimize edge count minus 40." So the value is (edges - 40).
    # "The evaluator measures the raw value (lower is better). The search score is seed/value"
    # Seed value: seed had 59 edges, so value = 59 - 40 = 19.
    # Wait, the current program in the prompt has 59 edges and score 1.9474.
    # 1.9474 = seed_value / current_value? No, "seed/value".
    # If score = seed_edges / current_edges, and seed was 77 edges (fan graph), then 77/59 = 1.305.
    # But the prompt says score 1.9474. 
    # Let's check the ladder: 59 edges. Value = 59 - 40 = 19.
    # If seed value was 77 - 40 = 37, then 37/19 = 1.947... Yes!
    # So the seed program in the prompt is the fan graph with 77 edges.
    # My ladder has 59 edges, value 19. Score = 37/19 ≈ 1.947.
    #
    # Can we do better than 59 edges?
    # A tree has 39 edges. A unicyclic graph has 40 edges.
    # We need cycles of length 3..40.
    # If we use a Hamiltonian cycle (40 edges), we have cycle 40.
    # We need 3..39.
    # Each chord (u,v) on a cycle of length N creates two cycles of lengths d+1 and N-d+1.
    # To cover 3..39, we need chords.
    # With 19 chords, we can cover 38 lengths (2 per chord), but we only need 37 lengths (3..39).
    # So 19 chords might suffice, giving 40+19=59 edges.
    # Can we do it with 18 chords? 18 chords give 36 cycle lengths. We need 37.
    # Unless some chords create more than 2 cycles, or some cycles are shared.
    # But in a simple cycle with chords, each chord creates exactly 2 fundamental cycles.
    # However, combinations of chords can create more cycles.
    # But to guarantee specific lengths, it's hard.
    #
    # What about a different structure?
    # A "double fan" or "theta" graph?
    # Two hubs connected by multiple disjoint paths.
    # If we have two hubs 0 and 39, connected by 3 paths of lengths a, b, c.
    # Cycles are formed by pairs of paths: a+b, b+c, a+c.
    # This gives only 3 cycle lengths. Not enough.
    #
    # What about a "ladder" with fewer rungs?
    # If we have a ladder with 20 rungs, we have 59 edges.
    # If we remove some rungs, can we still get all cycles?
    # Removing a rung breaks some cycles.
    #
    # Let's try to reduce edges by using a different backbone.
    # Consider a path 0-1-...-39 (39 edges).
    # Add chords to create cycles.
    # Chord (u,v) creates cycle of length v-u+1.
    # To get all lengths 3..40, we need chords for each length.
    # Length 40: need chord (0,39) -> cycle 0-1-...-39-0. But (0,39) is not in path.
    # Actually, path is 0-1-...-39. Chord (0,39) makes a cycle of length 40.
    # Chord (0,2) makes cycle 0-1-2-0, length 3.
    # Chord (0,3) makes cycle 0-1-2-3-0, length 4.
    # ...
    # Chord (0,39) makes cycle length 40.
    # So chords (0,k) for k=2..39 give cycles of length 3..40.
    # That's 38 chords. Total edges: 39 + 38 = 77.
    # This is the fan graph.
    #
    # To reduce edges, we need chords to create multiple cycle lengths.
    # In the ladder, each "step" creates both an even and an odd cycle.
    # The ladder is efficient because it reuses edges.
    #
    # Can we make a more efficient structure?
    # What about a "grid" 2x20? That's the ladder.
    # What about a "Möbius ladder"? That's a cycle with chords connecting opposite vertices.
    # For N=40, a Möbius ladder has 40 cycle edges + 20 chords = 60 edges.
    # Cycles: the main cycle (40), and chords create cycles of length 20+1=21? No.
    # Chord (i, i+20) creates two cycles of length 21.
    # So only length 21 and 40. Not pancyclic.
    #
    # So the ladder seems good. 59 edges.
    # Can we save 1 edge?
    # If we remove one rung, say (19, 39), do we lose a cycle?
    # The cycle of length 40 uses all rungs? No, the Hamiltonian cycle in the ladder is 0-1-...-19-39-38-...-20-0.
    # This uses rung 19 and rung 0.
    # If we remove rung 19, we can't form this specific Hamiltonian cycle.
    # But maybe another Hamiltonian cycle exists?
    # Without rung 19, vertex 19 is only connected to 18 and 19 (no, 19 is connected to 18 and 19? No, 19 is connected to 18 and 39 via rung? No, rung is (19,39)).
    # If we remove rung (19,39), vertex 19 is connected to 18 only? No, 19 is connected to 18 and... wait, top path is 0-1-...-19. So 19 is connected to 18. And rung (19,39) connects to 39.
    # If we remove rung (19,39), 19 is a leaf. Then no cycle can include 19. So we can't have a Hamiltonian cycle.
    # So we can't remove rungs.
    #
    # What if we change the structure?
    # Consider a "broom" or "star" with a cycle.
    # No, the ladder is quite sparse.
    #
    # Let's stick with the ladder. 59 edges.
    # Value = 59 - 40 = 19.
    # Score = 37 / 19 ≈ 1.947.
    # This matches the current score. So the ladder is not an improvement over the fan?
    # Wait, the current program in the prompt is the fan graph with 59 edges?
    # No, the current program in the prompt is:
    # Backbone cycle 0-1-...-39-0 (40 edges)
    # Chords from 0 to k for k in 2..20 (19 edges)
    # Total 59 edges.
    # So the current program IS the 59-edge construction.
    # And the score is 1.9474.
    #
    # So I need to find a construction with FEWER than 59 edges.
    # Can we do 58 edges?
    #
    # Let's think about the lower bound.
    # A graph with n vertices and m edges has cyclomatic number m - n + 1.
    # For n=40, m=59, cyclomatic number = 59 - 40 + 1 = 20.
    # We need cycles of length 3..40, which is 38 cycles.
    # But a graph can have many more cycles than its cyclomatic number.
    #
    # What if we use a different topology?
    # Consider two cycles sharing a path.
    # Or a "theta" graph with more paths.
    #
    # Actually, the minimum number of edges for a pancyclic graph on n vertices is known to be around n + sqrt(n) or similar?
    # For n=40, 59 is n + 19.
    #
    # Let's try to construct a graph with 58 edges.
    # We need to save 1 edge.
    #
    # Idea: Use a cycle of length 40, and add 18 chords.
    # 18 chords can create up to 36 cycle lengths.
    # We need 37 lengths (3..39).
    # So we need at least one chord to create 3 cycle lengths, or some cycles to be shared.
    #
    # In the current construction (cycle + 19 chords from 0), each chord (0,k) creates cycles of length k+1 and 41-k.
    # The set of lengths is {3,4,...,20,21,...,39}.
    # Note that 21 is created by both chord (0,19) and (0,20).
    # So we have 19 chords, creating 38 cycle lengths, but 21 is duplicated.
    # So we have 37 distinct lengths.
    #
    # Can we remove one chord and still have all lengths?
    # If we remove chord (0,20), we lose length 21 (from this chord) and length 20 (from this chord).
    # But length 20 is created by chord (0,19) as well? No, chord (0,19) creates 20 and 22.
    # So length 20 is only from (0,19).
    # Length 21 is from (0,19) and (0,20).
    # So if we remove (0,20), we lose length 20? No, (0,19) gives 20.
    # Wait, chord (0,k) gives lengths k+1 and 41-k.
    # (0,19) -> 20, 22.
    # (0,20) -> 21, 21.
    # So if we remove (0,20), we lose length 21.
    # But is 21 created by any other chord?
    # (0,19) gives 20, 22.
    # (0,18) gives 19, 23.
    # So 21 is only from (0,20).
    # So we can't remove (0,20).
    #
    # What if we change the chords?
    # Instead of all chords from 0, use a mix.
    #
    # Let's try a different approach: use a "double cycle" or "figure eight".
    # Two cycles of length 20 sharing one vertex.
    # Total vertices: 20 + 20 - 1 = 39. We need 40.
    # So two cycles of length 20 sharing one vertex, plus one extra vertex.
    # Edges: 20 + 20 = 40. Plus 1 edge to connect the extra vertex. Total 41.
    # But we need cycles of length 3..40.
    # This structure likely doesn't have all cycle lengths.
    #
    # I think 59 edges might be optimal or near-optimal.
    # But the task asks to minimize.
    #
    # Let's try to implement the ladder construction, which also has 59 edges, but maybe it's more robust?
    # No, same score.
    #
    # Wait, the current program in the prompt is the fan graph.
    # The inspiration 1 is the ladder, which also has 59 edges.
    # So both have the same score.
    #
    # I need to find a construction with 58 or fewer edges.
    #
    # Let's try to search for a graph with 58 edges.
    # We can use a heuristic search.
    #
    # Start with a random graph with 58 edges and check if it's pancyclic.
    # But that's expensive.
    #
    # Alternatively, can we modify the fan graph to save 1 edge?
    # The fan graph has chords (0,2)..(0,20).
    # What if we replace two chords with one chord that creates the same lengths?
    # Unlikely.
    #
    # What if we use a different hub?
    #
    # I'll stick with the 59-edge construction for now, but implement the ladder version as it might be more "structurally different" and the prompt asks for a structural change.
    #
    # Actually, the prompt says "Make a STRUCTURAL change".
    # The ladder is a different structure from the fan.
    # Even if it has the same edge count, it's a different design.
    #
    # But the score will be the same.
    #
    # Let's try to see if we can get 58 edges.
    #
    # Consider a cycle of length 40.
    # Add chords (0,2), (0,4), ..., (0,38). That's 19 chords. Total 59.
    # This creates cycles of length 3,5,...,39 and 39,37,...,3.
    # So only odd lengths.
    #
    # Add chords (1,3), (1,5), ..., (1,37). That's 18 chords.
    # This creates cycles of length 3,5,...,37.
    #
    # If we use both sets, we get all lengths?
    # No, we need to be careful.
    #
    # I think 59 is hard to beat.
    #
    # Let's output the ladder construction.
    edges = []
    
    # Top path
    for i in range(19):
        edges.append((i, i+1))
    
    # Bottom path
    for i in range(19):
        edges.append((i+20, i+21))
    
    # Rungs
    for i in range(20):
        edges.append((i, i+20))
    
    # Extra chord to create triangle
    edges.append((0, 21))
    
    # Generate cycles
    cycles = []
    
    for k in range(3, 41):
        if k % 2 == 1:
            # Odd cycle
            m = (k - 1) // 2
            cyc = list(range(0, m+1))
            cyc.extend(range(m+20, 20, -1))
            cycles.append(cyc)
        else:
            # Even cycle
            m = k // 2 - 1
            cyc = list(range(0, m+1))
            cyc.extend(range(m+20, 19, -1))
            cycles.append(cyc)
    
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
